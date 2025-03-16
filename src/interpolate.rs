use crate::convolve_rv::{
    convolve_rotation, shift_and_resample, shift_resample_and_add_binary_components, ArraySegment,
    WavelengthDispersion,
};
use crate::cubic::{
    calculate_interpolation_coefficients, calculate_interpolation_coefficients_linear, LocalGrid,
    LocalGridLinear,
};
use anyhow::{anyhow, bail, Context, Result};
use itertools::Itertools;
use nalgebra::{self as na};

pub type FluxFloat = f32; // Float type used for spectra

pub enum CowVector<'a> {
    Borrowed(na::DVectorView<'a, u16>),
    Owned(na::DVector<u16>),
}

impl<'a> CowVector<'a> {
    pub fn into_owned(self) -> na::DVector<u16> {
        match self {
            CowVector::Borrowed(view) => view.into_owned(),
            CowVector::Owned(vec) => vec,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            CowVector::Borrowed(view) => view.len(),
            CowVector::Owned(vec) => vec.len(),
        }
    }

    pub fn rows(&self, start: usize, len: usize) -> na::DVectorView<u16> {
        match self {
            CowVector::Borrowed(view) => view.rows(start, len),
            CowVector::Owned(vec) => vec.rows(start, len),
        }
    }

    pub fn fixed_rows<const N: usize>(
        &'a self,
        start: usize,
    ) -> na::Matrix<
        u16,
        na::Const<N>,
        na::Const<1>,
        na::ViewStorage<'a, u16, na::Const<N>, na::Const<1>, na::Const<1>, na::Dyn>,
    > {
        match self {
            CowVector::Borrowed(view) => view.fixed_rows::<N>(start),
            CowVector::Owned(vec) => vec.fixed_rows::<N>(start),
        }
    }
}

/// Represents the bounds of a 3D model grid.
pub trait GridBounds: Clone {
    fn limits(&self) -> (na::SVector<f64, 3>, na::SVector<f64, 3>);
    fn is_within_bounds(&self, param: na::SVector<f64, 3>) -> bool;
    fn get_limits_at(&self, param: na::SVector<f64, 3>, index: usize) -> Result<(f64, f64)>;
}

/// Range of values in one dimension of the grid
#[derive(Clone)]
pub struct Range {
    pub values: Vec<f64>,
}

impl Range {
    /// Get index of the gridpoint to the right of x, or the index of x if it is in the grid itself.
    pub fn get_right_index(&self, x: f64) -> Result<usize, usize> {
        self.values.binary_search_by(|v| v.partial_cmp(&x).unwrap())
    }

    /// Find the index of the gridpoint to the left of x, or None if x is out of bounds.
    /// If x is a gridpoint, return (index of x) - 1, unless x is the first gridpoint.
    pub fn find_left_neighbor_index(&self, x: f64, limits: (usize, usize)) -> Option<usize> {
        match self.get_right_index(x) {
            Ok(i) => {
                if i == limits.0 {
                    Some(limits.0)
                } else {
                    Some(i - 1)
                }
            }
            Err(i) => {
                if i == limits.0 {
                    None
                } else {
                    Some(i - 1)
                }
            }
        }
    }

    /// Find the indices of the four neighbors of x in the grid, for cubic interpolation.
    /// The limits argument will be used to determine whether the neighbors are within the grid.
    /// This is needed because the limits may be different at different points in the grid,
    /// i.e. different logg limits for different Teff values.
    /// When x is outside the limits, Err will be returned.
    /// When x is inside the limits, but near the bounds, the neighbors that fall outside will be set to None.
    /// In that case the interpolation will fall back to quadratic
    pub fn find_neighbors(&self, x: f64, limits: (usize, usize)) -> Result<[Option<usize>; 4]> {
        let (left, right) = limits;
        match self.get_right_index(x) {
            // In case the value matches a grid point exactly
            Ok(i) => {
                if i < left {
                    Err(anyhow!(
                        "Index {} out of left bound, limits={:?}, x={}",
                        i,
                        limits,
                        x
                    ))
                } else if i > right {
                    Err(anyhow!(
                        "Index {} out of right bound, limits={:?}, x={}",
                        i,
                        limits,
                        x
                    ))
                } else if i == left {
                    Ok([None, Some(i), Some(i + 1), Some(i + 2)])
                } else if i == right {
                    Ok([Some(i - 2), Some(i - 1), Some(i), None])
                } else if i == right - 1 {
                    Ok([Some(i - 2), Some(i - 1), Some(i), Some(i + 1)])
                } else {
                    Ok([Some(i - 1), Some(i), Some(i + 1), Some(i + 2)])
                }
            }
            // In case the value is between two grid points
            Err(i) => {
                if i <= left {
                    Err(anyhow!(
                        "Index {} out of left bound, limits={:?}, x={}",
                        i,
                        limits,
                        x
                    ))
                } else if i > right {
                    Err(anyhow!(
                        "Index {} out of right bound, limits={:?}, x={}",
                        i,
                        limits,
                        x
                    ))
                } else if i == left + 1 {
                    Ok([None, Some(i - 1), Some(i), Some(i + 1)])
                } else if i == right {
                    Ok([Some(i - 2), Some(i - 1), Some(i), None])
                } else {
                    Ok([Some(i - 2), Some(i - 1), Some(i), Some(i + 1)])
                }
            }
        }
    }

    /// Find the indices of the two neighbors of x in the grid, for linear interpolation.
    /// The limits argument will be used to determine whether the neighbors are within the grid.
    /// This is needed because the limits may be different at different points in the grid,
    /// i.e. different logg limits for different Teff values.
    /// `limits` contains the first and last index that is available.
    /// When x is outside the limits, Err will be returned.
    /// Unlike for cubic interpolation, the neighbors will always be within the grid.
    pub fn find_neighbors_linear(&self, x: f64, limits: (usize, usize)) -> Result<[usize; 2]> {
        let (left, right) = limits;
        match self.get_right_index(x) {
            // In case the value matches a grid point exactly
            Ok(i) => {
                if i < left {
                    Err(anyhow!(
                        "Index {} out of left bound, limits={:?}, x={}",
                        i,
                        limits,
                        x
                    ))
                } else if i > right {
                    Err(anyhow!(
                        "Index {} out of right bound, limits={:?}, x={}",
                        i,
                        limits,
                        x
                    ))
                } else if i == right {
                    Ok([i - 1, i])
                } else {
                    Ok([i, i + 1])
                }
            }
            // In case the value is between two grid points
            Err(i) => {
                if i <= left {
                    Err(anyhow!(
                        "Index {} out of left bound, limits={:?}, x={}",
                        i,
                        limits,
                        x
                    ))
                } else if i > right {
                    Err(anyhow!(
                        "Index {} out of right bound, limits={:?}, x={}",
                        i,
                        limits,
                        x
                    ))
                } else {
                    Ok([i - 1, i])
                }
            }
        }
    }

    /// Return the index of x in the grid, or None if x is not a gridpoint.
    pub fn try_get_index(&self, x: f64, limits: (usize, usize)) -> Option<usize> {
        let precision = 0.001;
        let i = match self.find_left_neighbor_index(x, limits) {
            Some(i) => i,
            None => return None,
        };
        let j = i + 1;
        if (self.values[i] - x).abs() < precision {
            Some(i)
        } else if (self.values[j] - x).abs() < precision {
            Some(j)
        } else {
            None
        }
    }

    /// Get the value at index i.
    pub fn get(&self, i: usize) -> Result<f64> {
        self.values
            .get(i)
            .ok_or_else(|| anyhow!("Index out of bounds ({})", i))
            .copied()
    }

    /// Number of gridpoints.
    pub fn n(&self) -> usize {
        self.values.len()
    }

    /// Get the first and last values of the grid
    pub fn get_first_and_last(&self) -> (f64, f64) {
        (self.values[0], *self.values.last().unwrap())
    }
}

/// Represents the bounds of our 3D model grid that has the same metallicy range everywhere,
/// but with different logg limits for different Teff values.
/// Currently the only implementor of GridBounds.
#[derive(Clone)]
pub struct Grid {
    pub teff: Range,
    pub m: Range,
    pub logg: Range,
    pub logg_limits: Vec<(usize, usize)>, // For every Teff value, the min and max logg index.
    // For every Teff value, the number of (teff, logg) pairs with lower Teff.
    pub cumulative_grid_size: Vec<usize>,
}

impl Grid {
    /// Create a new Grid from a list of (Teff, M, logg) tuples that are the gridpoints.
    pub fn new(model_labels: Vec<[f64; 3]>) -> Result<Self> {
        let mut teffs = Vec::new();
        let mut ms = Vec::new();
        let mut loggs = Vec::new();
        for [teff, m, logg] in model_labels.iter() {
            if !teffs.contains(teff) {
                let pos = teffs
                    .binary_search_by(|v| v.partial_cmp(teff).unwrap())
                    .unwrap_or_else(|x| x);
                teffs.insert(pos, *teff);
            }
            if !ms.contains(m) {
                let pos = ms
                    .binary_search_by(|v| v.partial_cmp(m).unwrap())
                    .unwrap_or_else(|x| x);
                ms.insert(pos, *m);
            }
            if !loggs.contains(logg) {
                let pos = loggs
                    .binary_search_by(|v| v.partial_cmp(logg).unwrap())
                    .unwrap_or_else(|x| x);
                loggs.insert(pos, *logg);
            }
        }

        let mut logg_limits = vec![(loggs.len() - 1, 0); teffs.len()];
        for [teff, _, logg] in model_labels {
            let i = teffs
                .binary_search_by(|v| v.partial_cmp(&teff).unwrap())
                .unwrap();
            let j = loggs
                .binary_search_by(|v| v.partial_cmp(&logg).unwrap())
                .unwrap();
            if j < logg_limits[i].0 {
                logg_limits[i].0 = j;
            }
            if j > logg_limits[i].1 {
                logg_limits[i].1 = j;
            }
        }
        // Sanity check
        for (left, right) in logg_limits.iter() {
            if left > right {
                bail!("Invalid logg limits");
            }
        }

        let cumulative_grid_size = logg_limits
            .iter()
            .scan(0, |acc, (left, right)| {
                let n = right - left + 1;
                let old = *acc;
                *acc += n;
                Some(old)
            })
            .collect();

        Ok(Grid {
            teff: Range { values: teffs },
            m: Range { values: ms },
            logg: Range { values: loggs },
            logg_limits,
            cumulative_grid_size,
        })
    }

    /// List all gridpoints as Teff, m, logg tuples.
    pub fn list_gridpoints(&self) -> Vec<[f64; 3]> {
        self.teff
            .values
            .iter()
            .enumerate()
            .flat_map(|(i, teff)| {
                let (left, right) = self.logg_limits[i];
                self.logg.values[left..=right]
                    .iter()
                    .map(move |logg| (teff, logg))
            })
            .cartesian_product(self.m.values.iter())
            .map(|((teff, logg), m)| [*teff, *m, *logg])
            .collect()
    }

    /// Check if the given Teff, logg pair is within the bounds of the grid.
    pub fn is_teff_logg_between_bounds(&self, teff: f64, logg: f64) -> bool {
        match self.teff.get_right_index(teff) {
            Ok(i) => {
                let bounds = self.logg_limits[i];
                logg >= self.logg.get(bounds.0).unwrap() && logg <= self.logg.get(bounds.1).unwrap()
            }
            Err(i) => {
                if i == 0 || i == self.teff.n() {
                    return false;
                }
                let bounds_left = self.logg_limits[i - 1];
                let bounds_right = self.logg_limits[i];
                logg >= self.logg.get(bounds_left.0).unwrap()
                    && logg <= self.logg.get(bounds_left.1).unwrap()
                    && logg >= self.logg.get(bounds_right.0).unwrap()
                    && logg <= self.logg.get(bounds_right.1).unwrap()
            }
        }
    }

    /// Check if the given M value is within the bounds of the grid
    pub fn is_m_between_bounds(&self, m: f64) -> bool {
        m >= self.m.values[0] && m <= *self.m.values.last().unwrap()
    }

    /// Get the index limits in logg at a certain teff.
    /// logg gridpoints exist both left and right of the teff value between the returned limits.
    pub fn get_logg_index_limits_at(&self, teff: f64) -> Result<(usize, usize)> {
        match self.teff.get_right_index(teff) {
            Ok(i) => {
                if i == self.teff.n() - 1 {
                    Ok(self.logg_limits[i])
                } else {
                    let left = self.logg_limits[i];
                    let right = self.logg_limits[i + 1];
                    Ok((left.0.max(right.0), left.1.min(right.1)))
                }
            }
            Err(i) => {
                if i == 0 || i == self.teff.n() {
                    bail!("Teff out of bounds");
                }
                let left = self.logg_limits[i - 1];
                let right = self.logg_limits[i];
                Ok((left.0.max(right.0), left.1.min(right.1)))
            }
        }
    }

    /// Get the index limits in teff at a certain logg.
    /// teff gridpoints exist both left and right of the logg value between the returned limits.
    pub fn get_teff_index_limits_at(&self, logg: f64) -> Result<(usize, usize)> {
        let min_index = self
            .logg_limits
            .iter()
            .position(|(left, right)| {
                logg >= self.logg.values[*left] && logg <= self.logg.values[*right]
            })
            .ok_or(anyhow!("No teff bounds at logg={}", logg))?;
        let max_index = self.logg_limits.len()
            - 1
            - self
                .logg_limits
                .iter()
                .rev()
                .position(|(left, right)| {
                    logg >= self.logg.values[*left] && logg <= self.logg.values[*right]
                })
                .ok_or(anyhow!("No teff bounds at logg={}", logg))?;
        Ok((min_index, max_index))
    }

    /// Get the grid of 64 neighboring gridpoints for a certain Teff, M, logg.
    pub fn get_local_grid(&self, teff: f64, m: f64, logg: f64) -> Result<LocalGrid> {
        if !self.is_teff_logg_between_bounds(teff, logg) {
            bail!("Teff, logg out of bounds ({}, {})", teff, logg);
        }
        if !self.is_m_between_bounds(m) {
            bail!("M out of bounds ({})", m);
        }
        let teff_limits = self.get_teff_index_limits_at(logg)?;
        let m_limits = (0, self.m.n() - 1);
        let logg_limits = self.get_logg_index_limits_at(teff)?;

        let teff_neighbor_indices =
            self.teff
                .find_neighbors(teff, teff_limits)
                .with_context(|| {
                    format!(
                        "failed getting neighbors for teff: {}, {:?}",
                        teff, teff_limits
                    )
                })?;
        let m_neighbor_indices = self.m.find_neighbors(m, m_limits).with_context(|| {
            format!(
                "failed getting neighbors for m: {}, {:?}",
                logg, logg_limits
            )
        })?;
        let logg_neighbor_indices =
            self.logg
                .find_neighbors(logg, logg_limits)
                .with_context(|| {
                    format!(
                        "failed getting neighbors for logg: {}, {:?}",
                        logg, logg_limits
                    )
                })?;

        let teff_logg_indices = na::SMatrix::from_columns(&teff_neighbor_indices.map(
            |teff_neighbor| match teff_neighbor {
                Some(i) => {
                    let (left, right) = self.logg_limits[i];
                    logg_neighbor_indices
                        .map(|k| match k {
                            Some(k) => {
                                if k >= left && k <= right {
                                    Some((i, k))
                                } else {
                                    None
                                }
                            }
                            None => None,
                        })
                        .into()
                }
                None => [None; 4].into(),
            },
        ));

        Ok(LocalGrid {
            teff: (
                teff,
                teff_neighbor_indices.map(|x| {
                    x.map(|x| {
                        self.teff
                            .get(x)
                            .with_context(|| {
                                format!(
                                    "teff={}, neighbors: {:?}, limits: {:?}",
                                    teff, teff_neighbor_indices, teff_limits
                                )
                            })
                            .unwrap()
                    })
                }),
            ),
            logg: (
                logg,
                logg_neighbor_indices.map(|x| {
                    x.map(|x| {
                        self.logg
                            .get(x)
                            .with_context(|| {
                                format!(
                                    "logg={}, neighbors: {:?}, limits: {:?}",
                                    logg, logg_neighbor_indices, logg_limits
                                )
                            })
                            .unwrap()
                    })
                }),
            ),
            teff_logg_indices,
            m: (
                m,
                m_neighbor_indices.map(|x| {
                    x.map(|x| {
                        self.m
                            .get(x)
                            .with_context(|| {
                                format!(
                                    "m={}, neighbors: {:?}, limits: {:?}",
                                    m, m_neighbor_indices, m_limits
                                )
                            })
                            .unwrap()
                    })
                }),
            ),
            m_indices: m_neighbor_indices.into(),
        })
    }

    /// Get the grid of 8 neighboring gridpoints for linear interpolation, for a certain Teff, M, logg.
    pub fn get_local_grid_linear(&self, teff: f64, m: f64, logg: f64) -> Result<LocalGridLinear> {
        if !self.is_teff_logg_between_bounds(teff, logg) {
            bail!("Teff, logg out of bounds ({}, {})", teff, logg);
        }
        if !self.is_m_between_bounds(m) {
            bail!("M out of bounds ({})", m);
        }
        let teff_limits = self.get_teff_index_limits_at(logg)?;
        let m_limits = (0, self.m.n() - 1);
        let logg_limits = self.get_logg_index_limits_at(teff)?;

        let teff_neighbor_indices = self
            .teff
            .find_neighbors_linear(teff, teff_limits)
            .with_context(|| {
                format!(
                    "failed getting neighbors for teff: {}, {:?}",
                    teff, teff_limits
                )
            })?;
        let m_neighbor_indices = self.m.find_neighbors_linear(m, m_limits).with_context(|| {
            format!(
                "failed getting neighbors for m: {}, {:?}",
                logg, logg_limits
            )
        })?;
        let logg_neighbor_indices = self
            .logg
            .find_neighbors_linear(logg, logg_limits)
            .with_context(|| {
                format!(
                    "failed getting neighbors for logg: {}, {:?}",
                    logg, logg_limits
                )
            })?;

        Ok(LocalGridLinear {
            teff: (
                teff,
                teff_neighbor_indices.map(|x| {
                    self.teff
                        .get(x)
                        .with_context(|| {
                            format!(
                                "teff={}, neighbors: {:?}, limits: {:?}",
                                teff, teff_neighbor_indices, teff_limits
                            )
                        })
                        .unwrap()
                }),
            ),
            teff_indices: teff_neighbor_indices.into(),
            logg: (
                logg,
                logg_neighbor_indices.map(|x| {
                    self.logg
                        .get(x)
                        .with_context(|| {
                            format!(
                                "logg={}, neighbors: {:?}, limits: {:?}",
                                logg, logg_neighbor_indices, logg_limits
                            )
                        })
                        .unwrap()
                }),
            ),
            logg_indices: logg_neighbor_indices.into(),
            m: (
                m,
                m_neighbor_indices.map(|x| {
                    self.m
                        .get(x)
                        .with_context(|| {
                            format!(
                                "m={}, neighbors: {:?}, limits: {:?}",
                                m, m_neighbor_indices, m_limits
                            )
                        })
                        .unwrap()
                }),
            ),
            m_indices: m_neighbor_indices.into(),
        })
    }

    /// Clamp a parameter tuple (Teff, M, logg) to the grid bounds along one dimension.
    /// index: 0 for Teff, 1 for M, 2 for logg
    /// It is required that a point inside the gridpoints can be found,
    /// only by changing the parameter value in the given dimension.
    pub fn clamp_1d(&self, param: na::SVector<f64, 3>, dimension: usize) -> Result<f64> {
        let limits = self.get_limits_at(param, dimension)?;
        Ok(param[dimension].clamp(limits.0, limits.1))
    }
}

impl GridBounds for Grid {
    fn limits(&self) -> (na::SVector<f64, 3>, na::SVector<f64, 3>) {
        (
            na::Vector3::new(
                *self.teff.values.first().unwrap(),
                *self.m.values.first().unwrap(),
                *self.logg.values.first().unwrap(),
            ),
            na::Vector3::new(
                *self.teff.values.last().unwrap(),
                *self.m.values.last().unwrap(),
                *self.logg.values.last().unwrap(),
            ),
        )
    }

    fn get_limits_at(&self, param: na::SVector<f64, 3>, index: usize) -> Result<(f64, f64)> {
        Ok(match index {
            0 => {
                let (left_index, right_index) = self
                    .get_teff_index_limits_at(param[2])
                    .context(anyhow!("Cannot clamp teff at logg={}", param[2]))?;
                (self.teff.get(left_index)?, self.teff.get(right_index)?)
            }
            1 => self.m.get_first_and_last(),
            2 => {
                let (left_index, right_index) = self
                    .get_logg_index_limits_at(param[0])
                    .context(anyhow!("Cannot clamp logg at Teff={}", param[0]))?;
                (self.logg.get(left_index)?, self.logg.get(right_index)?)
            }
            _ => panic!("Index out of bounds {}", index),
        })
    }

    fn is_within_bounds(&self, param: na::SVector<f64, 3>) -> bool {
        // When the parameters are not in the rectangular grid:
        self.is_m_between_bounds(param[1]) & self.is_teff_logg_between_bounds(param[0], param[2])
    }
}

#[derive(Clone, Debug)]
pub enum WlGrid {
    // first, step, total
    Linspace(f64, f64, usize),
    // log10(first), step log10, total
    Logspace(f64, f64, usize),
    NonUniform(na::DVector<f64>),
}

impl WlGrid {
    /// Number of pixels.
    pub fn n(&self) -> usize {
        match self {
            WlGrid::Linspace(_, _, n) => *n,
            WlGrid::Logspace(_, _, n) => *n,
            WlGrid::NonUniform(v) => v.len(),
        }
    }

    /// Get first and last wavelength value.
    pub fn get_first_and_last(&self) -> (f64, f64) {
        match self {
            WlGrid::Linspace(first, step, total) => (*first, *first + step * *total as f64),
            WlGrid::Logspace(first, step, total) => (
                10_f64.powf(*first),
                10_f64.powf(*first + step * *total as f64),
            ),
            WlGrid::NonUniform(v) => (v[0], v[v.len() - 1]),
        }
    }

    /// Get the index of a wavelength in the grid as a float.
    /// i.e. the wavelength falls between indices floor(result) and ceil(result).
    pub fn get_float_index_of_wl(&self, wl: f64) -> f64 {
        match self {
            WlGrid::Linspace(first, step, _) => (wl - first) / step,
            WlGrid::Logspace(first, step, _) => (wl.log10() - first) / step,
            WlGrid::NonUniform(v) => {
                let i = v
                    .data
                    .as_vec()
                    .binary_search_by(|x| x.partial_cmp(&wl).unwrap());
                match i {
                    Ok(i) => i as f64,
                    Err(i) => {
                        if i == 0 {
                            0.0
                        } else if i == v.len() {
                            v.len() as f64 - 1.0
                        } else {
                            let left = v[i - 1];
                            let right = v[i];
                            (wl - left) / (right - left) + i as f64 - 1.0
                        }
                    }
                }
            }
        }
    }

    /// Get an iterator over pixels.
    pub fn iterate(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        match self {
            WlGrid::Linspace(first, step, total) => {
                Box::new((0..*total).map(move |i| (first + step * i as f64)))
            }
            WlGrid::Logspace(first, step, total) => {
                Box::new((0..*total).map(move |i| 10_f64.powf(first + step * i as f64)))
            }
            WlGrid::NonUniform(v) => Box::new(v.iter().copied()),
        }
    }
}

pub struct BinaryComponents {
    pub norm_model1: ArraySegment,
    pub continuum1: ArraySegment,
    pub norm_model2: ArraySegment,
    pub continuum2: ArraySegment,
    pub lr: f32,
}

pub trait Interpolator: Send + Sync {
    type GB: GridBounds;

    fn synth_wl(&self) -> WlGrid;
    fn grid_bounds(&self) -> Self::GB;
    fn interpolate(&self, teff: f64, m: f64, logg: f64) -> Result<na::DVector<FluxFloat>>;
    fn interpolate_linear(&self, teff: f64, m: f64, logg: f64) -> Result<na::DVector<FluxFloat>>;
    fn interpolate_and_convolve(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        teff: f64,
        m: f64,
        logg: f64,
        vsini: f64,
    ) -> Result<ArraySegment>;
    fn interpolate_linear_and_convolve(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        teff: f64,
        m: f64,
        logg: f64,
        vsini: f64,
    ) -> Result<ArraySegment>;
    fn produce_model(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        teff: f64,
        m: f64,
        logg: f64,
        vsini: f64,
        rv: f64,
    ) -> Result<na::DVector<FluxFloat>>;
    fn produce_model_linear(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        teff: f64,
        m: f64,
        logg: f64,
        vsini: f64,
        rv: f64,
    ) -> Result<na::DVector<FluxFloat>>;
    fn produce_model_on_grid(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        teff: f64,
        m: f64,
        logg: f64,
        vsini: f64,
        rv: f64,
    ) -> Result<na::DVector<FluxFloat>>;

    fn produce_binary_model(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        star1_parameters: &na::SVector<f64, 5>,
        star2_parameters: &na::SVector<f64, 5>,
        light_ratio: f32,
    ) -> Result<na::DVector<FluxFloat>> {
        let model1 = self.produce_model(
            target_dispersion,
            star1_parameters[0],
            star1_parameters[1],
            star1_parameters[2],
            star1_parameters[3],
            star1_parameters[4],
        )?;
        let model2 = self.produce_model(
            target_dispersion,
            star2_parameters[0],
            star2_parameters[1],
            star2_parameters[2],
            star2_parameters[3],
            star2_parameters[4],
        )?;
        Ok(model1 * light_ratio + model2 * (1.0 - light_ratio))
    }

    fn produce_binary_components(
        &self,
        continuum_interpolator: &impl Interpolator,
        target_dispersion: &impl WavelengthDispersion,
        star1_parameters: &na::SVector<f64, 4>,
        star2_parameters: &na::SVector<f64, 4>,
        light_ratio: f32,
    ) -> Result<BinaryComponents> {
        let norm_model1 = self.interpolate_and_convolve(
            target_dispersion,
            star1_parameters[0],
            star1_parameters[1],
            star1_parameters[2],
            star1_parameters[3],
        )?;
        // For the continuum we can take some shortcuts:
        // use linear interpolation and skip rotation and resolution broadening
        let continuum1 = continuum_interpolator.interpolate_linear(
            star1_parameters[0],
            star1_parameters[1],
            star1_parameters[2],
        )?;

        let norm_model2 = self.interpolate_and_convolve(
            target_dispersion,
            star2_parameters[0],
            star2_parameters[1],
            star2_parameters[2],
            star2_parameters[3],
        )?;
        let continuum2 = continuum_interpolator.interpolate_linear(
            star2_parameters[0],
            star2_parameters[1],
            star2_parameters[2],
        )?;

        let lr = light_ratio * continuum1.mean() / continuum2.mean();
        Ok(BinaryComponents {
            norm_model1,
            norm_model2,
            continuum1: continuum1.into(),
            continuum2: continuum2.into(),
            lr,
        })
    }

    fn produce_binary_model_norm(
        &self,
        continuum_interpolator: &impl Interpolator,
        target_dispersion: &impl WavelengthDispersion,
        star1_parameters: &na::SVector<f64, 5>,
        star2_parameters: &na::SVector<f64, 5>,
        light_ratio: f32,
    ) -> Result<na::DVector<FluxFloat>> {
        let components = self.produce_binary_components(
            continuum_interpolator,
            target_dispersion,
            &star1_parameters.fixed_rows::<4>(0).into_owned(), // Skip RV shift in this stage
            &star2_parameters.fixed_rows::<4>(0).into_owned(),
            light_ratio,
        )?;

        shift_resample_and_add_binary_components(
            &self.synth_wl(),
            &components,
            target_dispersion,
            [star1_parameters[4], star2_parameters[4]],
        )
    }
}

pub trait ModelFetcher: Send + Sync {
    fn grid(&self) -> &Grid;
    fn find_spectrum(&self, i: usize, j: usize, k: usize) -> Result<(CowVector, f32)>;
}

#[derive(Clone)]
pub struct GridInterpolator<F: ModelFetcher> {
    pub fetcher: F,
    pub synth_wl: WlGrid,
}

impl<F: ModelFetcher> GridInterpolator<F> {
    pub fn new(model_fetcher: F, synth_wl: WlGrid) -> Self {
        Self {
            fetcher: model_fetcher,
            synth_wl,
        }
    }

    pub fn grid(&self) -> &Grid {
        self.fetcher.grid()
    }
}

const BATCH_SIZE: usize = 128;

impl<F: ModelFetcher> Interpolator for GridInterpolator<F> {
    type GB = Grid;

    fn grid_bounds(&self) -> Self::GB {
        self.fetcher.grid().clone()
    }

    fn synth_wl(&self) -> WlGrid {
        self.synth_wl.clone()
    }

    fn interpolate(&self, teff: f64, m: f64, logg: f64) -> Result<na::DVector<FluxFloat>> {
        let local_grid = self.grid().get_local_grid(teff, m, logg)?;

        let factors = calculate_interpolation_coefficients(&local_grid)?;
        let neighbors = local_grid
            .teff_logg_indices
            .iter()
            .flat_map(|teff_logg_index| {
                local_grid
                    .m_indices
                    .iter()
                    .map(move |m_index| match (teff_logg_index, m_index) {
                        (Some((i, j)), Some(k)) => self.fetcher.find_spectrum(*i, *k, *j),
                        _ => Ok((
                            CowVector::Owned(na::DVector::zeros(self.synth_wl.n())),
                            1.0 / 65535.0,
                        )),
                    })
            })
            .collect::<Result<Vec<(CowVector, f32)>>>()?;

        let model_length = neighbors[0].0.len();
        let mut interpolated: na::DVector<FluxFloat> = na::DVector::zeros(model_length);
        let mut mat = na::SMatrix::<FluxFloat, BATCH_SIZE, 64>::zeros();
        for i in 0..(model_length / BATCH_SIZE) {
            let start = i * BATCH_SIZE;
            for j in 0..64 {
                let (column, factor) = &neighbors[j];
                mat.set_column(
                    j,
                    &column
                        .fixed_rows::<BATCH_SIZE>(start)
                        .map(|x| (x as FluxFloat) * factor),
                );
            }

            mat.mul_to(&factors, &mut interpolated.rows_mut(start, BATCH_SIZE));
        }
        // Add remaining part
        let start = (model_length / BATCH_SIZE) * BATCH_SIZE;
        let remaining = model_length - start;
        let mut mat = na::Matrix::<FluxFloat, na::Dyn, na::Const<64>, _>::zeros(remaining);
        for j in 0..64 {
            let (column, factor) = &neighbors[j];
            mat.set_column(
                j,
                &column
                    .rows(start, remaining)
                    .map(|x| (x as FluxFloat) / 65535.0 * factor),
            );
        }
        mat.mul_to(&factors, &mut interpolated.rows_mut(start, remaining));
        Ok(interpolated)
    }

    fn interpolate_linear(&self, teff: f64, m: f64, logg: f64) -> Result<na::DVector<FluxFloat>> {
        let local_grid = self.grid().get_local_grid_linear(teff, m, logg)?;

        let factors = calculate_interpolation_coefficients_linear(&local_grid)?;
        let neighbors = local_grid
            .teff_indices
            .iter()
            .cartesian_product(local_grid.logg_indices.iter())
            .cartesian_product(local_grid.m_indices.iter())
            .map(|((i, j), k)| self.fetcher.find_spectrum(*i, *k, *j))
            .collect::<Result<Vec<(CowVector, f32)>>>()?;

        let model_length = neighbors[0].0.len();
        let mut interpolated: na::DVector<FluxFloat> = na::DVector::zeros(model_length);
        let mut mat = na::SMatrix::<FluxFloat, BATCH_SIZE, 8>::zeros();
        for i in 0..(model_length / BATCH_SIZE) {
            let start = i * BATCH_SIZE;
            for j in 0..8 {
                let (column, factor) = &neighbors[j];
                mat.set_column(
                    j,
                    &column
                        .fixed_rows::<BATCH_SIZE>(start)
                        .map(|x| (x as FluxFloat) / 65535.0 * factor),
                );
            }
            mat.mul_to(&factors, &mut interpolated.rows_mut(start, BATCH_SIZE));
        }
        // Add remaining part
        let start = (model_length / BATCH_SIZE) * BATCH_SIZE;
        let remaining = model_length - start;
        let mut mat = na::Matrix::<FluxFloat, na::Dyn, na::Const<8>, _>::zeros(remaining);
        for j in 0..8 {
            let (column, factor) = &neighbors[j];
            mat.set_column(
                j,
                &column
                    .rows(start, remaining)
                    .map(|x| (x as FluxFloat) / 65535.0 * factor),
            );
        }
        mat.mul_to(&factors, &mut interpolated.rows_mut(start, remaining));
        Ok(interpolated)
    }

    fn interpolate_and_convolve(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        teff: f64,
        m: f64,
        logg: f64,
        vsini: f64,
    ) -> Result<ArraySegment> {
        let interpolated = self
            .interpolate(teff, m, logg)
            .with_context(|| format!("Error while interpolating at ({}, {}, {})", teff, m, logg))?;

        let convolved_for_rotation = convolve_rotation(&self.synth_wl, &interpolated, vsini)
            .with_context(|| format!("Error during rotational broadening, vsini={}", vsini))?;
        target_dispersion
            .convolve_segment(convolved_for_rotation)
            .context("Error during instrument resolution convolution")
    }

    fn interpolate_linear_and_convolve(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        teff: f64,
        m: f64,
        logg: f64,
        vsini: f64,
    ) -> Result<ArraySegment> {
        let interpolated = self
            .interpolate_linear(teff, m, logg)
            .with_context(|| format!("Error while interpolating at ({}, {}, {})", teff, m, logg))?;

        let convolved_for_rotation = convolve_rotation(&self.synth_wl, &interpolated, vsini)
            .with_context(|| format!("Error during rotational broadening, vsini={}", vsini))?;
        target_dispersion
            .convolve_segment(convolved_for_rotation)
            .context("Error during instrument resolution convolution")
    }

    fn produce_model(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        teff: f64,
        m: f64,
        logg: f64,
        vsini: f64,
        rv: f64,
    ) -> Result<na::DVector<FluxFloat>> {
        let interpolated = self
            .interpolate(teff, m, logg)
            .with_context(|| format!("Error while interpolating at ({}, {}, {})", teff, m, logg))?;

        let convolved_for_rotation = convolve_rotation(&self.synth_wl, &interpolated, vsini)
            .with_context(|| format!("Error during rotational broadening, vsini={}", vsini))?;
        let model = target_dispersion
            .convolve_segment(convolved_for_rotation)
            .context("Error during instrument resolution convolution")?;
        let output = shift_and_resample(&self.synth_wl, &model, target_dispersion, rv)
            .context("error during RV shifting / resampling")?;

        Ok(output)
    }

    fn produce_model_linear(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        teff: f64,
        m: f64,
        logg: f64,
        vsini: f64,
        rv: f64,
    ) -> Result<na::DVector<FluxFloat>> {
        let interpolated = self
            .interpolate_linear(teff, m, logg)
            .with_context(|| format!("Error while interpolating linearly at ({}, {}, {})", teff, m, logg))?;

        let convolved_for_rotation = convolve_rotation(&self.synth_wl, &interpolated, vsini)
            .with_context(|| format!("Error during rotational broadening, vsini={}", vsini))?;
        let model = target_dispersion
            .convolve_segment(convolved_for_rotation)
            .context("Error during instrument resolution convolution")?;
        let output = shift_and_resample(&self.synth_wl, &model, target_dispersion, rv)
            .context("error during RV shifting / resampling")?;

        Ok(output)
    }

    fn produce_model_on_grid(
        &self,
        target_dispersion: &impl WavelengthDispersion,
        teff: f64,
        m: f64,
        logg: f64,
        vsini: f64,
        rv: f64,
    ) -> Result<na::DVector<FluxFloat>> {
        let teff_limits = self.grid().get_teff_index_limits_at(logg)?;
        let m_limits = (0, self.grid().m.n() - 1);
        let logg_limits = self.grid().get_logg_index_limits_at(teff)?;

        let i = self
            .fetcher
            .grid()
            .teff
            .try_get_index(teff, teff_limits)
            .context("Teff not in grid")?;
        let j = self
            .fetcher
            .grid()
            .m
            .try_get_index(m, m_limits)
            .context("m not in grid")?;
        let k = self
            .fetcher
            .grid()
            .logg
            .try_get_index(logg, logg_limits)
            .context("logg not in grid")?;
        let (spec, factor) = self.fetcher.find_spectrum(i, j, k)?;

        let convolved_for_rotation = convolve_rotation(
            &self.synth_wl,
            &spec
                .into_owned()
                .map(|x| (x as FluxFloat) / 65535.0 * factor),
            vsini,
        )
        .with_context(|| format!("Error during rotational broadening, vsini={}", vsini))?;
        let model = target_dispersion
            .convolve_segment(convolved_for_rotation)
            .context("Error during instrument resolution convolution")?;
        let output = shift_and_resample(&self.synth_wl, &model, target_dispersion, rv)
            .context("error during RV shifting / resampling")?;

        Ok(output)
    }
}
