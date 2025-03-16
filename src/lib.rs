#![allow(non_snake_case, clippy::too_many_arguments, non_upper_case_globals)]
pub mod bounds;
pub mod continuum_fitting;
pub mod convolve_rv;
pub mod cubic;
pub mod fitting;
pub mod interpolate;
pub mod model_fetchers;
pub mod particleswarm;

use anyhow::{Context, Result};
use bounds::{BoundsConstraint, Constraint};
use continuum_fitting::{
    ChunkFitter, ConstantContinuum as RsConstantContinuum, ContinuumFitter,
    FixedContinuum as RsFixedContinuum, LinearModelFitter,
};
use convolve_rv::{
    shift_and_resample, ArraySegment, FixedTargetDispersionLogSpace, FixedTargetDispersionNonLog,
    NoConvolutionDispersionTarget, VariableTargetDispersion, WavelengthDispersion,
};
use cubic::{calculate_interpolation_coefficients, calculate_interpolation_coefficients_flat};
use enum_dispatch::enum_dispatch;
use fitting::{
    BinaryFitter, BinaryRVFitter, BinaryTimeriesKnownRVFitter, ObservedSpectrum, Observer,
    ParticleInfo, SingleFitter,
};
use indicatif::ProgressBar;
use interpolate::{FluxFloat, GridInterpolator, Interpolator};
use model_fetchers::{read_npy_file, CachedFetcher, InMemFetcher, OnDiskFetcher};
use nalgebra as na;
use nalgebra::Storage;
use npy::to_file;
use numpy::array::PyArray;
use numpy::{Ix1, Ix2, PyArrayLike};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyDict, PyFloat, PyInt, PyList};
use pyo3::{prelude::*, pyclass};
use rayon::prelude::*;
use regex::Regex;
use serde::Serialize;
use std::fs::{read_dir, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

/// Parameters to the PSO algorithm
#[derive(Clone, Debug)]
#[pyclass(module = "pasta", frozen)]
pub struct PSOSettings(fitting::PSOSettings);

#[pymethods]
impl PSOSettings {
    /// Create a new PSOSettings object.
    /// Inertia factor default: 0.7213475204.
    /// Cognitive factor default: 1.1931471806.
    /// Social factor default: 1.1931471806.
    /// Delta default: 1e-7 (forced move threshold).
    #[new]
    pub fn new(
        num_particles: usize,
        max_iters: u64,
        inertia_factor: Option<f64>,
        cognitive_factor: Option<f64>,
        social_factor: Option<f64>,
        delta: Option<f64>,
    ) -> Self {
        PSOSettings(fitting::PSOSettings {
            num_particles,
            max_iters,
            inertia_factor: inertia_factor.unwrap_or(0.7213475204),
            cognitive_factor: cognitive_factor.unwrap_or(1.1931471806),
            social_factor: social_factor.unwrap_or(1.1931471806),
            delta: delta.unwrap_or(1e-7),
        })
    }
}

/// Python to rust bindings.
impl From<PSOSettings> for fitting::PSOSettings {
    fn from(settings: PSOSettings) -> Self {
        settings.0
    }
}

#[derive(Clone, Debug, Serialize)]
#[pyclass(module = "pasta", frozen)]
pub struct Label {
    #[pyo3(get)]
    teff: f64,
    #[pyo3(get)]
    m: f64,
    #[pyo3(get)]
    logg: f64,
    #[pyo3(get)]
    vsini: f64,
    #[pyo3(get)]
    rv: f64,
}

impl From<fitting::Label<f64>> for Label {
    fn from(value: fitting::Label<f64>) -> Self {
        Self {
            teff: value.teff,
            m: value.m,
            logg: value.logg,
            vsini: value.vsini,
            rv: value.rv,
        }
    }
}

impl From<Label> for fitting::Label<f64> {
    fn from(val: Label) -> Self {
        fitting::Label {
            teff: val.teff,
            m: val.m,
            logg: val.logg,
            vsini: val.vsini,
            rv: val.rv,
        }
    }
}

#[pymethods]
impl Label {
    fn as_list(&self) -> [f64; 5] {
        [self.teff, self.m, self.logg, self.vsini, self.rv]
    }

    fn to_dict<'a>(&self, py: Python<'a>) -> Bound<'a, PyDict> {
        let dict = PyDict::new_bound(py);
        dict.set_item("teff", self.teff).unwrap();
        dict.set_item("m", self.m).unwrap();
        dict.set_item("logg", self.logg).unwrap();
        dict.set_item("vsini", self.vsini).unwrap();
        dict.set_item("rv", self.rv).unwrap();
        dict
    }

    #[staticmethod]
    fn from_dict(py: Python<'_>, dict: &PyDict) -> PyResult<Self> {
        Ok(Self {
            teff: dict
                .get_item("teff")?
                .unwrap()
                .downcast::<PyFloat>()?
                .extract()?,
            m: dict
                .get_item("m")?
                .unwrap()
                .downcast::<PyFloat>()?
                .extract()?,
            logg: dict
                .get_item("logg")?
                .unwrap()
                .downcast::<PyFloat>()?
                .extract()?,
            vsini: dict
                .get_item("vsini")?
                .unwrap()
                .downcast::<PyFloat>()?
                .extract()?,
            rv: dict
                .get_item("rv")?
                .unwrap()
                .downcast::<PyFloat>()?
                .extract()?,
        })
    }
}

#[derive(Clone, Debug, Serialize)]
#[pyclass(module = "pasta", frozen)]
pub struct LabelUncertainties {
    #[pyo3(get)]
    teff: (Option<f64>, Option<f64>),
    #[pyo3(get)]
    m: (Option<f64>, Option<f64>),
    #[pyo3(get)]
    logg: (Option<f64>, Option<f64>),
    #[pyo3(get)]
    vsini: (Option<f64>, Option<f64>),
    #[pyo3(get)]
    rv: (Option<f64>, Option<f64>),
}

impl From<fitting::Label<(Result<f64>, Result<f64>)>> for LabelUncertainties {
    fn from(value: fitting::Label<(Result<f64>, Result<f64>)>) -> Self {
        Self {
            teff: (value.teff.0.ok(), value.teff.1.ok()),
            m: (value.m.0.ok(), value.m.1.ok()),
            logg: (value.logg.0.ok(), value.logg.1.ok()),
            vsini: (value.vsini.0.ok(), value.vsini.1.ok()),
            rv: (value.rv.0.ok(), value.rv.1.ok()),
        }
    }
}

#[pymethods]
impl LabelUncertainties {
    fn to_json(&self) -> PyResult<String> {
        Ok(serde_json::to_string(self).unwrap())
    }

    fn to_dict<'a>(&self, py: Python<'a>) -> Bound<'a, PyDict> {
        let dict = PyDict::new_bound(py);
        dict.set_item("teff", self.teff).unwrap();
        dict.set_item("m", self.m).unwrap();
        dict.set_item("logg", self.logg).unwrap();
        dict.set_item("vsini", self.vsini).unwrap();
        dict.set_item("rv", self.rv).unwrap();
        dict
    }
}

/// Output of the PSO fitting algorithm.
#[derive(Clone, Debug, Serialize)]
#[pyclass(module = "pasta", frozen)]
pub struct OptimizationResult {
    /// Teff, [M/H], logg, vsini, RV
    #[pyo3(get)]
    pub label: Label,
    /// Fitted parameters for the continuum fitting function
    /// (polynomial coefficients)
    #[pyo3(get)]
    pub continuum_params: Vec<FluxFloat>,
    #[pyo3(get)]
    /// Chi2 value
    pub chi2: f64,
    #[pyo3(get)]
    /// Number of iterations used
    pub iterations: u64,
    /// Time taken
    #[pyo3(get)]
    pub time: f64,
}

/// Rust to Python bindings.
impl From<fitting::OptimizationResult> for OptimizationResult {
    fn from(result: fitting::OptimizationResult) -> Self {
        OptimizationResult {
            label: result.label.into(),
            continuum_params: result.continuum_params.data.into(),
            chi2: result.chi2,
            iterations: result.iters,
            time: result.time,
        }
    }
}

#[pymethods]
impl OptimizationResult {
    fn to_json(&self) -> PyResult<String> {
        Ok(serde_json::to_string(self).unwrap())
    }

    fn to_dict<'a>(&self, py: Python<'a>) -> Bound<'a, PyDict> {
        let dict = PyDict::new_bound(py);
        dict.set_item("label", self.label.to_dict(py)).unwrap();
        dict.set_item("continuum_params", self.continuum_params.clone())
            .unwrap();
        dict.set_item("chi2", self.chi2).unwrap();
        dict.set_item("iterations", self.iterations).unwrap();
        dict.set_item("time", self.time).unwrap();
        dict
    }

    #[staticmethod]
    fn from_dict(py: Python<'_>, dict: &PyDict) -> PyResult<Self> {
        Ok(Self {
            label: Label::from_dict(
                py,
                dict.get_item("label").unwrap().unwrap().downcast().unwrap(),
            )
            .unwrap(),
            continuum_params: dict
                .get_item("continuum_params")?
                .unwrap()
                .downcast::<PyList>()?
                .extract()?,
            chi2: dict
                .get_item("chi2")?
                .unwrap()
                .downcast::<PyFloat>()?
                .extract()?,
            iterations: dict
                .get_item("iterations")?
                .unwrap()
                .downcast::<PyInt>()?
                .extract()?,
            time: dict
                .get_item("time")?
                .unwrap()
                .downcast::<PyFloat>()?
                .extract()?,
        })
    }
}

/// Output of the PSO binary fitting algorithm.
#[derive(Clone, Debug, Serialize)]
#[pyclass(module = "pasta", frozen)]
pub struct BinaryOptimizationResult {
    /// (Teff, [M/H], logg, vsini, RV)
    #[pyo3(get)]
    pub label1: Label,
    #[pyo3(get)]
    pub label2: Label,
    #[pyo3(get)]
    pub light_ratio: f64,
    /// Fitted parameters for the continuum fitting function
    /// (polynomial coefficients)
    #[pyo3(get)]
    pub continuum_params: Vec<FluxFloat>,
    /// Chi2 value
    #[pyo3(get)]
    pub chi2: f64,
    /// Number of iterations used
    #[pyo3(get)]
    pub iterations: u64,
    /// Time taken
    #[pyo3(get)]
    pub time: f64,
}
/// Rust to Python bindings.
impl From<fitting::BinaryOptimizationResult> for BinaryOptimizationResult {
    fn from(result: fitting::BinaryOptimizationResult) -> Self {
        BinaryOptimizationResult {
            label1: result.label1.into(),
            label2: result.label2.into(),
            light_ratio: result.light_ratio,
            continuum_params: result.continuum_params.data.into(),
            chi2: result.chi2,
            iterations: result.iters,
            time: result.time,
        }
    }
}

#[pymethods]
impl BinaryOptimizationResult {
    fn to_json(&self) -> PyResult<String> {
        Ok(serde_json::to_string(self).unwrap())
    }

    fn to_dict<'a>(&self, py: Python<'a>) -> Bound<'a, PyDict> {
        let dict = PyDict::new_bound(py);
        dict.set_item("label1", self.label1.to_dict(py)).unwrap();
        dict.set_item("label2", self.label2.to_dict(py)).unwrap();
        dict.set_item("light_ratio", self.light_ratio).unwrap();
        dict.set_item("continuum_params", self.continuum_params.clone())
            .unwrap();
        dict.set_item("chi2", self.chi2).unwrap();
        dict.set_item("iterations", self.iterations).unwrap();
        dict.set_item("time", self.time).unwrap();
        dict
    }

    #[staticmethod]
    fn from_dict(py: Python<'_>, dict: &PyDict) -> PyResult<Self> {
        Ok(Self {
            label1: Label::from_dict(
                py,
                dict.get_item("label1")
                    .unwrap()
                    .unwrap()
                    .downcast()
                    .unwrap(),
            )
            .unwrap(),
            label2: Label::from_dict(
                py,
                dict.get_item("label2")
                    .unwrap()
                    .unwrap()
                    .downcast()
                    .unwrap(),
            )
            .unwrap(),
            light_ratio: dict
                .get_item("light_ratio")?
                .unwrap()
                .downcast::<PyFloat>()?
                .extract()?,
            continuum_params: dict
                .get_item("continuum_params")?
                .unwrap()
                .downcast::<PyList>()?
                .extract()?,
            chi2: dict
                .get_item("chi2")?
                .unwrap()
                .downcast::<PyFloat>()?
                .extract()?,
            iterations: dict
                .get_item("iterations")?
                .unwrap()
                .downcast::<PyInt>()?
                .extract()?,
            time: dict
                .get_item("time")?
                .unwrap()
                .downcast::<PyFloat>()?
                .extract()?,
        })
    }

    fn label_list(&self) -> [f64; 11] {
        [
            self.label1.teff,
            self.label1.m,
            self.label1.logg,
            self.label1.vsini,
            self.label1.rv,
            self.label2.teff,
            self.label2.m,
            self.label2.logg,
            self.label2.vsini,
            self.label2.rv,
            self.light_ratio,
        ]
    }
}

#[derive(Clone, Debug, Serialize)]
#[pyclass(module = "pasta", frozen)]
pub struct RVOptimizationResult {
    #[pyo3(get)]
    pub rv1: f64,
    #[pyo3(get)]
    pub rv2: f64,
    #[pyo3(get)]
    pub chi2: f64,
    /// Number of iterations used
    #[pyo3(get)]
    pub iterations: u64,
    /// Time taken
    #[pyo3(get)]
    pub time: f64,
}

/// Rust to Python bindings.
impl From<fitting::BinaryRVOptimizationResult> for RVOptimizationResult {
    fn from(result: fitting::BinaryRVOptimizationResult) -> Self {
        RVOptimizationResult {
            rv1: result.rv1,
            rv2: result.rv2,
            chi2: result.chi2,
            iterations: result.iters,
            time: result.time,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[pyclass(module = "pasta", frozen)]
pub struct RvLessLabel {
    #[pyo3(get)]
    teff: f64,
    #[pyo3(get)]
    m: f64,
    #[pyo3(get)]
    logg: f64,
    #[pyo3(get)]
    vsini: f64,
}

impl From<fitting::RvLessLabel<f64>> for RvLessLabel {
    fn from(value: fitting::RvLessLabel<f64>) -> Self {
        Self {
            teff: value.teff,
            m: value.m,
            logg: value.logg,
            vsini: value.vsini,
        }
    }
}

impl From<RvLessLabel> for fitting::RvLessLabel<f64> {
    fn from(val: RvLessLabel) -> Self {
        fitting::RvLessLabel {
            teff: val.teff,
            m: val.m,
            logg: val.logg,
            vsini: val.vsini,
        }
    }
}

#[pymethods]
impl RvLessLabel {
    fn to_dict<'a>(&self, py: Python<'a>) -> Bound<'a, PyDict> {
        let dict = PyDict::new_bound(py);
        dict.set_item("teff", self.teff).unwrap();
        dict.set_item("m", self.m).unwrap();
        dict.set_item("logg", self.logg).unwrap();
        dict.set_item("vsini", self.vsini).unwrap();
        dict
    }

    #[staticmethod]
    fn from_dict(py: Python<'_>, dict: &PyDict) -> PyResult<Self> {
        Ok(Self {
            teff: dict
                .get_item("teff")?
                .unwrap()
                .downcast::<PyFloat>()?
                .extract()?,
            m: dict
                .get_item("m")?
                .unwrap()
                .downcast::<PyFloat>()?
                .extract()?,
            logg: dict
                .get_item("logg")?
                .unwrap()
                .downcast::<PyFloat>()?
                .extract()?,
            vsini: dict
                .get_item("vsini")?
                .unwrap()
                .downcast::<PyFloat>()?
                .extract()?,
        })
    }
}

#[derive(Clone, Debug, Serialize)]
#[pyclass(module = "pasta", frozen)]
pub struct BinaryTimeseriesOptimizationResult {
    #[pyo3(get)]
    pub label1: RvLessLabel,
    #[pyo3(get)]
    pub label2: RvLessLabel,
    #[pyo3(get)]
    pub light_ratio: f64,
    #[pyo3(get)]
    pub continuum_params: Vec<Vec<FluxFloat>>,
    #[pyo3(get)]
    pub chis: Vec<f64>,
    #[pyo3(get)]
    pub iters: u64,
    #[pyo3(get)]
    pub time: f64,
}

impl From<fitting::BinaryTimeseriesOptimizationResult> for BinaryTimeseriesOptimizationResult {
    fn from(result: fitting::BinaryTimeseriesOptimizationResult) -> Self {
        BinaryTimeseriesOptimizationResult {
            label1: result.label1.into(),
            label2: result.label2.into(),
            light_ratio: result.light_ratio,
            continuum_params: result
                .continuum_params
                .into_iter()
                .map(|x| x.data.into())
                .collect(),
            chis: result.chis,
            iters: result.iters,
            time: result.time,
        }
    }
}

#[pymethods]
impl BinaryTimeseriesOptimizationResult {
    fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    fn to_dict<'a>(&self, py: Python<'a>) -> Bound<'a, PyDict> {
        let dict = PyDict::new_bound(py);
        dict.set_item("label1", self.label1.to_dict(py)).unwrap();
        dict.set_item("label2", self.label2.to_dict(py)).unwrap();
        dict.set_item("light_ratio", self.light_ratio).unwrap();
        dict.set_item("continuum_params", self.continuum_params.clone())
            .unwrap();
        dict.set_item("chis", self.chis.clone()).unwrap();
        dict.set_item("iters", self.iters).unwrap();
        dict.set_item("time", self.time).unwrap();
        dict
    }

    #[staticmethod]
    fn from_dict(py: Python<'_>, dict: &PyDict) -> PyResult<Self> {
        Ok(Self {
            label1: RvLessLabel::from_dict(
                py,
                dict.get_item("label1")
                    .unwrap()
                    .unwrap()
                    .downcast()
                    .unwrap(),
            )
            .unwrap(),
            label2: RvLessLabel::from_dict(
                py,
                dict.get_item("label2")
                    .unwrap()
                    .unwrap()
                    .downcast()
                    .unwrap(),
            )
            .unwrap(),
            light_ratio: dict
                .get_item("light_ratio")?
                .unwrap()
                .downcast::<PyFloat>()?
                .extract()?,
            continuum_params: dict
                .get_item("continuum_params")?
                .unwrap()
                .downcast::<PyList>()?
                .extract()?,
            chis: dict
                .get_item("chis")?
                .unwrap()
                .downcast::<PyList>()?
                .extract()?,
            iters: dict
                .get_item("iters")?
                .unwrap()
                .downcast::<PyInt>()?
                .extract()?,
            time: dict
                .get_item("time")?
                .unwrap()
                .downcast::<PyFloat>()?
                .extract()?,
        })
    }
}

#[derive(Clone, Debug)]
#[pyclass(module = "pasta", frozen)]
pub struct WlGrid(interpolate::WlGrid);

#[pymethods]
impl WlGrid {
    #[new]
    pub fn new(min: f64, step: f64, len: usize, log: Option<bool>) -> Self {
        if log.unwrap_or(false) {
            WlGrid(interpolate::WlGrid::Logspace(min, step, len))
        } else {
            WlGrid(interpolate::WlGrid::Linspace(min, step, len))
        }
    }

    #[staticmethod]
    pub fn non_uniform(wl: Vec<f64>) -> Self {
        WlGrid(interpolate::WlGrid::NonUniform(wl.into()))
    }

    pub fn get_array(&self) -> Vec<f64> {
        self.0.iterate().collect()
    }
}

#[enum_dispatch(WavelengthDispersion)]
#[derive(Clone, Debug)]
enum WavelengthDispersionWrapper {
    NoConvolutionDispersionTarget,
    FixedTargetDispersionLogSpace,
    FixedTargetDispersionNonLog,
    VariableTargetDispersion,
}

#[derive(Clone, Debug)]
#[pyclass(module = "pasta", frozen)]
pub struct PyWavelengthDispersion(WavelengthDispersionWrapper);

#[pymethods]
impl PyWavelengthDispersion {
    /// Convolve a spectrum with the dispersion kernel.
    pub fn convolve<'a>(
        &self,
        py: Python<'a>,
        flux: Vec<FluxFloat>,
    ) -> PyResult<Bound<'a, PyArray<FluxFloat, Ix1>>> {
        Ok(PyArray::from_vec_bound(
            py,
            self.0.convolve(na::DVector::from_vec(flux))?.data.into(),
        ))
    }

    /// Get the wavelength grid.
    pub fn wavelength(&self) -> Vec<f64> {
        self.0.wavelength().clone().data.into()
    }

    /// Convolve every model spectrum in a directory with the dispersion kernel
    /// Filenames will be kept the same.
    /// The wavelength grid of the models in `input_directory`` is specified by `input_wavelength`.
    /// `includes_factor` specifies whether a factor to multiply the spectrum with is included in the file,
    /// i.e. whether this represents a normalized or unnormalized model.
    pub fn convolve_and_resample_directory(
        &self,
        input_directory: String,
        output_directory: String,
        includes_factor: bool,
        input_wavelength: WlGrid,
    ) {
        let out_dir = Path::new(output_directory.as_str());
        let files: Vec<PathBuf> = read_dir(input_directory)
            .unwrap()
            .map(|x| x.unwrap().path())
            .collect();
        let bar = ProgressBar::new(files.len() as u64);
        files.into_par_iter().for_each(|file| {
            let out_file = out_dir.join(file.file_name().unwrap());
            if out_file.exists() {
                return;
            }
            let arr: na::DVector<u16> = read_npy_file(File::open(file).unwrap()).unwrap().into();
            if includes_factor {
                let bytes1 = arr[0].to_le_bytes();
                let bytes2 = arr[1].to_le_bytes();
                let factor = f32::from_le_bytes([bytes1[0], bytes1[1], bytes2[0], bytes2[1]]);

                let spectrum_float = arr
                    .rows(2, arr.len() - 2)
                    .map(|x| (x as FluxFloat) / 65535.0 * factor);
                let convolved = self.0.convolve(spectrum_float).unwrap();
                let resampled =
                    shift_and_resample(&input_wavelength.0, &convolved.into(), &self.0, 0.0)
                        .unwrap();

                let max = resampled.max();
                let max_bytes = max.to_le_bytes();
                let first_u16 = u16::from_le_bytes([max_bytes[0], max_bytes[1]]);
                let second_u16 = u16::from_le_bytes([max_bytes[2], max_bytes[3]]);

                let resampled_u16 = [first_u16, second_u16]
                    .into_iter()
                    .chain(
                        resampled
                            .into_iter()
                            .map(|x| (x.max(0.0) / max * 65535.0) as u16),
                    )
                    .collect::<Vec<_>>();
                to_file(out_file, resampled_u16).unwrap();
            } else {
                let spectrum_float = arr.map(|x| (x as FluxFloat) / 65535.0);
                let convolved = self.0.convolve(spectrum_float).unwrap();
                let resampled =
                    shift_and_resample(&input_wavelength.0, &convolved.into(), &self.0, 0.0)
                        .unwrap();
                let resampled_u16 = resampled
                    .into_iter()
                    .map(|x| (x.clamp(0.0, 1.0) * 65535.0) as u16)
                    .collect::<Vec<_>>();
                to_file(out_file, resampled_u16).unwrap();
            };

            bar.inc(1);
        });
    }
}

/// Create a new wavelength dispersion object for an instrument with variable resolution.
/// `wl` is the wavelength grid of the instrument.
/// `disp` is the resolution at each wavelength ($\lambda / \Delta \lambda$).
/// `synth_wl` is the wavelength grid of the synthetic spectra that will be convolved for this resolution.
#[pyfunction]
fn VariableResolutionDispersion(
    wl: Vec<f64>,
    disp: Vec<FluxFloat>,
    synth_wl: WlGrid,
) -> PyResult<PyWavelengthDispersion> {
    Ok(PyWavelengthDispersion(
        VariableTargetDispersion::new(wl.into(), &disp.into(), synth_wl.0)?.into(),
    ))
}

/// Create a new wavelength dispersion object for an instrument with fixed resolution.
/// `wl` is the wavelength grid of the instrument.
/// `resolution` is the resolution of the instrument.
/// `synth_wl` is the wavelength grid of the synthetic spectra that will be convolved for this resolution.
#[pyfunction]
fn FixedResolutionDispersion(
    wl: Vec<f64>,
    resolution: f64,
    synth_wl: WlGrid,
) -> PyResult<PyWavelengthDispersion> {
    Ok(PyWavelengthDispersion(match synth_wl.0 {
        interpolate::WlGrid::Logspace(_, _, _) => {
            FixedTargetDispersionLogSpace::new(wl.into(), resolution, &synth_wl.0)?.into()
        }
        _ => FixedTargetDispersionNonLog::new(wl.into(), resolution, synth_wl.0)?.into(),
    }))
}

/// Create a new wavelength dispersion object for models that have already been convolved to the instrument resolution.
/// This skips the instrument resolution convolution step in the model production.
/// `wl` is the wavelength grid of the instrument.
#[pyfunction]
fn NoConvolutionDispersion(wl: Vec<f64>, wl_grid: WlGrid) -> PyWavelengthDispersion {
    PyWavelengthDispersion(NoConvolutionDispersionTarget::new(wl.into(), &wl_grid.0).into())
}

/// All available methods to fit the pseudo continuum
#[enum_dispatch(ContinuumFitter)]
#[derive(Clone, Debug)]
enum ContinuumFitterWrapper {
    ChunkFitter,
    LinearModelFitter,
    RsFixedContinuum,
    RsConstantContinuum,
}

/// Represents a constraint during the fitting process.
#[pyclass(module = "pasta", frozen)]
#[derive(Clone, Debug)]
pub struct PyConstraintWrapper(BoundsConstraint);

/// Constraint that fixes one parameter to a given value during the fitting process.
/// `parameter` is the index of the parameter to fix.
/// `value` is the value to fix the parameter to.
#[pyfunction]
fn FixConstraint(parameter: usize, value: f64) -> PyConstraintWrapper {
    PyConstraintWrapper(BoundsConstraint {
        parameter,
        constraint: Constraint::Fixed(value),
    })
}

/// Constraint that bounds one parameter to a given range during the fitting process.
/// `parameter` is the index of the parameter to bound.
/// `lower` is the lower bound.
/// `upper` is the upper bound.
#[pyfunction]
fn RangeConstraint(parameter: usize, lower: f64, upper: f64) -> PyConstraintWrapper {
    PyConstraintWrapper(BoundsConstraint {
        parameter,
        constraint: Constraint::Range(lower, upper),
    })
}

fn get_observer_and_init<const N: usize>(
    trace_directory: Option<String>,
    best_particle_file: Option<String>,
    resume: bool,
) -> (Observer, Option<Vec<na::SVector<f64, N>>>) {
    match (trace_directory, best_particle_file, resume) {
        (Some(trace_directory), None, false) => (
            Observer::PSOobserver {
                dir: trace_directory.into(),
                file_prefix: "it".into(),
                iteration_offset: 0,
            },
            None,
        ),
        (Some(trace_directory), None, true) => {
            let paths = read_dir(&trace_directory).unwrap();
            let path_strings = paths
                .into_iter()
                .map(|path| path.unwrap().file_name().to_str().unwrap().to_string())
                .collect::<Vec<_>>();
            let re = Regex::new(r"_(\d+)\.json").unwrap();
            let iterations = path_strings
                .iter()
                .map(|path| {
                    re.captures(path)
                        .unwrap()
                        .get(1)
                        .unwrap()
                        .as_str()
                        .parse::<usize>()
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let (i, _) = iterations
                .iter()
                .enumerate()
                .max_by_key(|(_, &value)| value)
                .unwrap();
            let file = File::open(Path::new(&trace_directory).join(&path_strings[i])).unwrap();
            let reader = BufReader::new(file);
            let particles: Vec<ParticleInfo> = serde_json::from_reader(reader).unwrap();
            let positions = particles
                .into_iter()
                .map(|p| na::SVector::<f64, N>::from_vec(p.position))
                .collect::<Vec<_>>();
            (
                Observer::PSOobserver {
                    dir: trace_directory.into(),
                    file_prefix: "it".into(),
                    iteration_offset: iterations[i] + 1,
                },
                Some(positions),
            )
        }
        (None, Some(best_particle_file), false) => (
            Observer::BestParticleObserver {
                filename: best_particle_file.into(),
                iteration_offset: 0,
            },
            None,
        ),
        (None, Some(_), true) => panic!("Cannot resumed from best particle info"),
        (Some(_), Some(_), _) => {
            panic!("Cannot specify both trace_directory and best_particle_file")
        }
        _ => (Observer::DummyObserver, None),
    }
}

#[derive(Clone, Debug)]
#[pyclass(module = "pasta", frozen)]
pub struct PyContinuumFitter(ContinuumFitterWrapper);

#[pymethods]
impl PyContinuumFitter {
    /// Fit the pseudo continuum given a model spec and an observed spec.
    /// i.e. fit the function to `observed` / `model`.
    /// Returns the fitted parameters of the continuum function and the chi2 value.
    fn fit_continuum<'a>(
        &self,
        py: Python<'a>,
        model: Vec<FluxFloat>,
        observed: Vec<FluxFloat>,
        observed_var: Vec<FluxFloat>,
    ) -> PyResult<(Bound<'a, PyArray<FluxFloat, Ix1>>, f64)> {
        let observed_spectrum = ObservedSpectrum::from_vecs(observed, observed_var);
        let (params, ls) = self
            .0
            .fit_continuum(&observed_spectrum, &na::DVector::from_vec(model))?;
        Ok((PyArray::from_vec_bound(py, params.data.into()), ls))
    }

    /// Fit the pseudo continuum given a model spec and an observed spec,
    /// and return the fitted continuum function itself instead of its parameters.
    fn fit_and_return_continuum<'a>(
        &self,
        py: Python<'a>,
        synth: Vec<FluxFloat>,
        observed: Vec<FluxFloat>,
        observed_var: Vec<FluxFloat>,
    ) -> PyResult<Bound<'a, PyArray<FluxFloat, Ix1>>> {
        let observed_spectrum = ObservedSpectrum::from_vecs(observed, observed_var);
        let result = self.0.fit_continuum_and_return_continuum(
            &observed_spectrum,
            &na::DVector::from_vec(synth),
        )?;
        Ok(PyArray::from_vec_bound(py, result.data.into()))
    }

    /// Fit the pseudo continuum given a model spec and an observed spec,
    /// and return the normalized observed spectrum.
    fn fit_continuum_and_return_normalized_spec<'a>(
        &self,
        py: Python<'a>,
        synth: Vec<FluxFloat>,
        observed: Vec<FluxFloat>,
        observed_var: Vec<FluxFloat>,
    ) -> PyResult<Bound<'a, PyArray<FluxFloat, Ix1>>> {
        let observed_spectrum = ObservedSpectrum::from_vecs(observed, observed_var);
        let continuum = self.0.fit_continuum_and_return_continuum(
            &observed_spectrum,
            &na::DVector::from_vec(synth),
        )?;
        let result = observed_spectrum.flux.component_div(&continuum);
        Ok(PyArray::from_vec_bound(py, result.data.into()))
    }

    /// Fit the pseudo continuum given a model spec and an observed spec,
    /// and return the chi2 value.
    fn fit_and_return_chi2(
        &self,
        synth: Vec<FluxFloat>,
        y: Vec<FluxFloat>,
        yerr: Vec<FluxFloat>,
    ) -> PyResult<f64> {
        let observed_spectrum = ObservedSpectrum::from_vecs(y, yerr);
        let (_, chi2) = self
            .0
            .fit_continuum(&observed_spectrum, &na::DVector::from_vec(synth))?;
        Ok(chi2)
    }

    /// Construct the continuum function from its parameters.
    fn build_continuum<'a>(
        &self,
        py: Python<'a>,
        params: Vec<FluxFloat>,
    ) -> PyResult<Bound<'a, PyArray<FluxFloat, Ix1>>> {
        Ok(PyArray::from_vec_bound(
            py,
            self.0
                .build_continuum(&na::DVector::from_vec(params))?
                .data
                .into(),
        ))
    }
}

/// Continuum model that divides the spectrum into `n_chunks` chunks, with `overlap` overlap (0-1),
/// and fits a `p_order` order polynomial to each chunk.
#[pyfunction]
fn ChunkContinuumFitter(
    wl: Vec<f64>,
    n_chunks: Option<usize>,
    p_order: Option<usize>,
    overlap: Option<f64>,
) -> PyContinuumFitter {
    PyContinuumFitter(
        ChunkFitter::new(
            wl.into(),
            n_chunks.unwrap_or(5),
            p_order.unwrap_or(8),
            overlap.unwrap_or(0.2),
        )
        .into(),
    )
}

/// Continuum model that fits a linear model to the pseudo continuum.
#[pyfunction]
fn LinearModelContinuumFitter(design_matrix: PyArrayLike<FluxFloat, Ix2>) -> PyContinuumFitter {
    PyContinuumFitter(LinearModelFitter::new(design_matrix.as_matrix().into()).into())
}

/// Continuum model that fixes the continuum to a fixed array. Hence no fitting is done
#[pyfunction]
fn FixedContinuum(
    continuum: Vec<FluxFloat>,
    ignore_first_and_last: Option<usize>,
) -> PyContinuumFitter {
    PyContinuumFitter(
        RsFixedContinuum::new(continuum.into(), ignore_first_and_last.unwrap_or(0)).into(),
    )
}

/// Continuum model that fixes the continuum to a constant value. Hence no fitting is done
#[pyfunction]
fn ConstantContinuum() -> PyContinuumFitter {
    PyContinuumFitter(continuum_fitting::ConstantContinuum().into())
}

/// Implement methods for all the interpolator classes.
/// We can't use traits with pyo3, so we have to use macros.
macro_rules! implement_methods {
    ($name: ident, $fetcher_type: ty, $PySingleFitter: ident, $PyBinaryFitter: ident, $PyRVFitter: ident, $PyTSRVFitter: ident) => {
        #[pymethods]
        impl $name {
            /// Produce a model spectrum by interpolating, convolving,
            /// shifting by rv and resampling to the requested wavelength dispersion.
            pub fn produce_model<'a>(
                &self,
                py: Python<'a>,
                dispersion: PyWavelengthDispersion,
                teff: f64,
                m: f64,
                logg: f64,
                vsini: f64,
                rv: f64,
            ) -> PyResult<Bound<'a, PyArray<FluxFloat, Ix1>>> {
                let interpolated = self
                    .0
                    .produce_model(&dispersion.0, teff, m, logg, vsini, rv)?;
                // let interpolated = [0.0; LEN_C];
                Ok(PyArray::from_vec_bound(
                    py,
                    interpolated.iter().copied().collect(),
                ))
            }

            /// Produce a normalized model spectrum for a binary system.
            /// A reference to a `continuum_interpolator` that points to a grid of unnormalized models is required.
            /// That is because the continuum information is needed to correctly add the two spectra.
            /// The labels of the stars are given in `labels1` and `labels2`.
            /// The `light_ratio` is the ratio of the light of the first star to the light of the second star,
            /// taking into account the different continua.
            pub fn produce_binary_model_norm<'a>(
                &self,
                py: Python<'a>,
                continuum_interpolator: &$name,
                dispersion: PyWavelengthDispersion,
                labels1: [f64; 5],
                labels2: [f64; 5],
                light_ratio: f64,
            ) -> PyResult<Bound<'a, PyArray<FluxFloat, Ix1>>> {
                let interpolated = self.0.produce_binary_model_norm(
                    &continuum_interpolator.0,
                    &dispersion.0,
                    &na::Vector5::new(labels1[0], labels1[1], labels1[2], labels1[3], labels1[4]),
                    &na::Vector5::new(labels2[0], labels2[1], labels2[2], labels2[3], labels2[4]),
                    light_ratio as f32,
                )?;
                // let interpolated = [0.0; LEN_C];
                Ok(PyArray::from_vec_bound(
                    py,
                    interpolated.iter().copied().collect(),
                ))
            }

            /// Build the wavelength dispersion kernels (debugging purposes).
            pub fn get_kernels<'a>(
                &self,
                py: Python<'a>,
                wl: Vec<f64>,
                disp: Vec<FluxFloat>,
            ) -> PyResult<Bound<'a, PyArray<FluxFloat, Ix2>>> {
                let target_dispersion = VariableTargetDispersion::new(
                    wl.into(),
                    &disp.into(),
                    self.0.synth_wl().clone(),
                )?;
                let matrix = target_dispersion.kernels;
                let v: Vec<Vec<FluxFloat>> = matrix
                    .row_iter()
                    .map(|x| x.data.into_owned().into())
                    .collect();
                Ok(PyArray::from_vec2_bound(py, &v[..])?)
            }

            /// Produce multiple model spectra with multithreading.
            /// labels: Vec of (Teff, [M/H], logg, vsini, RV) tuples.
            pub fn produce_model_bulk<'a>(
                &self,
                py: Python<'a>,
                dispersion: PyWavelengthDispersion,
                labels: Vec<[f64; 5]>,
                progress: Option<bool>,
            ) -> PyResult<Bound<'a, PyArray<FluxFloat, Ix2>>> {
                let progress_bar = if progress.unwrap_or(false) {
                    ProgressBar::new(labels.len() as u64)
                } else {
                    ProgressBar::hidden()
                };
                let vec: Vec<Vec<FluxFloat>> = labels
                    .into_par_iter()
                    .map(|[teff, m, logg, vsini, rv]| {
                        progress_bar.inc(1);
                        Ok(self
                            .0
                            .produce_model(&dispersion.0, teff, m, logg, vsini, rv)?
                            .iter()
                            .copied()
                            .collect())
                    })
                    .collect::<Result<_>>()?;
                Ok(PyArray::from_vec2_bound(py, &vec[..])?)
            }

            /// Produce a model directly from a grid model (without interpolating).
            /// Throws an error if (teff, m, logg) is not in the grid.
            pub fn produce_model_on_grid<'a>(
                &self,
                py: Python<'a>,
                dispersion: PyWavelengthDispersion,
                teff: f64,
                mh: f64,
                logg: f64,
                vsini: f64,
                rv: f64,
            ) -> PyResult<Bound<'a, PyArray<FluxFloat, Ix1>>> {
                let out = self
                    .0
                    .produce_model_on_grid(&dispersion.0, teff, mh, logg, vsini, rv)?;
                Ok(PyArray::from_vec_bound(py, out.data.into()))
            }

            /// Interpolate in grid.
            /// Doesn't do convoluton, shifting and resampling.
            pub fn interpolate<'a>(
                &self,
                py: Python<'a>,
                teff: f64,
                m: f64,
                logg: f64,
            ) -> PyResult<Bound<'a, PyArray<FluxFloat, Ix1>>> {
                let interpolated = self.0.interpolate(teff, m, logg)?;
                Ok(PyArray::from_vec_bound(
                    py,
                    interpolated.iter().copied().collect(),
                ))
            }

            /// Interpolate in grid with linear instead of cubic interpolation.
            /// Doesn't do convoluton, shifting and resampling.
            pub fn interpolate_linear<'a>(
                &self,
                py: Python<'a>,
                teff: f64,
                m: f64,
                logg: f64,
            ) -> PyResult<Bound<'a, PyArray<FluxFloat, Ix1>>> {
                let interpolated = self.0.interpolate_linear(teff, m, logg)?;
                Ok(PyArray::from_vec_bound(
                    py,
                    interpolated.iter().copied().collect(),
                ))
            }

            pub fn interpolate_and_convolve<'a>(
                &self,
                py: Python<'a>,
                dispersion: PyWavelengthDispersion,
                teff: f64,
                m: f64,
                logg: f64,
                vsini: f64,
            ) -> PyResult<Bound<'a, PyArray<FluxFloat, Ix1>>> {
                let interpolated =
                    self.0
                        .interpolate_and_convolve(&dispersion.0, teff, m, logg, vsini)?;
                Ok(PyArray::from_vec_bound(
                    py,
                    interpolated.as_view().iter().copied().collect(),
                ))
            }

            /// Get a `SingleFitter` object, used to fit spectra of single stars with the PSO algorithm.
            pub fn get_fitter(
                &self,
                dispersion: PyWavelengthDispersion,
                continuum_fitter: PyContinuumFitter,
                vsini_range: (f64, f64),
                rv_range: (f64, f64),
            ) -> $PySingleFitter {
                $PySingleFitter(SingleFitter::new(
                    dispersion.0,
                    continuum_fitter.0,
                    vsini_range,
                    rv_range,
                ))
            }

            /// Get a `BinaryFitter` object, used to fit spectra of binary stars with the PSO algorithm.
            pub fn get_binary_fitter(
                &self,
                dispersion: PyWavelengthDispersion,
                continuum_fitter: PyContinuumFitter,
                vsini_range: (f64, f64),
                rv_range: (f64, f64),
            ) -> $PyBinaryFitter {
                $PyBinaryFitter(BinaryFitter::new(
                    dispersion.0,
                    continuum_fitter.0,
                    vsini_range,
                    rv_range,
                ))
            }

            /// Get a `BinaryRVFitter` object, used to fit the RV's of binary stars,
            /// where the other labels are already known.
            pub fn get_binary_rv_fitter(
                &self,
                dispersion: PyWavelengthDispersion,
                synth_wl: WlGrid,
                continuum_fitter: PyContinuumFitter,
                rv_range: (f64, f64),
            ) -> $PyRVFitter {
                $PyRVFitter(BinaryRVFitter::new(
                    dispersion.0.clone(),
                    synth_wl.0,
                    continuum_fitter.0,
                    rv_range,
                ))
            }

            pub fn get_timeseries_known_rv_fitter(
                &self,
                dispersion: PyWavelengthDispersion,
                synth_wl: WlGrid,
                continuum_fitter: PyContinuumFitter,
                vsini_range: (f64, f64),
            ) -> $PyTSRVFitter {
                $PyTSRVFitter(BinaryTimeriesKnownRVFitter::new(
                    dispersion.0,
                    synth_wl.0,
                    continuum_fitter.0,
                    vsini_range,
                ))
            }

            /// Fit a continuum to an observed spectrum and model,
            /// as given by the labels (Teff, [M/H], logg, vsini, RV).
            /// Return the continuum.
            pub fn fit_continuum<'a>(
                &self,
                py: Python<'a>,
                fitter: PyContinuumFitter,
                dispersion: PyWavelengthDispersion,
                observed_flux: Vec<FluxFloat>,
                observed_var: Vec<FluxFloat>,
                label: [f64; 5],
            ) -> PyResult<Bound<'a, PyArray<FluxFloat, Ix1>>> {
                let observed_spectrum = ObservedSpectrum::from_vecs(observed_flux, observed_var);
                let synth = self.0.produce_model(
                    &dispersion.0,
                    label[0],
                    label[1],
                    label[2],
                    label[2],
                    label[4],
                )?;
                Ok(PyArray::from_vec_bound(
                    py,
                    fitter
                        .0
                        .fit_continuum_and_return_continuum(&observed_spectrum, &synth)?
                        .iter()
                        .copied()
                        .collect(),
                ))
            }

            /// Fit the continuum for an observed spectrum and model,
            /// as given by the labels (Teff, [M/H], logg, vsini, RV).
            /// Return the parameters of the fit function.
            pub fn fit_continuum_and_return_model<'a>(
                &self,
                py: Python<'a>,
                fitter: PyContinuumFitter,
                dispersion: PyWavelengthDispersion,
                observed_flux: Vec<FluxFloat>,
                observed_var: Vec<FluxFloat>,
                label: [f64; 5],
            ) -> PyResult<Bound<'a, PyArray<FluxFloat, Ix1>>> {
                let observed_spectrum = ObservedSpectrum::from_vecs(observed_flux, observed_var);
                let synth = self.0.produce_model(
                    &dispersion.0,
                    label[0],
                    label[1],
                    label[2],
                    label[3],
                    label[4],
                )?;
                Ok(PyArray::from_vec_bound(
                    py,
                    fitter
                        .0
                        .fit_continuum_and_return_fit(&observed_spectrum, &synth)?
                        .iter()
                        .copied()
                        .collect(),
                ))
            }

            /// Compute the chi2 value at a given set of labels for multiple spectra.
            /// Continuum is fitted.
            pub fn chi2_bulk(
                &self,
                observed_wavelength: Vec<Vec<f64>>,
                observed_flux: Vec<Vec<FluxFloat>>,
                observed_var: Vec<Vec<FluxFloat>>,
                labels: Vec<[f64; 5]>,
                progress: Option<bool>,
                allow_nan: Option<bool>,
            ) -> PyResult<Vec<f64>> {
                let progress_bar = if progress.unwrap_or(false) {
                    ProgressBar::new(labels.len() as u64)
                } else {
                    ProgressBar::hidden()
                };
                Ok(labels
                    .into_par_iter()
                    .zip(observed_wavelength)
                    .zip(observed_flux)
                    .zip(observed_var)
                    .map(|((([teff, m, logg, vsini, rv], wl), flux), var)| {
                        progress_bar.inc(1);
                        let obs = ObservedSpectrum::from_vecs(flux, var);
                        let target_dispersion =
                            NoConvolutionDispersionTarget::new(wl.into(), &self.0.synth_wl());
                        let synth_model_result =
                            self.0
                                .produce_model(&target_dispersion, teff, m, logg, vsini, rv);
                        let synth_model = match allow_nan {
                            Some(true) => match synth_model_result {
                                Ok(x) => x,
                                Err(_) => return f64::NAN,
                            },
                            _ => synth_model_result.unwrap(),
                        };
                        // let synth_model = synth_model_result.unwrap();
                        let fitter =
                            ChunkFitter::new(target_dispersion.wavelength().clone(), 5, 8, 0.2);
                        let (_, chi2) = fitter.fit_continuum(&obs, &synth_model).unwrap();
                        chi2
                    })
                    .collect())
            }

            pub fn ccf<'a>(
                &self,
                py: Python<'a>,
                vs: Vec<f64>,
                teff: f64,
                m: f64,
                logg: f64,
                dispersion: &PyWavelengthDispersion,
                observed_flux: Vec<FluxFloat>,
            ) -> PyResult<Bound<'a, PyArray<FluxFloat, Ix1>>> {
                if dispersion.0.wavelength().len() != observed_flux.len() {
                    return Err(PyValueError::new_err(
                        "Dispersion and observed flux must have the same length.",
                    ));
                }
                let flux = na::DVector::from_vec(observed_flux);
                let model_spectrum = self.0.interpolate(teff, m, logg).unwrap().into();
                Ok(PyArray::from_vec_bound(
                    py,
                    vs.into_par_iter()
                        .map(|v| {
                            let shifted = shift_and_resample(
                                &self.0.synth_wl(),
                                &model_spectrum,
                                &dispersion.0,
                                v,
                            )?;
                            Ok(shifted.dot(&flux))
                        })
                        .collect::<Result<_>>()?,
                ))
            }

            /// Clamp a parameter tuple (Teff, M, logg) to the grid bounds along one dimension.
            /// index: 0 for Teff, 1 for M, 2 for logg
            /// It is required that a point inside the gridpoints can be found,
            /// only by changing the parameter value in the given dimension.
            pub fn clamp_1d(&self, teff: f64, m: f64, logg: f64, index: usize) -> PyResult<f64> {
                Ok(self
                    .0
                    .grid_bounds()
                    .clamp_1d(na::Vector3::new(teff, m, logg), index)?)
            }

            /// Calculate the interpolation coefficients for a given set of labels:
            /// 4x4 for teff and logg, 4 for m.
            /// Returned in order (teff, logg, m)
            /// (Debugging purposes)
            pub fn calculate_interpolation_coefficients_flat(
                &self,
                teff: f64,
                m: f64,
                logg: f64,
            ) -> PyResult<([[f64; 4]; 4], [[f64; 4]; 4], [f64; 4])> {
                let local_grid = self.0.grid().get_local_grid(teff, m, logg)?;
                let (x, y, z) = calculate_interpolation_coefficients_flat(&local_grid)?;
                Ok((x.into(), y.into(), z.into()))
            }

            /// Calculate the grid of interpolation coefficients for a given set of labels (debugging purposes).
            pub fn calculate_interpolation_coefficients(
                &self,
                teff: f64,
                m: f64,
                logg: f64,
            ) -> PyResult<Vec<FluxFloat>> {
                let local_grid = self.0.grid().get_local_grid(teff, m, logg)?;
                Ok(calculate_interpolation_coefficients(&local_grid)?
                    .into_iter()
                    .cloned()
                    .collect())
            }

            /// Get the indices of the neighbors of a given set of labels. (0, 0, 0) if outside the grid.
            pub fn get_neighbor_indices(
                &self,
                teff: f64,
                m: f64,
                logg: f64,
            ) -> PyResult<Vec<(usize, usize, usize)>> {
                let local_grid = self.0.grid().get_local_grid(teff, m, logg)?;
                Ok(local_grid
                    .teff_logg_indices
                    .iter()
                    .flat_map(|teff_logg_index| {
                        local_grid.m_indices.iter().map(move |m_index| {
                            match (teff_logg_index, m_index) {
                                (Some((i, j)), Some(k)) => (*i, *k, *j),
                                _ => (0, 0, 0),
                            }
                        })
                    })
                    .collect())
            }

            /// Get all unique teff values in the grid.
            pub fn teffs(&self) -> Vec<f64> {
                self.0.grid().teff.values.clone()
            }

            /// Get all unique M values in the grid.
            pub fn ms(&self) -> Vec<f64> {
                self.0.grid().m.values.clone()
            }

            /// Get all unique logg values in the grid.
            pub fn loggs(&self) -> Vec<f64> {
                self.0.grid().logg.values.clone()
            }

            /// Get the outer logg limits of the grid.
            pub fn logg_limits(&self) -> Vec<(usize, usize)> {
                self.0.grid().logg_limits.clone()
            }

            pub fn cumulative_grid_size(&self) -> Vec<usize> {
                self.0.grid().cumulative_grid_size.clone()
            }

            pub fn list_gridpoints(&self) -> Vec<[f64; 3]> {
                self.0.grid().list_gridpoints()
            }

            pub fn is_teff_logg_between_bounds(&self, teff: f64, logg: f64) -> bool {
                self.0.grid().is_teff_logg_between_bounds(teff, logg)
            }
        }

        #[pyclass]
        pub struct $PySingleFitter(
            SingleFitter<WavelengthDispersionWrapper, ContinuumFitterWrapper>,
        );

        #[pymethods]
        impl $PySingleFitter {
            /// Fit an observed spectrum of a single star with the PSO algorithm.
            /// trace_directory is optional. If specified, the traces of the optimization process will be saved there.
            /// If `parallelize` is true, the optimization will be parallelized over particles.
            pub fn fit(
                &self,
                interpolator: &$name,
                observed_flux: PyArrayLike<FluxFloat, Ix1>,
                observed_var: PyArrayLike<FluxFloat, Ix1>,
                settings: PSOSettings,
                iterations: Option<[u64; 3]>,
                first_stage_res: Option<f64>,
                trace_directory: Option<String>,
                best_particle_file: Option<String>,
                parallelize: Option<bool>,
                constraints: Option<Vec<PyConstraintWrapper>>,
            ) -> PyResult<OptimizationResult> {
                let observed_spectrum = ObservedSpectrum {
                    flux: observed_flux.as_matrix().column(0).into_owned(),
                    var: observed_var.as_matrix().column(0).into_owned(),
                };
                let constraints = match constraints {
                    Some(constraints) => constraints
                        .into_iter()
                        .map(|constraint| constraint.0)
                        .collect(),
                    None => vec![],
                };
                Ok(self
                    .0
                    .fit_three_stage(
                        &interpolator.0,
                        &observed_spectrum.into(),
                        [
                            settings.clone().into(),
                            settings.clone().into(),
                            settings.into(),
                        ],
                        iterations.unwrap_or([30, 60, 10]),
                        first_stage_res.unwrap_or(10_000.0),
                        get_observer_and_init::<5>(trace_directory, best_particle_file, false).0,
                        parallelize.unwrap_or(true),
                        constraints,
                    )?
                    .into())
            }

            /// Fit many observed spectra, multithreading over spectra instead of particles.
            /// A 2D matrix is expected for observed_fluxes and vars, where each row is a spectrum.
            pub fn fit_bulk(
                &self,
                interpolator: &$name,
                observed_fluxes: Vec<Vec<FluxFloat>>,
                observed_vars: Vec<Vec<FluxFloat>>,
                settings: PSOSettings,
                iterations: Option<[u64; 3]>,
                first_stage_res: Option<f64>,
                constraints: Option<Vec<Vec<PyConstraintWrapper>>>,
                best_particle_files: Option<Vec<String>>,
                progress: Option<bool>,
            ) -> PyResult<Vec<Option<OptimizationResult>>> {
                let progress_bar = if progress.unwrap_or(false) {
                    ProgressBar::new(observed_fluxes.len() as u64)
                } else {
                    ProgressBar::hidden()
                };
                let constraints = match constraints {
                    Some(all_constraints) => all_constraints
                        .into_iter()
                        .map(|constraints| {
                            constraints
                                .into_iter()
                                .map(|constraint| constraint.0)
                                .collect()
                        })
                        .collect(),
                    None => vec![vec![]; observed_fluxes.len()],
                };
                let observers = match best_particle_files {
                    Some(files) => files
                        .into_iter()
                        .map(|file| Observer::BestParticleObserver {
                            filename: file.into(),
                            iteration_offset: 0,
                        })
                        .collect(),
                    None => vec![Observer::DummyObserver; observed_fluxes.len()],
                };
                Ok(observed_fluxes
                    .into_par_iter()
                    .zip(observed_vars)
                    .zip(constraints)
                    .zip(observers)
                    .enumerate()
                    .map(|(i, (((flux, var), constraint), observer))| {
                        let observed_spectrum = ObservedSpectrum::from_vecs(flux, var);
                        let result = self.0.fit_three_stage(
                            &interpolator.0,
                            &observed_spectrum.into(),
                            [
                                settings.clone().into(),
                                settings.clone().into(),
                                settings.clone().into(),
                            ],
                            iterations.unwrap_or([30, 60, 10]),
                            first_stage_res.unwrap_or(10_000.0),
                            observer,
                            false,
                            constraint,
                        );

                        progress_bar.inc(1);

                        Some(
                            result
                                .with_context(|| format!("Error for spectrum {}", i))
                                .unwrap()
                                .into(),
                        )
                    })
                    .collect::<Vec<_>>())
            }

            /// Calculate uncertainties with the chi2 landscape method
            /// parameters: best fit
            /// search_radius: radius in which the intersection point is searched for every parameter
            pub fn compute_uncertainty(
                &self,
                interpolator: &$name,
                observed_flux: Vec<FluxFloat>,
                observed_var: Vec<FluxFloat>,
                spec_res: f64,
                label: [f64; 5],
                search_radius: Option<[f64; 5]>,
            ) -> PyResult<LabelUncertainties> {
                let search_radius = search_radius.unwrap_or([2000.0, 0.3, 0.3, 40.0, 40.0]);
                let observed_spectrum = ObservedSpectrum::from_vecs(observed_flux, observed_var);
                Ok(self
                    .0
                    .compute_uncertainty(
                        &interpolator.0,
                        &observed_spectrum,
                        spec_res,
                        fitting::Label::from_array(label).into(),
                        fitting::Label::from_array(search_radius).into(),
                    )?
                    .into())
            }

            /// Uncertainties with the chi2 landscape method,
            /// for many spectra with multithreading
            pub fn compute_uncertainty_bulk(
                &self,
                interpolator: &$name,
                observed_fluxes: Vec<Vec<FluxFloat>>,
                observed_vars: Vec<Vec<FluxFloat>>,
                spec_res: f64,
                labels: Vec<[f64; 5]>,
                search_radius: Option<[f64; 5]>,
            ) -> Vec<LabelUncertainties> {
                let search_radius = search_radius.unwrap_or([2000.0, 0.3, 0.3, 40.0, 40.0]);
                let bar = ProgressBar::new(observed_fluxes.len() as u64);
                observed_fluxes
                    .into_par_iter()
                    .zip(observed_vars)
                    .zip(labels)
                    .map(|((flux, var), params)| {
                        bar.inc(1);
                        let obs = ObservedSpectrum::from_vecs(flux, var);
                        self.0
                            .compute_uncertainty(
                                &interpolator.0,
                                &obs,
                                spec_res,
                                fitting::Label::from_array(params).into(),
                                fitting::Label::from_array(search_radius).into(),
                            )
                            .unwrap()
                            .into()
                    })
                    .collect()
            }

            /// Compute the chi2 value at a given set of labels, continuum is fitted.
            pub fn chi2(
                &self,
                interpolator: &$name,
                observed_flux: Vec<FluxFloat>,
                observed_var: Vec<FluxFloat>,
                labels: Vec<[f64; 5]>,
                allow_nan: Option<bool>,
            ) -> PyResult<Vec<f64>> {
                let observed_spectrum = ObservedSpectrum::from_vecs(observed_flux, observed_var);
                Ok(labels
                    .into_par_iter()
                    .map(|labels| {
                        let chi = self
                            .0
                            .chi2(&interpolator.0, &observed_spectrum, labels.into());
                        match allow_nan {
                            Some(true) => match chi {
                                Ok(x) => x,
                                Err(_) => f64::NAN,
                            },
                            _ => chi.unwrap(),
                        }
                    })
                    .collect())
            }
        }

        #[pyclass]
        pub struct $PyBinaryFitter(
            BinaryFitter<WavelengthDispersionWrapper, ContinuumFitterWrapper>,
        );

        #[pymethods]
        impl $PyBinaryFitter {
            pub fn fit(
                &self,
                interpolator: &$name,
                continuum_interpolator: &$name,
                observed_flux: PyArrayLike<FluxFloat, Ix1>,
                observed_var: PyArrayLike<FluxFloat, Ix1>,
                settings: PSOSettings,
                trace_directory: Option<String>,
                best_particle_file: Option<String>,
                parallelize: Option<bool>,
                constraints: Option<Vec<PyConstraintWrapper>>,
                resume: Option<bool>,
            ) -> PyResult<BinaryOptimizationResult> {
                let observed_spectrum = ObservedSpectrum {
                    flux: observed_flux.as_matrix().column(0).into_owned(),
                    var: observed_var.as_matrix().column(0).into_owned(),
                };
                let constraints = match constraints {
                    Some(constraints) => constraints
                        .into_iter()
                        .map(|constraint| constraint.0)
                        .collect(),
                    None => vec![],
                };
                let (observer, initial_positions) = get_observer_and_init(
                    trace_directory,
                    best_particle_file,
                    resume.unwrap_or(false),
                );
                let result = self.0.fit(
                    &interpolator.0,
                    &continuum_interpolator.0,
                    &observed_spectrum.into(),
                    settings.into(),
                    observer,
                    parallelize.unwrap_or(true),
                    constraints,
                    initial_positions,
                );
                Ok(result?.into())
            }

            pub fn fit_two_stage(
                &self,
                interpolator: &$name,
                continuum_interpolator: &$name,
                observed_flux: PyArrayLike<FluxFloat, Ix1>,
                observed_var: PyArrayLike<FluxFloat, Ix1>,
                settings1: PSOSettings,
                settings2: PSOSettings,
                trace_directory: Option<String>,
                best_particle_file: Option<String>,
                parallelize: Option<bool>,
                constraints: Option<Vec<PyConstraintWrapper>>,
                resume: Option<bool>,
            ) -> PyResult<BinaryOptimizationResult> {
                let observed_spectrum = ObservedSpectrum {
                    flux: observed_flux.as_matrix().column(0).into_owned(),
                    var: observed_var.as_matrix().column(0).into_owned(),
                };
                let constraints = match constraints {
                    Some(constraints) => constraints
                        .into_iter()
                        .map(|constraint| constraint.0)
                        .collect(),
                    None => vec![],
                };
                let (observer, initial_positions) = get_observer_and_init(
                    trace_directory,
                    best_particle_file,
                    resume.unwrap_or(false),
                );
                let result = self.0.fit_two_stage(
                    &interpolator.0,
                    &continuum_interpolator.0,
                    &observed_spectrum.into(),
                    settings1.into(),
                    settings2.into(),
                    observer,
                    parallelize.unwrap_or(true),
                    constraints,
                    initial_positions,
                );
                Ok(result?.into())
            }

            pub fn fit_two_stage_bulk(
                &self,
                interpolator: &$name,
                continuum_interpolator: &$name,
                observed_fluxes: Vec<Vec<FluxFloat>>,
                observed_vars: Vec<Vec<FluxFloat>>,
                settings1: PSOSettings,
                settings2: PSOSettings,
                constraints: Option<Vec<Vec<PyConstraintWrapper>>>,
            ) -> PyResult<Vec<BinaryOptimizationResult>> {
                let n = observed_fluxes.len();
                let constraints = match constraints {
                    Some(constraints) => constraints
                        .into_iter()
                        .map(|constraint| {
                            constraint
                                .into_iter()
                                .map(|constraint| constraint.0)
                                .collect()
                        })
                        .collect(),
                    None => vec![vec![]; n],
                };
                Ok(observed_fluxes
                    .into_par_iter()
                    .zip(observed_vars)
                    .zip(constraints)
                    .map(|((flux, var), constraint)| {
                        let observed_spectrum = ObservedSpectrum::from_vecs(flux, var);
                        self.0
                            .fit_two_stage(
                                &interpolator.0,
                                &continuum_interpolator.0,
                                &observed_spectrum.into(),
                                settings1.clone().into(),
                                settings2.clone().into(),
                                Observer::DummyObserver,
                                false,
                                constraint,
                                None,
                            )
                            .unwrap()
                            .into()
                    })
                    .collect::<Vec<_>>())
            }

            pub fn chi2(
                &self,
                interpolator: &$name,
                continuum_interpolator: &$name,
                observed_flux: Vec<FluxFloat>,
                observed_var: Vec<FluxFloat>,
                labels: Vec<[f64; 11]>,
                allow_nan: Option<bool>,
            ) -> PyResult<Vec<f64>> {
                let observed_spectrum = ObservedSpectrum::from_vecs(observed_flux, observed_var);
                Ok(labels
                    .into_par_iter()
                    .map(|label| {
                        let chi = self.0.chi2(
                            &interpolator.0,
                            &continuum_interpolator.0,
                            &observed_spectrum,
                            label.into(),
                        );
                        match allow_nan {
                            Some(false) => chi.unwrap(),
                            _ => match chi {
                                Ok(x) => x,
                                Err(_) => f64::NAN,
                            },
                        }
                    })
                    .collect())
            }
        }

        #[pyclass]
        pub struct $PyRVFitter(BinaryRVFitter<ContinuumFitterWrapper, WavelengthDispersionWrapper>);

        #[pymethods]
        impl $PyRVFitter {
            pub fn fit(
                &self,
                model1: Vec<FluxFloat>,
                model2: Vec<FluxFloat>,
                continuum1: Vec<FluxFloat>,
                continuum2: Vec<FluxFloat>,
                light_ratio: f64,
                observed_flux: Vec<FluxFloat>,
                observed_var: Vec<FluxFloat>,
                settings: PSOSettings,
                trace_directory: Option<String>,
                best_particle_file: Option<String>,
                constraints: Option<Vec<PyConstraintWrapper>>,
            ) -> PyResult<RVOptimizationResult> {
                let observed_spectrum = ObservedSpectrum::from_vecs(observed_flux, observed_var);
                let constraints = match constraints {
                    Some(constraints) => constraints
                        .into_iter()
                        .map(|constraint| constraint.0)
                        .collect(),
                    None => vec![],
                };
                let result = self.0.fit(
                    &model1.into(),
                    &model2.into(),
                    &continuum1.into(),
                    &continuum2.into(),
                    light_ratio,
                    &observed_spectrum,
                    settings.into(),
                    get_observer_and_init::<2>(trace_directory, best_particle_file, false).0,
                    constraints,
                );
                Ok(result?.into())
            }

            pub fn chi2(
                &self,
                model1: Vec<FluxFloat>,
                model2: Vec<FluxFloat>,
                continuum1: Vec<FluxFloat>,
                continuum2: Vec<FluxFloat>,
                light_ratio: f64,
                observed_flux: Vec<FluxFloat>,
                observed_var: Vec<FluxFloat>,
                labels: Vec<[f64; 2]>,
                allow_nan: Option<bool>,
            ) -> PyResult<Vec<f64>> {
                let observed_spectrum = ObservedSpectrum::from_vecs(observed_flux, observed_var);
                let model1 = ArraySegment::from(model1);
                let model2 = ArraySegment::from(model2);
                let continuum1 = ArraySegment::from(continuum1);
                let continuum2 = ArraySegment::from(continuum2);
                Ok(labels
                    .into_par_iter()
                    .map(|label| {
                        let chi = self.0.chi2(
                            &model1,
                            &model2,
                            &continuum1,
                            &continuum2,
                            light_ratio,
                            &observed_spectrum,
                            label.into(),
                        );
                        match allow_nan {
                            Some(false) => chi.unwrap(),
                            _ => match chi {
                                Ok(x) => x,
                                Err(_) => f64::NAN,
                            },
                        }
                    })
                    .collect())
            }
        }

        #[pyclass]
        pub struct $PyTSRVFitter(
            BinaryTimeriesKnownRVFitter<WavelengthDispersionWrapper, ContinuumFitterWrapper>,
        );

        #[pymethods]
        impl $PyTSRVFitter {
            pub fn fit(
                &self,
                interpolator: &$name,
                continuum_interpolator: &$name,
                observed_fluxes: Vec<Vec<FluxFloat>>,
                observed_vars: Vec<Vec<FluxFloat>>,
                settings: PSOSettings,
                rvs: Vec<[f64; 2]>,
                trace_directory: Option<String>,
                best_particle_file: Option<String>,
                parallelize: Option<bool>,
                constraints: Option<Vec<PyConstraintWrapper>>,
            ) -> PyResult<BinaryTimeseriesOptimizationResult> {
                let constraints = match constraints {
                    Some(constraints) => constraints
                        .into_iter()
                        .map(|constraint| constraint.0)
                        .collect(),
                    None => vec![],
                };
                let observed_spectra = observed_fluxes
                    .into_iter()
                    .zip(observed_vars)
                    .map(|(flux, var)| ObservedSpectrum::from_vecs(flux, var))
                    .collect();
                let result = self.0.fit(
                    &interpolator.0,
                    &continuum_interpolator.0,
                    &observed_spectra,
                    settings.into(),
                    &rvs,
                    get_observer_and_init::<9>(trace_directory, best_particle_file, false).0,
                    parallelize.unwrap_or(true),
                    constraints,
                );
                Ok(result?.into())
            }

            pub fn chi2(
                &self,
                interpolator: &$name,
                continuum_interpolator: &$name,
                observed_fluxes: Vec<Vec<FluxFloat>>,
                observed_vars: Vec<Vec<FluxFloat>>,
                rvs: Vec<[f64; 2]>,
                labels: Vec<[f64; 9]>,
                allow_nan: Option<bool>,
            ) -> PyResult<Vec<f64>> {
                let observed_spectra = observed_fluxes
                    .into_iter()
                    .zip(observed_vars)
                    .map(|(flux, var)| ObservedSpectrum::from_vecs(flux, var))
                    .collect();
                Ok(labels
                    .into_par_iter()
                    .map(|label| {
                        let chi = self.0.chi2(
                            &interpolator.0,
                            &continuum_interpolator.0,
                            &observed_spectra,
                            &rvs,
                            label.into(),
                        );
                        match allow_nan {
                            Some(false) => chi.unwrap(),
                            _ => match chi {
                                Ok(x) => x,
                                Err(_) => f64::NAN,
                            },
                        }
                    })
                    .collect())
            }
        }
    };
}

#[derive(Clone, Debug)]
#[pyclass(module = "pasta", frozen)]
pub struct PyInterpolator();

/// Interpolator that loads every spectrum from disk every time.
#[pyclass]
#[derive(Clone)]
pub struct OnDiskInterpolator(GridInterpolator<OnDiskFetcher>);

#[pymethods]
impl OnDiskInterpolator {
    #[new]
    pub fn new(dir: &str, includes_factor: bool, wavelength: WlGrid) -> PyResult<Self> {
        let fetcher = OnDiskFetcher::new(dir, includes_factor)?;
        Ok(Self(GridInterpolator::new(fetcher, wavelength.0)))
    }
}

/// Interpolator that loads every grid spectrum into memory in the beginning.
#[pyclass]
pub struct InMemInterpolator(GridInterpolator<InMemFetcher>);

#[pymethods]
impl InMemInterpolator {
    #[new]
    fn new(dir: &str, includes_factor: bool, wavelength: WlGrid) -> PyResult<InMemInterpolator> {
        let fetcher = InMemFetcher::new(dir, includes_factor)?;
        Ok(InMemInterpolator(GridInterpolator::new(
            fetcher,
            wavelength.0,
        )))
    }

    #[staticmethod]
    fn from_tar_zstd(
        file_path: &str,
        includes_factor: bool,
        wavelength: WlGrid,
    ) -> PyResult<InMemInterpolator> {
        let fetcher = InMemFetcher::from_tar_zstd(PathBuf::from(file_path), includes_factor)?;
        Ok(InMemInterpolator(GridInterpolator::new(
            fetcher,
            wavelength.0,
        )))
    }
}
/// Interpolator that caches the last _lrucap_ spectra
#[pyclass]
#[derive(Clone)]
pub struct CachedInterpolator(GridInterpolator<CachedFetcher>);

#[pymethods]
impl CachedInterpolator {
    #[new]
    pub fn new(
        dir: &str,
        includes_factor: bool,
        wavelength: WlGrid,
        lrucap: Option<usize>,
        n_shards: Option<usize>,
    ) -> PyResult<Self> {
        let fetcher = CachedFetcher::new(
            dir,
            includes_factor,
            lrucap.unwrap_or(4000),
            n_shards.unwrap_or(4),
        )?;
        Ok(Self(GridInterpolator::new(fetcher, wavelength.0)))
    }
    pub fn cache_size(&self) -> usize {
        self.0.fetcher.cache_size()
    }
}

implement_methods!(
    OnDiskInterpolator,
    OnDiskFetcher,
    OnDiskSingleFitter,
    OnDiskBinaryFitter,
    OnDiskRVFitter,
    OnDiskTSKnownRVFitter
);
implement_methods!(
    InMemInterpolator,
    InMemFetcher,
    InMemSingleFitter,
    InMemBinaryFitter,
    InMemRVFitter,
    InMemTSKnownRVFitter
);
implement_methods!(
    CachedInterpolator,
    CachedFetcher,
    CachedSingleFitter,
    CachedBinaryFitter,
    CachedRVFitter,
    CachedTSKnownRVFitter
);

#[pyfunction]
pub fn get_vsini_kernel(vsini: f64, synth_wl: WlGrid) -> Vec<FluxFloat> {
    let dvelo = match synth_wl.0 {
        interpolate::WlGrid::Logspace(_, step, _) => std::f64::consts::LN_10 * step,
        _ => panic!("Only logspace is supported"),
    };
    convolve_rv::build_rotation_kernel(vsini, dvelo).data.into()
}

#[pymodule]
fn pasta(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OnDiskInterpolator>()?;
    m.add_class::<InMemInterpolator>()?;
    m.add_class::<CachedInterpolator>()?;
    m.add_class::<WlGrid>()?;
    m.add_class::<PSOSettings>()?;
    m.add_class::<Label>()?;
    m.add_class::<OptimizationResult>()?;
    m.add_class::<BinaryOptimizationResult>()?;
    m.add_class::<BinaryTimeseriesOptimizationResult>()?;
    m.add_function(wrap_pyfunction!(NoConvolutionDispersion, m)?)?;
    m.add_function(wrap_pyfunction!(FixedResolutionDispersion, m)?)?;
    m.add_function(wrap_pyfunction!(VariableResolutionDispersion, m)?)?;
    m.add_function(wrap_pyfunction!(LinearModelContinuumFitter, m)?)?;
    m.add_function(wrap_pyfunction!(ChunkContinuumFitter, m)?)?;
    m.add_function(wrap_pyfunction!(FixedContinuum, m)?)?;
    m.add_function(wrap_pyfunction!(ConstantContinuum, m)?)?;
    m.add_function(wrap_pyfunction!(FixConstraint, m)?)?;
    m.add_function(wrap_pyfunction!(RangeConstraint, m)?)?;
    m.add_function(wrap_pyfunction!(get_vsini_kernel, m)?)?;
    Ok(())
}
