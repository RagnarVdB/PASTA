use crate::bounds::{
    BinaryBounds, BinaryBoundsWithoutRV, BinaryRVBounds, BoundsConstraint, PSOBounds, SingleBounds,
};
use crate::continuum_fitting::ContinuumFitter;
use crate::convolve_rv::{
    shift_and_resample, shift_resample_and_add_binary_components, ArraySegment, WavelengthDispersion
};
use crate::interpolate::{FluxFloat, Interpolator, WlGrid};
use crate::particleswarm::{self};
use anyhow::{anyhow, bail, Context, Result};
use argmin::core::observers::{Observe, ObserverMode};
use argmin::core::{CostFunction as _, Executor, PopulationState, State, KV};
use argmin::solver::brent::BrentRoot;
use itertools::Itertools;
use nalgebra::{self as na, Storage};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

/// Observed specrum with flux and variance
#[derive(Clone, Debug)]
pub struct ObservedSpectrum {
    pub flux: na::DVector<FluxFloat>,
    pub var: na::DVector<FluxFloat>,
}

impl ObservedSpectrum {
    /// Load observed spectrum from vector of flux and vector of variance
    pub fn from_vecs(flux: Vec<FluxFloat>, var: Vec<FluxFloat>) -> Self {
        let flux = na::DVector::from_vec(flux);
        let var = na::DVector::from_vec(var);
        if flux.len() != var.len() {
            panic!("Flux and variance vectors must have the same length");
        }
        Self { flux, var }
    }
}

#[derive(Clone, Debug)]
pub struct Label<T> {
    pub teff: T,
    pub m: T,
    pub logg: T,
    pub vsini: T,
    pub rv: T,
}

impl<T> Label<T> {
    pub fn as_array(self) -> [T; 5] {
        [self.teff, self.m, self.logg, self.vsini, self.rv]
    }

    pub fn from_array(value: [T; 5]) -> Self {
        let [teff, m, logg, vsini, rv] = value;
        Self {
            teff,
            m,
            logg,
            vsini,
            rv,
        }
    }
}
impl<T: na::Scalar> Label<T> {
    fn as_vector(self) -> na::SVector<T, 5> {
        na::SVector::from_row_slice(self.as_array().as_slice())
    }
}

impl<S: na::Storage<f64, na::Const<5>, na::Const<1>>> From<na::Vector<f64, na::Const<5>, S>>
    for Label<f64>
{
    fn from(value: na::Vector<f64, na::Const<5>, S>) -> Self {
        Self {
            teff: value[0],
            m: value[1],
            logg: value[2],
            vsini: value[3],
            rv: value[4],
        }
    }
}

#[derive(Debug)]
pub struct RvLessLabel<T> {
    pub teff: T,
    pub m: T,
    pub logg: T,
    pub vsini: T,
}

impl<T> RvLessLabel<T> {
    pub fn as_array(self) -> [T; 4] {
        [self.teff, self.m, self.logg, self.vsini]
    }
}

impl<S: na::Storage<f64, na::Const<4>, na::Const<1>>> From<na::Vector<f64, na::Const<4>, S>>
    for RvLessLabel<f64>
{
    fn from(value: na::Vector<f64, na::Const<4>, S>) -> Self {
        Self {
            teff: value[0],
            m: value[1],
            logg: value[2],
            vsini: value[3],
        }
    }
}

#[derive(Debug)]
pub struct OptimizationResult {
    pub label: Label<f64>,
    pub continuum_params: na::DVector<FluxFloat>,
    pub chi2: f64,
    pub iters: u64,
    pub time: f64,
}

pub struct BinaryOptimizationResult {
    pub label1: Label<f64>,
    pub label2: Label<f64>,
    pub light_ratio: f64,
    pub continuum_params: na::DVector<FluxFloat>,
    pub chi2: f64,
    pub iters: u64,
    pub time: f64,
}

pub struct BinaryRVOptimizationResult {
    pub rv1: f64,
    pub rv2: f64,
    pub chi2: f64,
    pub iters: u64,
    pub time: f64,
}

pub struct BinaryTimeseriesOptimizationResult {
    pub label1: RvLessLabel<f64>,
    pub label2: RvLessLabel<f64>,
    pub light_ratio: f64,
    pub continuum_params: Vec<na::DVector<FluxFloat>>,
    pub chis: Vec<f64>,
    pub iters: u64,
    pub time: f64,
}

#[derive(Clone, Debug)]
pub struct PSOSettings {
    pub num_particles: usize,
    pub max_iters: u64,
    pub inertia_factor: f64,
    pub cognitive_factor: f64,
    pub social_factor: f64,
    pub delta: f64,
}

fn setup_pso<const N: usize, B: PSOBounds<N>>(
    bounds: B,
    settings: PSOSettings,
) -> particleswarm::ParticleSwarm<N, B, f64> {
    particleswarm::ParticleSwarm::new(bounds, settings.num_particles)
        .with_inertia_factor(settings.inertia_factor)
        .unwrap()
        .with_cognitive_factor(settings.cognitive_factor)
        .unwrap()
        .with_social_factor(settings.social_factor)
        .unwrap()
        .with_delta(settings.delta)
        .unwrap()
}

pub trait Observer<const N: usize>:
    Observe<PopulationState<particleswarm::Particle<na::SVector<f64, N>, f64>, f64>> + 'static
{
}

impl<const N: usize>
    Observe<PopulationState<particleswarm::Particle<na::SVector<f64, N>, f64>, f64>>
    for Box<dyn Observer<N>>
{
    fn observe_iter(
        &mut self,
        state: &PopulationState<particleswarm::Particle<na::SVector<f64, N>, f64>, f64>,
        kv: &KV,
    ) -> std::result::Result<(), anyhow::Error> {
        self.as_mut().observe_iter(state, kv)
    }
}

impl<const N: usize> Observer<N> for Box<dyn Observer<N>> {}

pub struct DummyObserver<const N: usize>();

impl<const N: usize>
    Observe<PopulationState<particleswarm::Particle<na::SVector<f64, N>, f64>, f64>>
    for DummyObserver<N>
{
    fn observe_iter(
        &mut self,
        _state: &PopulationState<particleswarm::Particle<na::SVector<f64, N>, f64>, f64>,
        _kv: &KV,
    ) -> Result<()> {
        Ok(())
    }
}

impl<const N: usize> Observer<N> for DummyObserver<N> {}

#[derive(Serialize)]
struct ParticleInfo {
    position: Vec<f64>,
    cost: f64,
}

pub struct PSOobserver {
    dir: PathBuf,
    file_prefix: String,
}

impl PSOobserver {
    pub fn new(directory: &str, file_prefix: &str) -> Self {
        Self {
            dir: PathBuf::from(directory),
            file_prefix: file_prefix.to_string(),
        }
    }
}

impl<const N: usize>
    Observe<PopulationState<particleswarm::Particle<na::SVector<f64, N>, f64>, f64>>
    for PSOobserver
{
    fn observe_iter(
        &mut self,
        state: &PopulationState<particleswarm::Particle<na::SVector<f64, N>, f64>, f64>,
        _kv: &KV,
    ) -> Result<()> {
        let iter = state.get_iter();
        let particles = state
            .get_population()
            .ok_or(argmin::core::Error::msg("No particles"))?;
        let values = particles
            .iter()
            .map(|p| ParticleInfo {
                position: p.position.data.as_slice().to_owned(),
                cost: p.cost,
            })
            .collect::<Vec<_>>();
        let filename = self.dir.join(format!("{}_{}.json", self.file_prefix, iter));
        let f = BufWriter::new(File::create(filename)?);
        serde_json::to_writer(f, &values)?;
        Ok(())
    }
}

impl<const N: usize> Observer<N> for PSOobserver {}

pub struct BestParticleObserver<const N: usize> {
    file: PathBuf,
    best_particles: Vec<ParticleInfo>,
}

impl<const N: usize> BestParticleObserver<N> {
    pub fn new(file: PathBuf) -> Self {
        Self {
            file,
            best_particles: Vec::new(),
        }
    }
}

impl<const N: usize>
    Observe<PopulationState<particleswarm::Particle<na::SVector<f64, N>, f64>, f64>>
    for BestParticleObserver<N>
{
    fn observe_iter(
        &mut self,
        state: &PopulationState<particleswarm::Particle<na::SVector<f64, N>, f64>, f64>,
        _kv: &KV,
    ) -> Result<()> {
        let best_particle = state
            .get_best_param()
            .ok_or(argmin::core::Error::msg("No best particle"))?;
        let particle_info = ParticleInfo {
            position: best_particle.position.data.as_slice().to_owned(),
            cost: best_particle.cost,
        };
        self.best_particles.push(particle_info);
        serde_json::to_writer(&File::create(&self.file)?, &self.best_particles)?;
        Ok(())
    }
}

impl<const N: usize> Observer<N> for BestParticleObserver<N> {}

/// Cost function used in the PSO fitting
struct CostFunction<'a, I: Interpolator, T: WavelengthDispersion, F: ContinuumFitter> {
    interpolator: &'a I,
    target_dispersion: &'a T,
    observed_spectrum: &'a ObservedSpectrum,
    continuum_fitter: &'a F,
    parallelize: bool,
}

impl<I: Interpolator, T: WavelengthDispersion, F: ContinuumFitter> argmin::core::CostFunction
    for CostFunction<'_, I, T, F>
{
    type Param = na::SVector<f64, 5>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output> {
        let teff = params[0];
        let m = params[1];
        let logg = params[2];
        let vsini = params[3];
        let rv = params[4];
        let synth_spec =
            self.interpolator
                .produce_model(self.target_dispersion, teff, m, logg, vsini, rv)?;
        let (_, ls) = self
            .continuum_fitter
            .fit_continuum(self.observed_spectrum, &synth_spec)?;
        Ok(ls)
    }

    fn parallelize(&self) -> bool {
        self.parallelize
    }
}

struct UncertaintyCostFunction<'a, I: Interpolator, T: WavelengthDispersion, F: ContinuumFitter> {
    interpolator: &'a I,
    target_dispersion: &'a T,
    observed_spectrum: &'a ObservedSpectrum,
    continuum_fitter: &'a F,
    label: &'a na::SVector<f64, 5>,
    label_index: usize,
    target_value: f64,
    search_radius: f64,
}

impl<I: Interpolator, T: WavelengthDispersion, F: ContinuumFitter> argmin::core::CostFunction
    for UncertaintyCostFunction<'_, I, T, F>
{
    type Param = f64;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output> {
        let mut new_label = *self.label;
        new_label[self.label_index] += param * self.search_radius;
        let synth_spec = self
            .interpolator
            .produce_model(
                self.target_dispersion,
                new_label[0],
                new_label[1],
                new_label[2],
                new_label[3],
                new_label[4],
            )
            .context(format!(
                "Error computing cost with param={:?}, new_label={:?}",
                param, new_label
            ))?;
        let (_, chi2) = self
            .continuum_fitter
            .fit_continuum(self.observed_spectrum, &synth_spec)?;
        Ok(chi2 - self.target_value)
    }
}

pub struct SingleFitter<T: WavelengthDispersion, F: ContinuumFitter> {
    target_dispersion: T,
    continuum_fitter: F,
    settings: PSOSettings,
    vsini_range: (f64, f64),
    rv_range: (f64, f64),
}

impl<T: WavelengthDispersion, F: ContinuumFitter> SingleFitter<T, F> {
    pub fn new(
        target_dispersion: T,
        continuum_fitter: F,
        settings: PSOSettings,
        vsini_range: (f64, f64),
        rv_range: (f64, f64),
    ) -> Self {
        Self {
            target_dispersion,
            continuum_fitter,
            settings,
            vsini_range,
            rv_range,
        }
    }

    pub fn fit(
        &self,
        interpolator: &impl Interpolator,
        observed_spectrum: &ObservedSpectrum,
        observer: impl Observer<5>,
        parallelize: bool,
        constraints: Vec<BoundsConstraint>,
    ) -> Result<OptimizationResult> {
        if observed_spectrum.flux.len() != self.target_dispersion.wavelength().len() {
            bail!("Observed spectrum and target dispersion must have the same length");
        }
        let cost_function = CostFunction {
            interpolator,
            target_dispersion: &self.target_dispersion,
            observed_spectrum,
            continuum_fitter: &self.continuum_fitter,
            parallelize,
        };
        let bounds = SingleBounds::new(interpolator.grid_bounds(), self.vsini_range, self.rv_range)
            .with_constraints(constraints);
        let solver = setup_pso(bounds, self.settings.clone());
        let fitter = Executor::new(cost_function, solver)
            .configure(|state| state.max_iters(self.settings.max_iters));
        let result = fitter.add_observer(observer, ObserverMode::Always).run()?;

        let best_param = result
            .state
            .best_individual
            .ok_or(anyhow!("No best parameter found"))?
            .position;

        let teff = best_param[0];
        let m = best_param[1];
        let logg = best_param[2];
        let vsini = best_param[3];
        let rv = best_param[4];

        let synth_spec =
            interpolator.produce_model(&self.target_dispersion, teff, m, logg, vsini, rv)?;
        let (continuum_params, _) = self
            .continuum_fitter
            .fit_continuum(observed_spectrum, &synth_spec)?;
        let time = match result.state.time {
            Some(t) => t.as_secs_f64(),
            None => 0.0,
        };
        Ok(OptimizationResult {
            label: Label {
                teff,
                m,
                logg,
                vsini,
                rv,
            },
            continuum_params,
            chi2: result.state.best_cost,
            time,
            iters: result.state.iter,
        })
    }

    pub fn chi2(
        &self,
        interpolator: &impl Interpolator,
        observed_spectrum: &ObservedSpectrum,
        label: na::SVector<f64, 5>,
    ) -> Result<f64> {
        if observed_spectrum.flux.len() != self.target_dispersion.wavelength().len() {
            bail!("Observed spectrum and target dispersion must have the same length");
        }
        let cost_function = CostFunction {
            interpolator,
            target_dispersion: &self.target_dispersion,
            observed_spectrum,
            continuum_fitter: &self.continuum_fitter,
            parallelize: false,
        };

        cost_function.cost(&label)
    }

    /// Compute uncertainty in every label
    /// by finding the intersection points of the chi2 function with a target value.
    pub fn compute_uncertainty(
        &self,
        interpolator: &impl Interpolator,
        observed_spectrum: &ObservedSpectrum,
        spec_res: f64,
        label: Label<f64>,
        search_radius: Label<f64>,
    ) -> Result<Label<(Result<f64>, Result<f64>)>> {
        if observed_spectrum.flux.len() != self.target_dispersion.wavelength().len() {
            bail!("Observed spectrum and target dispersion must have the same length");
        }
        let mut search_radius = search_radius.as_vector();
        let label = label.as_vector();
        search_radius[0] *= label[0];
        search_radius[3] = (label[3] * search_radius[3]).max(100.0);
        let best_synth_spec = interpolator.produce_model(
            &self.target_dispersion,
            label[0],
            label[1],
            label[2],
            label[3],
            label[4],
        )?;
        let (_, best_chi2) = self
            .continuum_fitter
            .fit_continuum(observed_spectrum, &best_synth_spec)?;
        let observed_wavelength = self.target_dispersion.wavelength();
        let n_p = observed_wavelength.len();
        let first_wl = observed_wavelength[0];
        let last_wl = observed_wavelength[n_p - 1];
        let n = 4.0 * spec_res * (last_wl - first_wl) / (first_wl + last_wl);
        let target_chi = best_chi2 * (1.0 + (2.0 / n).sqrt());

        let computer = |i| {
            // Maybe don't hardcode here
            let bounds = SingleBounds::new(interpolator.grid_bounds(), (0.0, 1e4), (-1e3, 1e3));

            let get_bound = |right: bool| {
                let costfunction = UncertaintyCostFunction {
                    interpolator,
                    target_dispersion: &self.target_dispersion,
                    observed_spectrum,
                    continuum_fitter: &self.continuum_fitter,
                    label: &label,
                    label_index: i,
                    target_value: target_chi,
                    search_radius: search_radius[i],
                };
                let mut bound_label = label;
                if right {
                    bound_label[i] += search_radius[i];
                } else {
                    bound_label[i] -= search_radius[i];
                }
                let bound = bounds.clamp_1d(bound_label, i)?;
                let bound_rescaled = (bound - label[i]) / search_radius[i];
                let solver = if right {
                    BrentRoot::new(0.0, bound_rescaled.min(1.0), 1e-3)
                } else {
                    BrentRoot::new(bound_rescaled, 0.0, 1e-3)
                };
                let executor = Executor::new(costfunction, solver);
                let result = executor.run().context(format!(
                    "Error while running uncertainty_chi2 on {:?}, bound={}",
                    label, bound
                ))?;
                let sol = result
                    .state
                    .get_best_param()
                    .context("result.state.get_best_param()")?;
                if right {
                    Ok(sol * search_radius[i])
                } else {
                    Ok(-sol * search_radius[i]) // Minus because we are looking for the negative of the solution
                }
            };

            (get_bound(false), get_bound(true))
        };
        Ok(Label {
            teff: computer(0),
            m: computer(1),
            logg: computer(2),
            vsini: computer(3),
            rv: computer(4),
        })
    }
}

struct BinaryCostFunction<'a, I: Interpolator, T: WavelengthDispersion, F: ContinuumFitter> {
    interpolator: &'a I,
    continuum_interpolator: &'a I,
    target_dispersion: &'a T,
    observed_spectrum: &'a ObservedSpectrum,
    continuum_fitter: &'a F,
    parallelize: bool,
}

impl<I: Interpolator, T: WavelengthDispersion, F: ContinuumFitter> argmin::core::CostFunction
    for BinaryCostFunction<'_, I, T, F>
{
    type Param = na::SVector<f64, 11>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output> {
        let star1_parameters = params.fixed_rows::<5>(0).into_owned();
        let star2_parameters = params.fixed_rows::<5>(5).into_owned();
        let light_ratio = params[10];

        let synth_spec = self.interpolator.produce_binary_model_norm(
            self.continuum_interpolator,
            self.target_dispersion,
            &star1_parameters,
            &star2_parameters,
            light_ratio as f32,
        )?;
        let (_, ls) = self
            .continuum_fitter
            .fit_continuum(self.observed_spectrum, &synth_spec)?;
        Ok(ls)
    }

    fn parallelize(&self) -> bool {
        self.parallelize
    }
}

pub struct BinaryFitter<T: WavelengthDispersion, F: ContinuumFitter> {
    target_dispersion: T,
    continuum_fitter: F,
    settings: PSOSettings,
    vsini_range: (f64, f64),
    rv_range: (f64, f64),
}

impl<T: WavelengthDispersion, F: ContinuumFitter> BinaryFitter<T, F> {
    pub fn new(
        target_dispersion: T,
        continuum_fitter: F,
        settings: PSOSettings,
        vsini_range: (f64, f64),
        rv_range: (f64, f64),
    ) -> Self {
        Self {
            target_dispersion,
            continuum_fitter,
            settings,
            vsini_range,
            rv_range,
        }
    }

    pub fn fit<I: Interpolator, O: Observer<11>>(
        &self,
        interpolator: &I,
        continuum_interpolator: &I,
        observed_spectrum: &ObservedSpectrum,
        observer: O,
        parallelize: bool,
        constraints: Vec<BoundsConstraint>,
    ) -> Result<BinaryOptimizationResult> {
        let cost_function = BinaryCostFunction {
            interpolator,
            continuum_interpolator,
            target_dispersion: &self.target_dispersion,
            observed_spectrum,
            continuum_fitter: &self.continuum_fitter,
            parallelize,
        };
        let bounds = BinaryBounds::new(
            interpolator.grid_bounds(),
            (0.0, 1.0),
            self.vsini_range,
            self.rv_range,
        )
        .with_constraints(constraints);
        let solver = setup_pso(bounds, self.settings.clone());
        let fitter = Executor::new(cost_function, solver)
            .configure(|state| state.max_iters(self.settings.max_iters));

        let result = fitter.add_observer(observer, ObserverMode::Always).run()?;

        let best_param = result
            .state
            .best_individual
            .ok_or(anyhow!("No best parameter found"))?
            .position;

        let star1_parameters = best_param.fixed_rows::<5>(0).into_owned();
        let star2_parameters = best_param.fixed_rows::<5>(5).into_owned();
        let light_ratio = best_param[10];

        let synth_spec = interpolator.produce_binary_model_norm(
            continuum_interpolator,
            &self.target_dispersion,
            &star1_parameters,
            &star2_parameters,
            light_ratio as f32,
        )?;
        let (continuum_params, _) = self
            .continuum_fitter
            .fit_continuum(observed_spectrum, &synth_spec)?;
        let time = match result.state.time {
            Some(t) => t.as_secs_f64(),
            None => 0.0,
        };
        Ok(BinaryOptimizationResult {
            label1: best_param.fixed_rows::<5>(0).into(),
            label2: best_param.fixed_rows::<5>(5).into(),
            light_ratio: best_param[10],
            continuum_params,
            chi2: result.state.best_cost,
            time,
            iters: result.state.iter,
        })
    }

    pub fn chi2<I: Interpolator>(
        &self,
        interpolator: &I,
        continuum_interpolator: &I,
        observed_spectrum: &ObservedSpectrum,
        label: na::SVector<f64, 11>,
    ) -> Result<f64> {
        let cost_function = BinaryCostFunction {
            interpolator,
            continuum_interpolator,
            target_dispersion: &self.target_dispersion,
            observed_spectrum,
            continuum_fitter: &self.continuum_fitter,
            parallelize: false,
        };

        cost_function.cost(&label)
    }
}

#[derive(Clone)]
struct RVCostFunction<'a, F: ContinuumFitter, D: WavelengthDispersion> {
    continuum_fitter: &'a F,
    observed_dispersion: &'a D,
    observed_spectrum: &'a ObservedSpectrum,
    synth_wl: &'a WlGrid,
    model1: &'a ArraySegment,
    model2: &'a ArraySegment,
    continuum1: &'a ArraySegment,
    continuum2: &'a ArraySegment,
    light_ratio: f64,
}

impl<F: ContinuumFitter, D: WavelengthDispersion> argmin::core::CostFunction
    for RVCostFunction<'_, F, D>
{
    type Param = na::SVector<f64, 2>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        let rv1 = param[0];
        let rv2 = param[1];

        let shifted_synth1 = shift_and_resample(
            self.synth_wl,
            &self.model1.clone(),
            self.observed_dispersion,
            rv1,
        )?;
        let shifted_synth2 = shift_and_resample(
            self.synth_wl,
            &self.model2.clone(),
            self.observed_dispersion,
            rv2,
        )?;
        let shifted_continuum1 = shift_and_resample(
            self.synth_wl,
            &self.continuum1.clone(),
            self.observed_dispersion,
            rv1,
        )?;
        let shifted_continuum2 = shift_and_resample(
            self.synth_wl,
            &self.continuum2.clone(),
            self.observed_dispersion,
            rv2,
        )?;

        let lr =
            self.light_ratio as FluxFloat * shifted_continuum2.mean() / shifted_continuum1.mean();

        let synth_spec = shifted_synth1.component_mul(&shifted_continuum1)
            + shifted_synth2.component_mul(&shifted_continuum2) * lr;
        let continuum = shifted_continuum1 + shifted_continuum2 * lr;
        let synth_spec_norm = synth_spec.component_div(&continuum);

        let (_, ls) = self
            .continuum_fitter
            .fit_continuum(self.observed_spectrum, &synth_spec_norm)?;
        Ok(ls)
    }

    fn parallelize(&self) -> bool {
        false
    }
}

pub struct BinaryRVFitter<F: ContinuumFitter, D: WavelengthDispersion> {
    observed_dispersion: D,
    synth_wl: WlGrid,
    continuum_fitter: F,
    settings: PSOSettings,
    rv_range: (f64, f64),
}

impl<F: ContinuumFitter, D: WavelengthDispersion> BinaryRVFitter<F, D> {
    pub fn new(
        observed_dispersion: D,
        synth_wl: WlGrid,
        continuum_fitter: F,
        settings: PSOSettings,
        rv_range: (f64, f64),
    ) -> Self {
        Self {
            observed_dispersion,
            synth_wl,
            continuum_fitter,
            settings,
            rv_range,
        }
    }

    pub fn fit(
        &self,
        model1: &ArraySegment,
        model2: &ArraySegment,
        continuum1: &ArraySegment,
        continuum2: &ArraySegment,
        light_ratio: f64,
        observed_spectrum: &ObservedSpectrum,
        observer: impl Observer<2>,
        constraints: Vec<BoundsConstraint>,
    ) -> Result<BinaryRVOptimizationResult> {
        let cost_function = RVCostFunction {
            observed_dispersion: &self.observed_dispersion,
            observed_spectrum,
            continuum_fitter: &self.continuum_fitter,
            model1,
            model2,
            continuum1,
            continuum2,
            light_ratio,
            synth_wl: &self.synth_wl,
        };
        let bounds = BinaryRVBounds::new(self.rv_range).with_constraints(constraints);
        let solver = setup_pso(bounds, self.settings.clone());
        let fitter = Executor::new(cost_function.clone(), solver)
            .configure(|state| state.max_iters(self.settings.max_iters));

        let result = fitter.add_observer(observer, ObserverMode::Always).run()?;

        let time = match result.state.time {
            Some(t) => t.as_secs_f64(),
            None => 0.0,
        };
        let best_param = result
            .state
            .best_individual
            .ok_or(anyhow!("No best parameter found"))?
            .position;
        let chi2 = result.state.best_cost;
        let rv1 = best_param[0];
        let rv2 = best_param[1];
        let chi_swapped = cost_function.cost(&na::Vector2::new(rv2, rv1))?;
        if chi_swapped < chi2 {
            Ok(BinaryRVOptimizationResult {
                rv1,
                rv2,
                chi2: chi_swapped,
                time,
                iters: result.state.iter,
            })
        } else {
            Ok(BinaryRVOptimizationResult {
                rv1,
                rv2,
                chi2,
                time,
                iters: result.state.iter,
            })
        }
    }

    pub fn chi2(
        &self,
        model1: &ArraySegment,
        model2: &ArraySegment,
        continuum1: &ArraySegment,
        continuum2: &ArraySegment,
        light_ratio: f64,
        observed_spectrum: &ObservedSpectrum,
        label: na::SVector<f64, 2>,
    ) -> Result<f64> {
        let cost_function = RVCostFunction {
            observed_dispersion: &self.observed_dispersion,
            observed_spectrum,
            continuum_fitter: &self.continuum_fitter,
            model1,
            model2,
            continuum1,
            continuum2,
            light_ratio,
            synth_wl: &self.synth_wl,
        };

        cost_function.cost(&label)
    }
}

struct BinaryTimeseriesKnownRVCostFunction<
    'a,
    I: Interpolator,
    T: WavelengthDispersion,
    F: ContinuumFitter,
> {
    interpolator: &'a I,
    continuum_interpolator: &'a I,
    target_dispersion: &'a T,
    observed_spectra: &'a Vec<ObservedSpectrum>,
    rvs: &'a Vec<[f64; 2]>,
    continuum_fitter: &'a F,
    synth_wl: &'a WlGrid,
    parallelize: bool,
}

impl<I: Interpolator, T: WavelengthDispersion, F: ContinuumFitter> argmin::core::CostFunction
    for BinaryTimeseriesKnownRVCostFunction<'_, I, T, F>
{
    type Param = na::SVector<f64, 9>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output> {
        let star1_parameters = params.fixed_rows::<4>(0).into_owned();
        let star2_parameters = params.fixed_rows::<4>(4).into_owned();
        let light_ratio = params[8] as f32;

        let components = self.interpolator.produce_binary_components(
            self.continuum_interpolator,
            self.target_dispersion,
            &star1_parameters,
            &star2_parameters,
            light_ratio,
        )?;

        let chi2 = self
            .observed_spectra
            .iter()
            .zip(self.rvs.iter())
            .map(|(spec, rv)| {
                let model = shift_resample_and_add_binary_components(
                    self.synth_wl,
                    &components,
                    self.target_dispersion,
                    *rv,
                )?;
                let (_, chi2) = self.continuum_fitter.fit_continuum(spec, &model)?;
                Ok(chi2)
            })
            .sum::<Result<f64>>()?;

        Ok(chi2)
    }

    fn parallelize(&self) -> bool {
        self.parallelize
    }
}

pub struct BinaryTimeriesKnownRVFitter<T: WavelengthDispersion, F: ContinuumFitter> {
    target_dispersion: T,
    synth_wl: WlGrid,
    continuum_fitter: F,
    settings: PSOSettings,
    vsini_range: (f64, f64),
}

impl<T: WavelengthDispersion, F: ContinuumFitter> BinaryTimeriesKnownRVFitter<T, F> {
    pub fn new(
        target_dispersion: T,
        synth_wl: WlGrid,
        continuum_fitter: F,
        settings: PSOSettings,
        vsini_range: (f64, f64),
    ) -> Self {
        Self {
            target_dispersion,
            synth_wl,
            continuum_fitter,
            settings,
            vsini_range,
        }
    }

    pub fn fit<I: Interpolator, O: Observer<9>>(
        &self,
        interpolator: &I,
        continuum_interpolator: &I,
        observed_spectra: &Vec<ObservedSpectrum>,
        rvs: &Vec<[f64; 2]>,
        observer: O,
        parallelize: bool,
        constraints: Vec<BoundsConstraint>,
    ) -> Result<BinaryTimeseriesOptimizationResult> {
        let cost_function = BinaryTimeseriesKnownRVCostFunction {
            interpolator,
            continuum_interpolator,
            target_dispersion: &self.target_dispersion,
            observed_spectra,
            rvs,
            synth_wl: &self.synth_wl,
            continuum_fitter: &self.continuum_fitter,
            parallelize,
        };
        let bounds =
            BinaryBoundsWithoutRV::new(interpolator.grid_bounds(), (0.0, 1.0), self.vsini_range)
                .with_constraints(constraints);
        let solver = setup_pso(bounds, self.settings.clone());
        let fitter = Executor::new(cost_function, solver)
            .configure(|state| state.max_iters(self.settings.max_iters));

        let result = fitter.add_observer(observer, ObserverMode::Always).run()?;

        let best_param = result
            .state
            .best_individual
            .ok_or(anyhow!("No best parameter found"))?
            .position;

        let star1_parameters = best_param.fixed_rows::<4>(0).into_owned();
        let star2_parameters = best_param.fixed_rows::<4>(4).into_owned();
        let light_ratio = best_param[8];

        let components = interpolator.produce_binary_components(
            continuum_interpolator,
            &self.target_dispersion,
            &star1_parameters,
            &star2_parameters,
            light_ratio as f32,
        )?;

        let (continuum_params, chis) = observed_spectra
            .iter()
            .zip(rvs.iter())
            .map(|(spec, rv)| {
                let model = shift_resample_and_add_binary_components(
                    &self.synth_wl,
                    &components,
                    &self.target_dispersion,
                    *rv,
                )?;
                self.continuum_fitter.fit_continuum(spec, &model)
            })
            .process_results(|iter| iter.unzip())?;

        let time = match result.state.time {
            Some(t) => t.as_secs_f64(),
            None => 0.0,
        };
        Ok(BinaryTimeseriesOptimizationResult {
            label1: star1_parameters.into(),
            label2: star2_parameters.into(),
            light_ratio: best_param[8],
            continuum_params,
            chis,
            time,
            iters: result.state.iter,
        })
    }

    pub fn chi2<I: Interpolator>(
        &self,
        interpolator: &I,
        continuum_interpolator: &I,
        observed_spectra: &Vec<ObservedSpectrum>,
        rvs: &Vec<[f64; 2]>,
        label: na::SVector<f64, 9>,
    ) -> Result<f64> {
        let cost_function = BinaryTimeseriesKnownRVCostFunction {
            interpolator,
            continuum_interpolator,
            target_dispersion: &self.target_dispersion,
            observed_spectra,
            rvs,
            synth_wl: &self.synth_wl,
            continuum_fitter: &self.continuum_fitter,
            parallelize: false,
        };
        cost_function.cost(&label)
    }
}

struct BinaryTimeseriesCostFunction<
    'a,
    I: Interpolator,
    T: WavelengthDispersion,
    F: ContinuumFitter,
    B: PSOBounds<2>,
> {
    interpolator: &'a I,
    continuum_interpolator: &'a I,
    target_dispersion: &'a T,
    observed_spectra: &'a Vec<ObservedSpectrum>,
    continuum_fitter: &'a F,
    synth_wl: &'a WlGrid,
    rv_fit_settings: PSOSettings,
    rv_bounds: B,
    parallelize: bool,
}

impl<I: Interpolator, T: WavelengthDispersion, F: ContinuumFitter, B: PSOBounds<2>>
    argmin::core::CostFunction for BinaryTimeseriesCostFunction<'_, I, T, F, B>
{
    type Param = na::SVector<f64, 9>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output> {
        let star1_parameters = params.fixed_rows::<4>(0).into_owned();
        let star2_parameters = params.fixed_rows::<4>(5).into_owned();
        let light_ratio = params[8];

        let synth_spec1 = self.interpolator.interpolate_and_convolve(
            self.target_dispersion,
            star1_parameters[0],
            star1_parameters[1],
            star1_parameters[2],
            star1_parameters[3],
        )?;
        let continuum1 = self.continuum_interpolator.interpolate_and_convolve(
            self.target_dispersion,
            star1_parameters[0],
            star1_parameters[1],
            star1_parameters[2],
            star1_parameters[3],
        )?;
        let synth_spec2 = self.interpolator.interpolate_and_convolve(
            self.target_dispersion,
            star2_parameters[0],
            star2_parameters[1],
            star2_parameters[2],
            star2_parameters[3],
        )?;
        let continuum2 = self.continuum_interpolator.interpolate_and_convolve(
            self.target_dispersion,
            star2_parameters[0],
            star2_parameters[1],
            star2_parameters[2],
            star2_parameters[3],
        )?;

        let chi2 = self
            .observed_spectra
            .iter()
            .map(|observed_spectrum| {
                let inner_cost_function = RVCostFunction {
                    continuum_fitter: self.continuum_fitter,
                    observed_dispersion: self.target_dispersion,
                    observed_spectrum,
                    synth_wl: self.synth_wl,
                    model1: &synth_spec1,
                    model2: &synth_spec2,
                    continuum1: &continuum1,
                    continuum2: &continuum2,
                    light_ratio,
                };
                let solver = setup_pso(self.rv_bounds.clone(), self.rv_fit_settings.clone());
                let fitter = Executor::new(inner_cost_function, solver)
                    .configure(|state| state.max_iters(self.rv_fit_settings.max_iters));
                let result = fitter.run()?;
                Ok::<f64, anyhow::Error>(result.state.best_cost)
            })
            .sum::<Result<f64>>()?;

        Ok(chi2)
    }

    fn parallelize(&self) -> bool {
        self.parallelize
    }
}
