#![allow(unused)]
use anyhow::Result;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra as na;
use npy::NpyData;
use pasta::continuum_fitting::{ChunkFitter, ContinuumFitter};
use pasta::convolve_rv::{
    convolve_rotation, shift_and_resample, NoConvolutionDispersionTarget, WavelengthDispersion,
};
use pasta::fitting::ObservedSpectrum;
use pasta::interpolate::{GridInterpolator, Interpolator, WlGrid};
use pasta::model_fetchers::InMemFetcher;
use rayon::prelude::*;
use std::io::Read;
use std::path::PathBuf;

const SMALL_GRID_PATH: &str = "/STER/hermesnet/hermes_norm_convolved_u16_small.tar.zst";

pub fn read_npy_file(file_path: PathBuf) -> Result<na::DVector<f64>> {
    let mut file = std::fs::File::open(file_path.clone())?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    let data: NpyData<f64> = NpyData::from_bytes(&buf)?;
    Ok(na::DVector::from_iterator(data.len(), data))
}

pub fn benchmark(c: &mut Criterion) {
    let wl_grid = WlGrid::Logspace(3.6020599913, 2e-6, 76_145);
    let interpolator = GridInterpolator::new(
        InMemFetcher::from_tar_zstd(SMALL_GRID_PATH.into(), false).unwrap(),
        wl_grid.clone(),
    );
    let wl = read_npy_file("wl_hermes.npy".into()).unwrap();
    let dispersion = NoConvolutionDispersionTarget::new(wl.clone(), &wl_grid);
    let interpolated = interpolator.interpolate(8000.0, 0.0, 3.5).unwrap();
    let convolved_for_rotation = convolve_rotation(&wl_grid, &interpolated, 100.0).unwrap();
    let model = dispersion
        .convolve_segment(convolved_for_rotation.clone())
        .unwrap();
    let model_arr = interpolator
        .produce_model(&dispersion, 8000.0, 0.0, 3.5, 100.0, 1.0)
        .unwrap();
    let output = shift_and_resample(&wl_grid, &model, &dispersion, 1.0).unwrap();
    let observed_spectrum = ObservedSpectrum {
        flux: output.clone(),
        var: output.map(|x| x.sqrt()),
    };
    let continuum_fitter = ChunkFitter::new(wl.clone(), 10, 5, 0.2);

    c.bench_function("chi2", |b| {
        b.iter(|| {
            let model = interpolator
                .produce_model(&dispersion, 8000.0, 0.0, 3.5, 100.0, 1.0)
                .unwrap();
            continuum_fitter.fit_continuum(&observed_spectrum, &model)
        })
    });

    c.bench_function("produce_model", |b| {
        b.iter(|| {
            interpolator
                .produce_model(&dispersion, 8000.0, 0.0, 3.5, 100.0, 1.0)
                .unwrap()
        })
    });
    c.bench_function("interpolate", |b| {
        b.iter(|| interpolator.interpolate(8000.0, 0.0, 3.5).unwrap())
    });
    c.bench_function("convolve_rotation", |b| {
        b.iter(|| convolve_rotation(&wl_grid, &interpolated, 100.0).unwrap())
    });
    c.bench_function("convolve resolution", |b| {
        b.iter(|| {
            dispersion
                .convolve_segment(convolved_for_rotation.clone())
                .unwrap()
        })
    });
    c.bench_function("resample", |b| {
        b.iter(|| shift_and_resample(&wl_grid, &model, &dispersion, 1.0).unwrap())
    });
    c.bench_function("fit_continuum", |b| {
        b.iter(|| continuum_fitter.fit_continuum(&observed_spectrum, &model_arr))
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
