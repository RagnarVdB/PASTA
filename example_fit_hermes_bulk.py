# This is an example of fitting a high resolution spectrum with particle swarm optimization.
# The models are assumed to be already convolved to the resolution of the instrument.
# Example for the HERMES spectrograph.

from json import dump, load
from pathlib import Path

import numpy as np
from astropy.io import fits  # type: ignore
from pasta import (  # type: ignore
    CachedInterpolator,
    ChunkContinuumFitter,
    FixedResolutionDispersion,
    InMemInterpolator,
    NoConvolutionDispersion,
    PSOSettings,
    WlGrid,
)
from tqdm.auto import tqdm, trange  # Optional progress bar
from multiprocessing import Pool

# Get more info on errors on Rust side

wl_rng = (4007, 5673)


# Read the observed spectrum from a fits file
# Return the wavelength, flux, and variance arrays
def read_and_prepare_spectrum(
    path: Path, dtype=np.float32
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with fits.open(path) as image:
        flux = np.array(image[0].data, dtype=dtype)  # type: ignore
        # Hermes uses a logarithmic wavelength scale
        wl = np.exp(
            np.linspace(
                image[0].header["CRVAL1"],  # type: ignore
                image[0].header["CDELT1"] * (len(flux) - 1) + image[0].header["CRVAL1"],  # type: ignore
                len(flux),
            )
        )
    # Read the variance from the corresponding file
    with fits.open(str(path).replace("merged", "mergedVar")) as image:
        var = np.array(image[0].data, dtype=dtype)  # type: ignore

    # Only use 4007-5673 Angstroms
    mask = (wl >= wl_rng[0]) & (wl <= wl_rng[1])
    wl = wl[mask]
    flux = flux[mask]
    var = var[mask]
    # Ignore pixels with zero or nan variance
    mask = (np.isnan(var)) | (var <= 0) | (np.isinf(var))
    var[mask] = 1e30  # Set a large value to ignore these pixels
    return (wl, flux, var)


# Set up the interpolator object for the model grid
# InMemInterpolator can be replaced by OnDiskInterpolator or CachedInterpolator
# if the models are too large to fit in memory

# We use the pre-convolved model grid
# model_path = "/scratch/ragnarv/hermes_norm_convolved_u16"
model_path = "/STER/hermesnet/hermes_norm_convolved_u16"
# The wavelength grid of the models (first wavelength, step, number of pixels) in log10 scale
wl_grid = WlGrid(np.log10(4000), 2e-6, 76145, log=True)

# Alternatively use the unconvoled models:
# model_path = "/scratch/ragnarv/unconvolved_norm_u16"
# wl_grid = WlGrid(np.log10(3500), 1.5e-6, 316699, log=True)

interpolator = CachedInterpolator(
    str(model_path),
    False,  # We are using normalized models that don't include the max value.
    wavelength=wl_grid,
    n_shards=24,  # Applies to CachedInterpolator only
    lrucap=50_000,  # Applies to CachedInterpolator only
)

# Output is written in json format
output_file = Path("output_bulk.json")

# Settings to the PSO algorithm
settings = PSOSettings(
    num_particles=46,
    max_iters=60,
    inertia_factor=-0.3085,
    cognitive_factor=0,
    social_factor=2.0273,
)

vsini_range = (1, 500)
rv_range = (-150, 150)

# Parameters to continuum fitting
number_of_chunks = 5
polynomial_degree = 8
blending_fraction = 0.2

# In case the output file already exists, overwrite it,
# or append to it and skip the already computed solutions
overwrite = False
if not overwrite and output_file.exists():
    with open(output_file) as f:
        solutions = load(f)
else:
    solutions = []

# Path to folder with observed spectra
spectra_folder = Path("/STER/hermesnet/observed")
# For HERMES, don't include the Var files here
flux_files = sorted(
    [f for f in list(spectra_folder.glob("*.fits")) if "Var" not in f.name]
)

# Read the first spectrum to get the wavelength grid
wl, _, _ = read_and_prepare_spectrum(flux_files[0])

# For the preconvolved grid, we don't need to convolve the models anymore
dispersion = NoConvolutionDispersion(wl, wl_grid)
# Otherwise specify the spectral resolution here
# dispersion = FixedResolutionDispersion(wl, 86000, wl_grid)

# The object for fitting the continuum in chunks
continuum_fitter = ChunkContinuumFitter(
    wl, number_of_chunks, polynomial_degree, blending_fraction
)

# The fitter object for the PSO algorithm
fitter = interpolator.get_fitter(
    dispersion, continuum_fitter, settings, vsini_range, rv_range
)

overwrite = False
if not overwrite and output_file.exists():
    solutions = load(open(output_file))
else:
    solutions = []

chunksize = 46  # Over how many spectra will be parallelized

for i in trange(0, len(flux_files), chunksize):
    chunk_files = flux_files[i : i + chunksize]
    existing_indices = [sol["index"] for sol in solutions]
    filtered_files = [
        file for file in chunk_files if file.stem.split("_")[0] not in existing_indices
    ]
    with Pool(23) as p:
        spectra = p.map(read_and_prepare_spectrum, filtered_files)

    fluxes = [spec[1] for spec in spectra]
    variances = [spec[2] for spec in spectra]

    chunk_solutions = fitter.fit_bulk(
        interpolator,
        fluxes,
        variances,
    )
    labels = [sol.label.as_list() for sol in chunk_solutions]
    chunk_uncertainties = fitter.compute_uncertainty_bulk(
        interpolator,
        fluxes,
        variances,
        86_000, # Spectral resolution
        labels,
        [1 / 10, 0.8, 1.0, 0.5, 60.0],
    )
    for file, solution, uncertainty in zip(filtered_files, chunk_solutions, chunk_uncertainties):
        index = file.stem.split("_")[0]
        if index in [sol["index"] for sol in solutions]:
            continue
        # Save as json with index of observation
        solutions.append({"index": index, **solution.to_dict(), "uncertainties": uncertainty.to_dict()})
        with open(output_file, "w") as f:
            dump(solutions, f)