from json import dump
from os import environ
from pathlib import Path
from sys import argv

import numpy as np
from astropy.io import fits  # type: ignore
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from pasta import (  # type: ignore
    ChunkContinuumFitter,
    InMemInterpolator,
    NoConvolutionDispersion,
    CachedInterpolator,
    PSOSettings,
    WlGrid,
)

environ["RUST_BACKTRACE"] = "1"

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


def plot_fit(ax, wl, flux, model, wl_range=None):
    mask = (
        np.ones_like(wl, dtype=bool)
        if wl_range is None
        else (wl >= wl_range[0]) & (wl <= wl_range[1])
    )
    ax.plot(wl[mask], flux[mask], c="k", linewidth=0.8)
    ax.plot(wl[mask], model[mask], c="r", linewidth=1)
    ax.axhline(1, c="blue", linestyle="--", linewidth=1)
    ax.margins(x=0)


def is_near_bound(value, bounds):
    for bound in bounds:
        if abs(value - bound) < 1e-3:
            return True
    return False


def process_night(night: str, output_json: Path, output_pdf: Path, n_cores: int):
    environ["RAYON_NUM_THREADS"] = str(n_cores)

    input_folder = Path(f"/STER/mercator/hermes/{night}/reduced/")
    flux_files = list(
        input_folder.glob("*_HRF_OBJ_ext_CosmicsRemoved_log_merged_c.fits")
    )
    if len(flux_files) == 0:
        print(f"No flux files found in {input_folder}")
        return

    print(f"Found {len(flux_files)} spectra to process")

    wl_grid = WlGrid(np.log10(4000), 2e-6, 76145, log=True)
    model_path = "/STER/hermesnet/hermes_norm_convolved_u16"
    interpolator = CachedInterpolator(
        str(model_path),
        False,  # We are using normalized models that don't include the max value.
        wavelength=wl_grid,
        n_shards=24,  # Applies to CachedInterpolator only
        lrucap=50_000,  # Applies to CachedInterpolator only
    )

    # Settings to the PSO algorithm
    settings = PSOSettings(
        num_particles=46,
        max_iters=60,
        inertia_factor=-0.3085,
        cognitive_factor=0,
        social_factor=2.0273,
    )
    vsini_range = (0, 500)
    rv_range = (-150, 150)
    # Parameters to continuum fitting
    number_of_chunks = 5
    polynomial_degree = 8
    blending_fraction = 0.2

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

    solutions = []
    with PdfPages(str(output_pdf)) as pdf:
        for i, file in enumerate(flux_files):
            print(f"{i}/{len(flux_files)}: {file}", end="\r")
            index = file.stem.split("_")[0]

            wl, flux, var = read_and_prepare_spectrum(file)
            # Do the particle swarm optimization fit
            solution = fitter.fit(
                interpolator,
                flux,
                var,
            )
            # The chi2 landscape will be sampled in this region to determine the uncertainty.
            # Teff and vsini are relative to the solution value, others absolute.
            search_radius = [1 / 10, 0.8, 1.0, 0.5, 60.0]
            uncertainty = fitter.compute_uncertainty(
                interpolator, flux, var, 86_000, solution.label.as_list(), search_radius
            )
            solutions.append(
                {
                    "index": index,
                    **solution.to_dict(),
                    "uncertainties": uncertainty.to_dict(),
                }
            )
            with open(output_json, "w") as f:
                dump(solutions, f)

            # Plots
            labels = solution.label.as_list()
            normalized_model = interpolator.produce_model(dispersion, *labels)
            continuum = continuum_fitter.build_continuum(solution.continuum_params)
            residuals = flux / continuum - normalized_model

            fig = plt.figure(layout="constrained", figsize=(9, 6), dpi=100)
            gs = GridSpec(3, 3, figure=fig)
            ax1 = fig.add_subplot(gs[:2, :])
            ax2 = fig.add_subplot(gs[2, 0])
            ax3 = fig.add_subplot(gs[2, 1])
            ax4 = fig.add_subplot(gs[2, 2])

            # Main plot
            axt = ax1.twinx()
            axt.plot(wl, residuals, c="k", linewidth=0.8)
            axt.axhline(0, c="blue", linestyle="--", linewidth=1)
            axt.set_ylabel("Residuals")
            ax1.set_ylim(
                np.quantile(flux / continuum, 0.001) - 0.05,
                np.quantile(flux / continuum, 0.999) + 0.05,
            )
            mn = np.quantile(residuals, 0.01)
            mx = np.quantile(residuals, 0.99)
            low = (6 * mn - 3 * mx) / 3
            high = 3 * mx - 2 * low
            axt.set_ylim(low, high)
            plot_fit(ax1, wl, flux / continuum, normalized_model)

            # Zoomed in plots
            rv_shift = 1 + labels[4] / 299_792.458
            HeI = 4471.479 * rv_shift
            range3 = (HeI - 20, HeI + 20)
            plot_fit(ax2, wl, flux / continuum, normalized_model, wl_range=range3)
            Hbeta = 4861.4 * rv_shift
            range1 = (Hbeta - 20, Hbeta + 20)
            plot_fit(ax3, wl, flux / continuum, normalized_model, wl_range=range1)
            Mgtriplet_left = 5159 * rv_shift
            Mgtriplet_right = 5195 * rv_shift
            range2 = (Mgtriplet_left, Mgtriplet_right)
            plot_fit(ax4, wl, flux / continuum, normalized_model, wl_range=range2)

            ax1.set_ylabel("Normalized flux")
            ax2.set_ylabel("Normalized flux")
            ax3.set_xlabel("Wavelength [A]")
            teff_bounds = (6.000, 10.000, 25.000, 30.000)
            m_bounds = (-0.8, 0.8)
            logg_bounds = (2.5, 3.0, 3.3, 5.0)
            vsini_bounds = (0, 5.00)
            rv_bounds = (-1.50, 1.50)
            # Apparently this is the only way to have different text colors
            text = ax1.text(
                0,
                1.02,
                f"{index}\n",
                transform=ax1.transAxes,
                fontsize="12",
            )
            text = ax1.annotate(
                f"$T_{{eff}}={round(labels[0])}$ K, ",
                color="r" if is_near_bound(labels[0] / 1000, teff_bounds) else "k",
                xycoords=text,
                xy=(0, 0),
                verticalalignment="bottom",
                fontsize="12",
            )
            text = ax1.annotate(
                f"[M/H]$={labels[1]:.2f}$ dex, ",
                color="r" if is_near_bound(labels[1], m_bounds) else "k",
                xycoords=text,
                xy=(1, 0),
                verticalalignment="bottom",
                fontsize="12",
            )
            text = ax1.annotate(
                f"$\\log g={labels[2]:.2f}$ dex, ",
                color="r" if is_near_bound(labels[2], logg_bounds) else "k",
                xycoords=text,
                xy=(1, 0),
                verticalalignment="bottom",
                fontsize="12",
            )
            text = ax1.annotate(
                f"$v \\sin i={labels[3]:.1f}$ km/s, ",
                color="r" if is_near_bound(labels[3] / 100, vsini_bounds) else "k",
                xycoords=text,
                xy=(1, 0),
                verticalalignment="bottom",
                fontsize="12",
            )
            text = ax1.annotate(
                f"$RV={labels[4]:.1f}$ km/s",
                color="r" if is_near_bound(labels[4] / 100, rv_bounds) else "k",
                xycoords=text,
                xy=(1, 0),
                verticalalignment="bottom",
                fontsize="12",
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    n_cores = 8
    arguments = argv[1:]
    if len(arguments) != 1:
        raise ValueError("One argument expected: night to be processed")
    night = arguments[0].replace("/", "").strip()
    output_json = Path(f"{night}_labels.json")
    output_pdf = Path(f"{night}_fits.pdf")
    process_night(night, output_json, output_pdf, n_cores)
