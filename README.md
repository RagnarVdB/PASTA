# PASTA (Pipeline for Automated Spectral Analysis)
 
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [The grid](#the-grid)
4. [Python API docs](#python-api-docs)
5. [Rust code overview](#rust-code-overview)

### [Rust docs here](https://simonvanschuylenbergh.github.io/PASTA/pasta/struct.OnDiskInterpolator.html)

## Introduction
PASTA is a fully automatic pipeline for estimating stellar labels from unnormalized spectra of single stars by comparison to model spectra.
In its current form, five labels are derived:
- Effective temperature $T_\text{eff}$
- Metallicity [M/H]
- Surface gravity $\log g$
- Rotational velocity $v \sin i$
- Radial velocity RV

The code does not have the capability of generating model spectra, but relies on a pre-computed grid of models from which spectra are produced through interpolation.
Hence the underlying physics of the pipeline is determined by the model grid that the user chooses to use.

The basic principle behind the pipeline is to compute the $\chi^2$ function between the observed spectrum and a set of model spectra, and find the set of labels that minimizes this function.
An important feature is that the spectra do not need to be normalized beforehand.
The pipeline handles normalization by fitting a function to represent the pseudo continuum for every model spectrum that is evaluated.
If $F_\text{obs}$ is the observed flux and $F_\text{model}$ that of the model for which the $\chi^2$ function is evaluated, the function representing the pseudo continuum is fitted to $\frac{F_\text{obs}}{F_\text{model}}$.
It is important that this function is flexible enough to fit the true pseudo continuum well, but not so flexible that it starts fitting the defects of the model spectrum, hence obscuring the $\chi^2$ landscape. 
The required ingredients for this method are a fast way to produce a model spectrum at any set of labels, a parametrized function to fit the pseudo continuum, and a minimization algorithm.

To produce a model spectrum, the code performs the following steps:
1. 3D cubic interpolation in a grid of models with different $T_\text{eff}$, [M/H] and $\log g$ to produce a model at arbitrary values of those three labels.
2. Convolve with a rotational broadening kernel depending on the desired value of $v \sin i$
3. Convolve with a gaussian kernel to match the spectral resolution of the instrument. This resolution may be constant or wavelength dependent. Alternatively the model grid can ve pre-convolved to improve speed.
4. Shift the spectrum along wavelength to the desired RV value.
5. Resample to the wavelength sampling of the observed spectrum


For the pseudo continuum function, we developed a method that splits the continuum into overlapping chunks, fits each with a polynomial, and merges them together to one smooth function.
The code allows the user to tweak the number of chunks, degree of the polynomials and width of the overlap regions.
Alternatively, it is possible to use a linear model by providing a design matrix.
We also provide a class to use linear models, and the normalization can also be turned off for use on pre-normalized spectra.
Additionally, due to the code's modularity it is relatively easy to add a custom normalization function.

The code uses particle swarm optimization (PSO) as the optimization algorithm. 
Our implementation is based on the one in [argmin-rs](https://github.com/argmin-rs/argmin).
The metaparameters of the algorithm (number of particles, social attraction factor, etc) can be chosen by the user.

See example usage in `example_fit_hermes.py` and `check_fit_hermes.py`.

## Installation
The code is written in Rust, and provides an interface to Python. It can be installed as a Python package, either from the pre-built wheel or by compiling the source code.


### Pre-built wheel
A pre-built wheel for x86_64 Linux is provided under the Github [releases](https://github.com/SimonVanSchuylenbergh/PASTA/releases).
It can be installed in the active Python environment with `pip`:
```sh
pip install pasta-0.1.0-cp37-abi3-manylinux_2_35_x86_64.whl
```

### Compiling from source
The Python package is built from source using [maturin](https://github.com/PyO3/maturin). To install in the active Python environment use
```sh
maturin develop --release
```

## The grid
The code expects a 3D grid of model spectra for varying $T_\text{eff}$, [M/H] and $\log g$.
The model spectra may be normalized, or include continuum flux.
In the former case, the model that is fitted in the normalization step represents the pseudo continuum (i.e. the product of the continuum of the star and the instrument response function), and in the latter case it represents only the instrument response function (and other contributions that are not modelled in the continuum of the models).

The files must be in Numpy's binary `.npy` format in uint16 and contain only one column with the flux values.
We chose to work with 16-bit integers to improve performance while still maintaining enough precision.
Therefore, the flux values in the models need to be converted to integers between 0 and $2^{16}-1$ before saving them in the npy format.
For normalized models ($0 \le F\le1$) this is done simply by multiplying by $2^{16}-1$ and using the `np.uint16` function.
For non-normalized models, things are more difficult, because the magnitude of the flux values can vary greatly between models.
Hence dividing every model by the same value would lead to insufficient precision.
Our solution is to divide every model by its own maximum, and storing that maximum value as an aside in the npy file.
The Rust code will then multiply the spectrum by this value again to retrieve the original spectrum.
The maximum value is stored as a 32-bit float in the first four bytes of the npy file.
When using these files, the flag `includes_factor=True` needs to be set in the interpolator object to signal that the first four bytes need to be interpreted as a factor to the rest of the spectrum.
Using `numpy`, the following function can be used to convert a model spectrum and store it in npy format including the factor:
```python
def store_model(flux: np.ndarray, filename: str):
    max_value = np.max(flux)
    converted_flux = np.array(flux / max_value * (2**16-1), dtype=np.uint16)
    max_byte_repr = np.frombuffer(np.float32(max_value).tobytes(), dtype=np.uint16)
    output_arr = np.concatenate([max_byte_repr, converted_flux])
    np.save(filename, output_arr)
```

It is recommended that the pixels are equally spaced in log-wavelength.
Part of the code also works with linearly spaced models, but not all of it.


The files must be named as
```
l{m/p}{[M/H]*100:04d}_{Teff:05d}_{logg*100:04d}.npy
```
e.g. for $T_\text{eff}=7100$ K, [M/H] $=-0.3$ dex, $\log g = 3.9$ dex this is
```
lm0030_07100_0390.npy
```

The grid does not need to be rectangular, but may have a 'staircase' shape in $T_\text{eff}$ vs $\log g$.
That is, not all $\log g$ values need to be available for every available $T_\text{eff}$.
The code will automatically detect the available range in $\log g$ for every different $T_\text{eff}$ in the grid.
The full range in [M/H] must exist for every $T_\text{eff}$, $\log g$ combination however.

## Python API overview
All the functionality of the code can be accessed from Python through a few exposed classes and functions. API level documentation can be found in the [generated Rust docs](https://simonvanschuylenbergh.github.io/PASTA). Below is an overview of the classes.

### [Interpolator](https://simonvanschuylenbergh.github.io/PASTA/pasta/struct.OnDiskInterpolator.html)
This is the object that handles retrieving the grid spectra and producing new model spectra.
There are three different `Interpolator` classes available that differ in the way they load the files from the grid.
`OnDiskInterpolator` loads the files straight from the disk every time they are needed.
When using multiple cores however, the disk speed becomes a major bottleneck.
`InMemInterpolator` loads all models of the grid into memory once, and keeps them in memory for the duration of the program run.
This is the recommended interpolator class when fitting many spectra, and when enough memory is available.
The third interpolator class is `CachedInterpolator`.
This one reads models from disk when they are needed, and keeps them in memory for later use.
This has the advantage of not needing to fit the full grid in memory, while still being faster than reading from disk every time.
Its downside is that it makes multithreading less efficient, because the code needs to avoid having two threads write to the same memory address at the same time.
This type is recommended when running on a smaller computer with few cores and little memory.
For many core systems, `OnDiskInterpolator` may be faster depending on the speed of the disk/SSD.

Arguments are

```python
CachedInterpolator(
    dir: str,
    includes_factor: bool
    wavelength: WlGrid,
    n_shards=4: int
    lrucap=4000: int,
)
```
The first argument is the path to the directory in which the models are stored.
The second argument specifies whether the model files include a factor that they need to be multiplied with (see above).
The third argument specifies the wavelength grid of the models (see [WlGrid](#wlgrid)).
Since the model files only contain flux values, the wavelength information needs to be provided through this argument.
The last two arguments are exclusive to `CachedInterpolator` and specify the number of shards of the cache, and the maximum number of models that are kept in memory at a time.

A full list of available methods on this object can be found [here](https://simonvanschuylenbergh.github.io/PASTA/pasta/struct.OnDiskInterpolator.html).
Below an overview of the most important ones:

#### get_fitter

```python
get_fitter(
    self,
    dispersion: PyWavelengthDispersion,
    continuum_fitter: PyContinuumFitter,
    vsini_range: (float, float),
    rv_range: (float, float),
) -> PySingleFitter
```
This function creates a `fitter` object that handles fitting observed spectra of single stars.

The first argument specifies the wavelength dispersion of the instrument that the spectrum was taken with.
It provides the information of the wavelength corresponding to every pixel, as well as the spectral resolution of the instrument which the synthetic spectra will be convolved to.
Three classes are provided: [FixedResolutionDispersion](#fixedresolutiondispersion) can be used when the spectral resolution of the instrument is constant across the wavelength range.
In this case the synthetic spectrum will be convolved with a single gaussian kernel.
[VariableResolutionDispersion](#variableresolutiondispersion) is used when the spectral resolution varies throughout the wavelength range, as is often the case with low resolution instruments. In this case every pixel will be convolved with its own gaussian kernel.
This does come at a computational cost.
[NoConvolutionDispersion](#noconvolutiondispersion) skips the convolution step, and is to be used in case the grid of models has already been convolved to the appropriate resolution.

The second argument specifies the function that will be fitted against the pseudo continuum before $\chi^2$ is evaluated.
Currently three classes for this are provided. [ChunkContinuumFitter](#chunkcontinuumfitter) divides the spectrum in a specified number of chunks, fits every chunk by a polynomial of specified degree, and blends them together. This method is generally recommended. [LinearModelContinuumFitter](#linearmodelcontinuumfitter) allows using any linear model by supplying a design matrix. [FixedContinuum](#fixedcontinuum) and [ConstantContinuum](#constantcontinuum) turn off continuum fitting and use a user specified pseudo continuum or an array of ones respectively.


#### produce_model
```python
produce_model(
    self,
    dispersion: PyWavelengthDispersion,
    teff: float,
    m: float,
    logg: float,
    vsini: float,
    rv: float,
) -> np.ndarray[np.float32]
```

#### fit_continuum_and_return_model
```python
fit_continuum_and_return_model(
    self,
    fitter: PyContinuumFitter,
    dispersion: PyWavelengthDispersion,
    observed_flux: list[float],
    observed_var: list[float],
    label: list[float; 5],
) -> list[float]
```


### [PySingleFitter](https://simonvanschuylenbergh.github.io/PASTA/pasta/struct.OnDiskSingleFitter.html)

This class handles fitting observed spectra of single stars, as well as computing uncertainties and sampling the $\chi^2$ landscape. It can be constructed by calling [`get_fitter`](#get_fitter) on an `Interpolator` object.
Its methods are:

#### fit_single_stage
```python
fit_single_stage(
    self,
    interpolator: Interpolator,
    observed_flux: np.ndarray[np.float32],
    observed_var: np.ndarray[np.float32],
    settings: PSOSettings,
    iterations=100: int,
    trace_directory=None: str | None,
    best_particle_file=None: str | None
    parallelize=True: bool,
    constraints=[]: List[Constraint] | []
) -> OptimizationResult
```

This is the method for fitting an observed spectrum through particle swarm optimization.
The first argument requires a reference to an interpolator object to handle the model generation.
The second and third arguments demand the flux and variance values of the observed spectrum. The lengths of these arrays must match with the wavelength array that was provided through the dispersion argument when constructing the fitter object.
The fourth and fifth arguments are optional and are used to save the particle positions throughout the optimization run. That data will be stored in json files in the specified directory.
The sixth argument specifies whether to speed up computations by multithreading over particles, i.e. let every particle be processed by its own thread.
The last argument can be used to constrain the search domain in one or more parameter to a specified range or a fixed value.

The function returns a [PyOptimizationResult](#optimizationresult) object that includes all the solved labels, continuum function parameters and other info. This data can be easily obtained using the `to_dict()` or `to_json()` method.

#### fit
```python
fit(
    self,
    interpolator: Interpolator,
    observed_flux: np.ndarray[np.float32],
    observed_var: np.ndarray[np.float32],
    settings: PSOSettings,
    iterations=[30, 60, 10]: List[int],
    first_stage_res=10000: float,
    trace_directory=None: str | None,
    best_particle_file=None: str | None
    parallelize=True: bool,
    constraints=[]: List[Constraint] | []
) -> OptimizationResult
```
Same as `fit_single_stage`, but using a three stage fitting algorithm. During the first stage both observed and model spectra are convolved to lower resolution (set by first_stage_res) to make finding the correct RV easier. During the second (and first) stage, linear interpolation is used to produce the model spectra, while in the third stage a more accurate but slower cubic interpolation is used.


#### fit_bulk
```python
fit(
    self,
    interpolator: Interpolator,
    observed_fluxes: List[np.ndarray[np.float32]],
    observed_vars: List[np.ndarray[np.float32]],
    settings: PSOSettings,
    iterations=[30, 60, 10]: List[int],
    first_stage_res=10000: float,
    constraints=[]: List[Constraint] | []
    best_particle_file=None: str | None
    progress=False: bool,
    allow_none=False: bool
) -> List[OptimizationResult]
```

Fit many spectra, parallelizing over spectra instead of particles. `If allow_none=True`, the function will not throw an exception when one of the optimizations fails, but instead return None for that spectrum. 

#### compute_uncertainty
```python
compute_uncertainty(
    self,
    interpolator: Interpolator,
    observed_flux: list[float],
    observed_var: list[float],
    spec_res: float,
    label: list[float; 5],
    search_radius=[2000.0, 0.3, 0.3, 40.0, 40.0]: list[float; 5],
) -> list[(float | None, float | None); 5]
```

Compute uncertainty margins for all parameters, after an optimal solution has been found (passed through `label`). `spec_res` should be set to the spectral resolution of the instrument used.

#### chi2
```python
chi2(
    self,
    interpolator: Interpolator,
    observed_flux: list[float],
    observed_var: list[float],
    labels: list[float; 5],
    allow_nan=False: bool,
    progress=False: bool
) -> list[float]
```


### [WlGrid](https://simonvanschuylenbergh.github.io/PASTA/pasta/struct.WlGrid.html)
This class specifies the wavelength grid of the model spectra
```python
WlGrid(min: float, step: float, len: int, log=False: bool)
```
For linearly spaced models (`log=False`, not fully supported), the arguments are the wavelength of the first pixel (Å), the step size between two pixels (Å), and the total number of pixels.
For log-spaced models (`log=True`), these are the $\log_{10}$ of the first wavelength, the step in $\log_{10}$ wavelength and the total number of pixels.

### [FixedResolutionDispersion](https://simonvanschuylenbergh.github.io/PASTA/pasta/fn.FixedResolutionDispersion.html)
This dispersion class is used for instruments that have a constant resolution across the wavelenth range.
```python
FixedResolutionDispersion(
    wl: list[float],
    resolution[float],
    synth_wl: WlGrid,
)
```
The first argument specifies the wavelength array of the observed spectra.
The second argument is the desired resolution ($\lambda / \Delta \lambda$) that the models wil be convolved to.
The last argument specifies the wavelength grid of the models that the class will be used on.


### [VariableResolutionDispersion](https://simonvanschuylenbergh.github.io/PASTA/pasta/fn.VariableResolutionDispersion.html)
```python
FixedResolutionDispersion(
    wl: list[float],
    disp: List[float],
    synth_wl: WlGrid,
)
```
disp: per pixel spectral resolution in Angstrom (FWHM)

### [NoConvolutionDispersion](https://simonvanschuylenbergh.github.io/PASTA/pasta/fn.NoConvolutionDispersion.html)
This dispersion class is used when the models have already been convolved to the instrument resolution.
It skips the convolution step during model generation.
Only the wavelength array of the observed spectra is required.
```
NoConvolutionDispersion(wl: list[float], synth_wl: WlGrid)
```

### [ChunkContinuumFitter](https://simonvanschuylenbergh.github.io/PASTA/pasta/fn.ChunkContinuumFitter.html)
```python
ChunkContinuumFitter(
    wl: list[float],
    n_chunks=5: int
    p_order=8: int,
    overlap=0.2: float,
)
```

### [LinearModelContinuumFitter](https://simonvanschuylenbergh.github.io/PASTA/pasta/fn.LinearModelContinuumFitter.html)

### [FixedContinuum](https://simonvanschuylenbergh.github.io/PASTA/pasta/fn.FixedContinuum.html)
```python
ChunkContinuumFitter(
    continuum: List[float],
    ignore_first_and_last=0: int
)
```

### [ConstantContinuum](https://simonvanschuylenbergh.github.io/PASTA/pasta/fn.ConstantContinuum.html)

### [PSOSettings](https://simonvanschuylenbergh.github.io/PASTA/pasta/struct.PSOSettings.html)

```python
PSOSetttings(
    num_particles: int,
    max_iters: int,
    inertia_factor=0.7213475204: float,
    cognitive_factor=1.1931471806: float,
    social_factor=1.1931471806: float,
    delta=1e-7: float,
)
```

### OptimizationResult
```python
label: tuple[float, float, float, float, float]
continuum_params: list[float]
chi2: float,
iterations: int,
time: float
```

#### to_dict
```python
to_dict(self) -> dict
```

## Rust code overview
