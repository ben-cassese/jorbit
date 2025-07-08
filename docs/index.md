# Jorbit Documentation
<!-- <div align="center"> <img src="./_static/jorbit_logo_dark.svg" width="80%"> </div> -->

Welcome to the documentation for `jorbit`! This package offers a framework for high-precision, JAX-based orbit determination and propagation within the solar system. It consists of several "front-end" classes and functions that are designed to spin up simulations with minimal effort, as well as a flexible "back-end" that allows for customization and extension. Much of the code is written in JAX, which allows for automatic differentiation and GPU acceleration.

Although `jorbit` does not rely on any other dynamics packages, its under-the-hood design is based heavily on both the IAS15 integrator in REBOUND ([Rein and Spiegel 2015](https://ui.adsabs.harvard.edu/abs/2015MNRAS.446.1424R/abstract), [Rein and Liu 2012](https://ui.adsabs.harvard.edu/abs/2012A%26A...537A.128R/abstract)), and on the `gr_full` effect in REBOUNDx ([Tamayo, Rein, Shi and Hernandez, 2019](https://ui.adsabs.harvard.edu/abs/2020MNRAS.491.2885T/abstract)). We thank these authors for making their work open-source and available to the community.

Check out the tutorials/demos for an idea of what `jorbit` can do, and the user guide for installation and contributing instructions. The API documentation is also available for a more detailed look at the code. Open source contributions and issues are welcome!

## Attribution

Hopefully PSJ

```{toctree}
:maxdepth: 1
:hidden:
:caption: User Guide

user_guide/installation
user_guide/contributing
user_guide/changelog
user_guide/cache
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Tutorials/Demos

tutorials/local_mpchecker.ipynb
tutorials/lightcurve_contamination.ipynb
tutorials/interact_de_ephemeris.ipynb
tutorials/generate_particle_ephemeris.ipynb
tutorials/compare_w_horizons.ipynb
tutorials/fit_an_orbit.ipynb
tutorials/deep_dive.ipynb
tutorials/apophis_flyby.ipynb
```


```{toctree}
:maxdepth: 1
:hidden:
:caption: Common API

basic/particle
basic/observations
basic/system
basic/mpchecker
basic/ephemeris

```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Core API
:titlesonly:

core/accelerations/accelerations
core/astrometry/astrometry
core/data/data
core/ephemeris/ephemeris
core/integrators/integrators
core/mpchecker/mpchecker
core/misc/misc
```
