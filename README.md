[![Documentation Status](https://readthedocs.org/projects/jorbit/badge/?version=latest)](https://jorbit.readthedocs.io/en/latest/?badge=latest)
![Build Status](https://github.com/ben-cassese/jorbit/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/github/ben-cassese/jorbit/graph/badge.svg?token=7AUZRB9MFO)](https://codecov.io/github/ben-cassese/jorbit)
![PyPI - Version](https://img.shields.io/pypi/v/jorbit)
![pypi-platforms](https://img.shields.io/pypi/pyversions/jorbit)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
![ruff-badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)

<p align="center">
  <img src="./docs/_static/jorbit_logo_github.svg" alt="jorbit logo">
</p>

jorbit is a python/JAX package for simulating and fitting orbits of objects within the solar system. Built primarily in JAX, jorbit can compute exact derivatives even for models that involve complex numerical integrators, like [IAS15](https://ui.adsabs.harvard.edu/abs/2015MNRAS.446.1424R/abstract), and acceleration functions, like [Parameterized Post-Newtonian gravitation](https://ui.adsabs.harvard.edu/abs/2020MNRAS.491.2885T/abstract). It has several high-level convenience wrappers for dealing with standard massless minor planets, but also can enable more complex simulations through user-provided acceleration functions.

For more information, check out the [documentation](https://jorbit.readthedocs.io/en/latest/index.html)!

## Installation

Exact installation instructions may depend on your hardware/environment, so see the [installation page](https://jorbit.readthedocs.io/en/latest/user_guide/installation.html) for more information. In general, you can install jorbit with pip:

Be aware! The first time you import jorbit, it will automatically download and cache (via the astropy caching mechanisms) about ~1 GB of files required to run its simulations, including the [JPL DE400 ephemeris](https://ui.adsabs.harvard.edu/abs/2021AJ....161..105P/abstract) files. Other jorbit functions may also need to download/cache additional files, so be sure to have a good internet connection and enough disk space available.

```bash
python -m pip install -U jorbit
```

jorbit was built with `uv`, so if you want to replicate the exact development environment, use the `uv.lock` file. Check out the [uv docs](https://docs.astral.sh/uv/) for more information.


## Example Usage

Many more examples can be found in the docs, but here are two simple examples to get you started:

To create an ephemeris for a minor planet:
```python
from astropy.time import Time
from jorbit import Particle

p = Particle.from_horizons(name="274301", time=Time("2025-01-01"))
ephem = p.ephemeris(
    times=Time(["2025-01-01", "2025-01-02", "2025-01-03"]),
    observer="Kitt Peak"
)
```

To identify all known minor planets nearby a certain position at a certain time:
```python
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from jorbit.mpchecker import mpchecker

mpchecker(
    coordinate=SkyCoord(ra=0 * u.deg, dec=0 * u.deg),
    time=Time("2025-01-01"),
    radius=10 * u.arcmin,
    extra_precision=True,
    observer="Palomar",
)
```

## Contributing
If you have any trouble with the code, feel free to open an issue! We welcome open-source contributions, so if you have a feature request or bug fix, please open a pull request. For more information, see the [contributing guide](https://jorbit.readthedocs.io/en/latest/user_guide/contributing.html).

## Attribution
jorbit is made freely available under the MIT License. If you use this code in your research, please cite the accompanying paper:

(coming soon)
