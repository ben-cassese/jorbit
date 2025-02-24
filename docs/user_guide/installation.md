# Installation

``jorbit`` is hosted on pypi and can be installed using ``pip``. If you If you run into issues, try installing in a fresh environment to avoid conflicts with other packages. If issues persist, please open an issue on the GitHub repository.

Separately, if you're more of a .venv person and are open to using [uv](https://docs.astral.sh/uv/), you can recreate the exact development environment by cloning the `uv.lock` file and running `uv sync`. This will create virtual environment in the current directory that can be activated with `source .venv/bin/activate` (or the equivalent on your system).

```{warning}
The first time you import ``jorbit``, it will automatically download and cache several necessary files, including the JPL DE 440 ephemeris that are used to factor in planetary perturbations. This is a one-time download of ~a GB and can take a few minutes depending on your internet connection.

When using the `mpchecker` functions, other files will similarly be automatically downloaded and cached. A warning will be issued each time a new file is downloaded, but if running on a shared system or if you have disk space concerns, be sure to keep track of your cache size.

See the [cache management](cache.md) page for more information.
```

## CPU Users

This is the most straightforward situation to be in when installing. All of the usual methods should work fine:

<span style="font-size:larger;">Option 1: pip install:</span>

```bash
python -m pip install -U jorbit
```

<span style="font-size:larger;">Option 2: install from source:</span>

```bash
python -m pip install git+https://github.com/ben-cassese/jorbit
```

<span style="font-size:larger;">Option 3a: clone and install an editable version:</span>

```bash
git clone https://github.com/ben-cassese/jorbit
cd jorbit
python -m pip install -e .
```

<span style="font-size:larger;">Option 3b: clone and install via uv:</span>

```bash
git clone https://github.com/ben-cassese/jorbit
python -m pip install uv # if you don't already have it
cd jorbit
uv sync
```

## GPU Users

Since ``jorbit`` relies heavily on ``JAX``, large portions can technically run on a GPU (or a TPU) as well as a CPU with no changes to the code. However, anyone attempting to do this should not expect automatic speedups. ``jorbit`` is not optimized for GPU use, since many of the operations are run sequentially and it was entirely developed on a CPU. There are some areas where a GPU could be beneficial (e.g. >1e6 massless particles interacting with a smaller number of massive particles), but in general try to manage expectations.

If you are interested in running ``jorbit`` on a GPU, be sure you first follow the instructions for installing ``jax`` and ``jaxlib`` on your specific system, then install ``jorbit`` as normal. If you run into any issues, or even better if you're interested in helping to optimize ``jorbit`` for GPU use, please open an issue on the GitHub repository.
