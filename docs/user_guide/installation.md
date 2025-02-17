# Installation

``jorbit`` is a pypi-hosted Python package that can be installed using ``pip``. If you run into issues, try installing in a fresh environment to avoid conflicts with other packages. If issues persist, please open an issue on the GitHub repository.

## Mac/Linux Users

This is the most straightforward situation to be in when installing. All of the usual methods should work fine:

<span style="font-size:larger;">Option 1: pip install:</span>

```bash
python -m pip install jorbit
```

<span style="font-size:larger;">Option 2: install from source:</span>

```bash
python -m pip install git+https://github.com/ben-cassese/jorbit
```

<span style="font-size:larger;">Option 3: clone and install an editable version:</span>

```bash
git clone https://github.com/ben-cassese/jorbit
cd squishyplanet
python -m pip install -e .
```

## Windows Users

Things may be slightly more complicated on Windows but still manageable. One just needs to be careful with their installation of ``jax`` and ``jaxlib``--- before installing ``squishyplanet``, check the [JAX installation instructions](https://jax.readthedocs.io/en/latest/installation.html#install-cpu) for Windows and make sure you have the correct version of ``jax`` and ``jaxlib`` installed in your environment. After that, any of the options above should work.

## GPU Users

Since ``jorbit`` relies heavily on ``JAX``, large portions can technically run on a GPU (or a TPU) as well as a CPU with no changes to the code. However, anyone attempting to do this will likely be disappointed with the performance, since in its current state  ``jorbit`` is not optimized for GPU use. Many of the operations are run sequentially and it was entirely developed on a CPU.

If you are interested in running ``jorbit`` on a GPU, be sure you first follow the instructions for installing ``jax`` and ``jaxlib`` on your specific system, then install ``jorbit`` as normal. If you run into any issues, or even better if you're interested in helping to optimize ``jorbit`` for GPU use, please open an issue on the GitHub repository.
