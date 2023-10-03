# Pivimage

**Note, that the project is still under development!**

`PIVimage` is a small library assisting with reading PIV images. It provides specific utilities to handle PIV images and
to perform certain operations on them, which are useful in the pre-processing stage of PIV analysis.

Available operations are ...

- rotating images
- interfacing with double frame images in one or two files
- plotting

More features will be added in the future.


## Installation

```bash
pip install pivimage
```

## Examples

<a target="_blank" href="https://colab.research.google.com/github/matthiasprobst/pivimage/blob/main/examples/Getting%20started.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Tests
Tests can be found in the folder `tests/`. They can be run with `pytest`:

```bash
pytest
```

Get a coverage report with

```bash
pytest --cov --cov-report html
```