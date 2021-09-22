# RSCA
**Measuring the chemical similarity of stars through applying metric-learning to open-clusters with experiments using the APOGEE DR16 survey**

## Summary

This is a companion repository for "Measuring chemical likeness of stars with RSCA" containing the code for reproducing experiments. Some effort has been made towards making the code easy to run and well documented but code has not been tested on external computer. Feel free to open a git issue if you run into any issues.

## Requirements

- `apogee`: We access the APOGEE survey data using the `apogee` package. Detailed instructions for installing `apogee` can be found in its associated repository. This repository was designed to work with DR16.

- Other dependencies are :`astropy`,`matplotlib`,`mpl_scatter_density`,`numpy`,`scikit-learn`,`scipy`. These must be manually installed.

Use the package manager pip to install

```bash
pip install setup.py -e .
```


### Structure

**`RSCA/apoNN/src` contains the core code for the algorithm. More precisely...**

`/apoNN/src/occam.py` contains code for cross-matching the Occam value-added catalogue with the APOGEE dataset cut. Is used for returning a filtered down Apogee Allstar Fits file containing only those OCCAM object that pass a data-cut.

`/apoNN/src/data.py` contains code for converting an Apogee Allstar Fits file into numpy arrays or masked arrays. It will also download any missing AspcapStar spectra but does so extremely slowly.


`/apoNN/src/vectors.py`contains wrappers around observations. Rather than directly manipulating numpy arrays, our algorithm wraps these through `Vector` classes. This allows for adding useful linear algebra utility functions and tools for naturally handling open-clusters and keeping track of member stars.

`/apoNN/src/fitters.py` contains the RSCA code (and a few unpublished variants), implemented in `Fitter` classes.

`/apoNN/src/evaluators.py` contains `Evaluator` objects. These are wrappers around `Fitter` that naturally handle the cross-validation, calculate the doppelganger rates, and allow for inspecting individual open-clusters.

**`RSCA/apoNN/scripts` contains Python scripts for downloading and saving the dataset** These must be run to download the dataset and reproduce the dataset cuts in the paper.

**`RSCA/apoNN/figures` contains Python scripts for reproducing figures** run these to recreate the figures. These are also a good starting point to seeing how the code is structured.

**`RSCA/outputs` contains any and all outputs. This includes datasets and generated figures**

## Citation

Please cite as:

```
TODO
```

