# RSCA
**Measuring the chemical similarity of stars through applying metric-learning to open-clusters with experiments using the APOGEE DR16 stellar survey**

## Summary

This is a companion repository for the paper "Measuring chemical likeness of stars with RSCA". It contains the code necessary for reproducing experiments. Some limited effort has been made towards ensuring the code is well-documented and easily instalable. However the code has not been tested on external machines. Users are welcomed to encouraged to open git issue if they run into any problems.

## Requirements

- `apogee`: We access the APOGEE survey data using the `apogee` package. Detailed instructions for installing `apogee` can be found in its [associated repository](https://github.com/jobovy/apogee). Our repository was designed to work with DR16 and so the associated environment variable within `apogee` should be appropriately set.

- Other dependencies are :`astropy`,`matplotlib`,`mpl_scatter_density`,`numpy`,`scikit-learn`,`scipy`. These must be manually installed.

Use the package manager pip to install

```bash
pip install setup.py -e .
```


### Structure

**`RSCA/apoNN/src` contains the core code for the algorithm. More precisely...**

`/apoNN/src/occam.py` contains code for cross-matching the Occam value-added catalogue with an APOGEE dataset cut. Is will return a filtered down Apogee AllStar Fits file containing only those OCCAM object within an APOGEE style catalogue.

`/apoNN/src/data.py` contains code for downloading and pre-processing the AspcapStar spectra in an APOGEE dataset. While, it can download spectra in AllStar not locally available, it does so extremelly inefficiently.

`/apoNN/src/vectors.py`contains data wrappers. Our codebase, rather than directly manipulating numpy arrays, wraps these into a `Vector` class allowing easier handling of open-clusters. This allows for keeping track of cluster member stars and addition of useful utility functions.

`/apoNN/src/fitters.py` contains the source code for the RSCA algorithm(as well as a few unpublished variants). These take the form of `Fitter` classes.

`/apoNN/src/evaluators.py` contains `Evaluator` objects. These are wrappers around `Fitter` that naturally handle cross-validation, doppelganger rate calculations and visualizations of RSCA runs.

**`RSCA/apoNN/scripts` contains Python scripts for downloading and saving the dataset with the same dataset cuts as used in the paper.** These must be run preliminarily to the code for generating figures.

**`RSCA/apoNN/figures` contains Python scripts for reproducing figures.** These are a good starting point for understanding how to run the codebase.

**`RSCA/outputs` contains any and all outputs. This includes intermediary pickled datasets as well as generated figures.**

## Citation

Please cite as:

```
TODO
```

