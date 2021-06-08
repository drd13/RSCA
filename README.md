## PPPCA

### Introduction

This is the companion repository for the PPPCA paper. It is designed to contain all code and information necessary for reproducing experiments published in the paper. Special effort has been made towards making the code easy to run and well documented. If you find any issue or have a question, feel free to open a github issue.

### Requirements

- `apogee`. This module makes use of the `apogee` module, maintaimned by Jo Bovy, for interfacing with the Apogee scientific data. Make sure to point the module towards the data release DR16 if you wish to reproduce results in the paper. Because the code is modular, the code should be easily extendable to other surveys.

- `scikit-learn`

- 'ppca'

### Structure

The codebase contains a module containing the core code for reproducing experiments as found in `/apoNN/apoNN/src/`. The code for reproducing figures can be found in `/apoNN/apoNN/figures/` (with generated figures found in `/apoNN/outputs/figures`). Before generating figures, the scripts in "/apoNN/apoNN/scripts/" must be run.

### Module

For those wanting to interact with the codebase, key files within `/apoNN/apoNN/src` are as follows:

`/apoNN/apoNN/src/occam.py` contains code for cross-matching the Occam and Apogee catalogue. Is used for returning a filtered down Apogee Allstar Fits file containing only those OCCAM object that pass a data-cut.

`/apoNN/apoNN/src/data.py` contains code for converting an Apogee Allstar Fits file into numpy arrays or masked arrays. If the ASPCAPSTAR files associated to each star in ALLstar have not been downloaded, this will download missing entries but extremely slowly.

`/apoNN/apoNN/src/vectors.py`contains wrappers around observations. Rather than directly manipulating numpy arrays, our algorithm wraps these through `Vector` classes. This allows for adding useful linear algebra utility functions and tools for naturally handling open-clusters and keeping track of member stars. 

`/apoNN/apoNN/src/fitters.py` contains the code for our algorithm, implemented in `Fitter` classes. This is only a few lines of code.

`/apoNN/apoNN/src/evaluators.py` contains `Evaluator` objects. These are wrappers around `Fitter` that naturally handle the cross-validation, calculate the doppelganger rates, and allow for inspecting individual open-clusters.  
