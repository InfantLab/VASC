# Video Actor Synchrony and Causality (VASC)
### Created by Caspar Addyman <c.addyman@gold.ac.uk>
#### Goldsmiths, University of London, 2020

_Version - Not even alpha!_

Videos of interacting humans are converted into time series data with [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose). The data are processed and then various statistical measures of synchrony and causality between actors are calculated using [scipy](https://www.scipy.org/scipylib/index.html).

This repository provides Python code and annotated Jupyter notebooks to perform these actions.

* Step 0: Getting started with this project. What to install (besides these files).
* Step 1: Process a video (or videos) with OpenPose, creating a JSON file with wireframe data for all identified persons ('actors'). 
* Step 2: Extract JSON data to numpy and perform basic validations (identifying individuals over time, tagging windows of interest, handle missing data).
* Step 3: Calculated cross-correlations, Granger Causality (and other measures) between actors in dataset. 

## Installation

To get this script working on a new system you need to do the following

1. You need a working Python 3.7 environment with support for Jupyter notebooks. The easiest way to do this is to install [Anaconda](https://www.anaconda.com/distribution/).
2. Install [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
3. Next you need to download the trained neural-network models that OpenPose uses. To do this go to the `models` subdirectory of OpenPose directory, and double-click / run the `models.bat` script.
3. Install OpenCV2 
4. Launch jupyter, open the Step 0 notebook and follow the instructions in there.

### Python dependencies

The main requirements are found in `environment.yml` which can be used to create a new [(ana)conda](https://docs.conda.io/en/latest/) environment like so:

```bash
conda create -f environment.yml
```

### Requirements
The main requirements are:

  - numpy
  - seaborn
  - pandas
  - glob2
  - opencv
  - pyarrow
  - xlrd
  - jupytext
  - ipywidgets
  - ipycanvas
  - nodejs

### Todo: 
A tutorial is providing using an example of mother infant interaction. 



#### Scientific Background
These tools were developed for a scientific project that was aims to see if parents and babies move in synchrony with each other and whether this predicts caring outcomes. The details are found here:

_Automated measurement of responsive caregiving at scale using machine learning._
Royal Academy of Engineering  / Global Challenges Research Fund
[Overview document](https://docs.google.com/document/d/1FoBBY_XxHAFbsKjmJ1Q1dIbDrpovvo3xLh1GNzJ1ziU/edit)

#### Funding:
This project was supported by the Royal Academy of Engineering Global Challenges Research Fund 
Grant:
[Frontiers of Development](https://www.raeng.org.uk/grants-and-prizes/grants/international-research-and-collaborations/frontiers/frontiers-of-development) - Tranche 2 - FoDSF\1920\2\100020



