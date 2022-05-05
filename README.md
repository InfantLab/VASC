# Video Actor Synchrony and Causality Toolkit (VASC)
### Created by Caspar Addyman <c.addyman@gold.ac.uk>
#### Goldsmiths, University of London, 2022

_Version - 0.2_

Videos of interacting humans are converted into time series data with [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose). The data are processed and then various statistical measures of synchrony and causality between actors are calculated using [scipy](https://www.scipy.org/scipylib/index.html).

This repository provides Python code and annotated Jupyter notebooks to perform these actions.

* Step 0: Getting started with this project. What to install (besides these files).
* Step 1: Process a video (or videos) with OpenPose, creating JSON file per frame with wireframe data for all identified persons ('actors'). Extract video by video, frame by frame data from JSON files and combine into a single numpy array.
* Step 2: Load nparray from step 1 & perform basic validations (identifying individuals over time, tagging windows of interest, handle missing data).
* Step 3: Perform fourier analysis to extract rythmic movements and compare across groups. 
* Step 4: Calculate cross-correlations, Granger Causality (and other measures) between multiple actors in same video. *Still in development*

## Installation

To get these scripts working on a new system you need to do the following

### Prequisites
First you need to make sure you have supporting software installed. 
1. You need a working Python environment (Python v3.7 or higher) with support for Jupyter notebooks. The easiest way to do this is to install [Anaconda](https://www.anaconda.com/distribution/).
2. Install [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
3. Next you need to download the trained neural-network models that OpenPose uses. To do this go to the `models` subdirectory of OpenPose directory, and double-click / run the `models.bat` script.

### Installing this code.

You have two options to install this code. Either download the contents of this repository as a [zip file](https://github.com/InfantLab/VASC/archive/master.zip) to your local machine. 
Or if you are familiar with GitHub you can [fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) this repository and keep you own copy of the files with a version history. We recommend using the [Github Desktop](https://docs.github.com/en/desktop) app to manage this.

### Running the code

1. From your Anaconda folder launch an anaconda command prompt.
2. Create a new environment for this project from our `environment.yml` file using the command ```conda env create -f environment.yml```
3. Switch to your newly created enviroment with command `conda activate VASC`
4. Launch from the command line with command `juptyer` or `jupyter lab`. Or launch by click the Jupyter icon within Ananconda Navigator (remember to switch enviroment first in the 'Applications on ...' drowdown.)
5. Open the notebook `Step 0.GettingStarted.ipynb` and follow the instructions in there.

#### Python dependencies

The main requirements for this project are found in the `environment.yml` file in this directory. This can be used to create a new [(ana)conda](https://docs.conda.io/en/latest/) environment like so:

```bash
conda create -f environment.yml
```

#### Requirements
The main requirements are:

  - numpy, pandas, glob2, opencv
  - pyarrow, xlrd, jupytext
  - ipywidgets, ipycanvas
  - nodejs

(and their dependencies). 

## DrumTutorial
The folder `DrumTutorial` will steps you through a small example of using Fourier transforms to extract drumming tempo from a set of short videos of infants and adults drumming. It will be downloaded when you install the contents of this folder. 

## Video Walkthrough
In the meantime, you can watch this [walkthrough video](https://goldsmiths.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=5a113d3f-8f7b-454b-b3c9-ace300c99c41). 


## Scientific Background
These tools were developed for a scientific project that was aims to see if parents and babies move in synchrony with each other and whether this predicts caring outcomes. The details are found here:

_Automated measurement of responsive caregiving at scale using machine learning._
Royal Academy of Engineering  / Global Challenges Research Fund
[Overview document](https://docs.google.com/document/d/1FoBBY_XxHAFbsKjmJ1Q1dIbDrpovvo3xLh1GNzJ1ziU/edit)

### Funding:
This project was supported by the Royal Academy of Engineering Global Challenges Research Fund 
Grant:
[Frontiers of Development](https://www.raeng.org.uk/grants-and-prizes/grants/international-research-and-collaborations/frontiers/frontiers-of-development) - Tranche 2 - FoDSF\1920\2\100020



