# Video Actor Synchrony and Causality (VASC)
### Created by Caspar Addyman <c.addyman@gold.ac.uk>
#### Goldsmiths, University of London, 2020

Videos of interacting humans are converted into time series data with [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose). The data are processed and then various statistical measures of synchrony and causality between actors are calculated using [scipy](https://www.scipy.org/scipylib/index.html).

This repository provides Python code and annotated Jupyter notebooks to perform these actions.

* Step 1: Process a video (or videos) with OpenPose, creating a JSON file with wireframe data for all identified persons ('actors'). 
* Step 2: Extract JSON data to numpy and perform basic validations (identifying individuals over time, tagging windows of interest, handle missing data).
* Step 3: Calculated cross-correlations, Granger Causality (and other measures) between actors in dataset. 

Todo: 
A tutorial is providing using an example of mother infant interaction. 

Funding:
This project was supported by the Royal Academy of Engineering Global Challenges Research Fund 
Grant:
[Frontiers of Development](https://www.raeng.org.uk/grants-and-prizes/grants/international-research-and-collaborations/frontiers/frontiers-of-development) - Tranche 2 - FoDSF\1920\2\100020

