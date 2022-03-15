# Drumming tutorial
## A demo of how to extract rhythmic movement data from videos. 
## March 2022

This page and contents of this folder helps you walk through using the VASC toolkit with a small subset of the dataset from our [Little Drummers](https://github.com/InfantLab/little-drummers) experiment. It uses the VASC toolkit scripts in the folder above to extract rate of drumming from a videos of infants banging on a table. For comparison, two videos of adults performing the same tasks are also included. More details of the experiment can be found in *Assessing sensorimotor synchronisation in toddlers using the LookIt online experiment platform and automated movement extraction* (Rocha and Addyman, 2022) [link to follow]

This folder contains

 * `Little Drummers Supplmentary Materials.docx` - A more detailed narrative account of the 
 * `LD.settings.json` - a structured file telling the scripts where to find videos and save outputs.
 * `LittleDrummers_TutorialManualCoding.xlsx` - A spreadsheet of supporting information and manual coding of drumming videos. (Used by Step 3) 
 * `videos` - a folder of 6 videos per participant (3 child, 2 adult, used with permission). 
 * `timeseries` - a folder where we store data arrays containing the generated movement data.

## Step 0 - Installation

In order to run the tutorial, download or clone a local copy of the VASC project including this tutorial. And follow the install instructions on the [main page](https://github.com/InfantLab/VASC). 

## Step 1 - Processing the videos

Open your local copy of the file [Step1.ProcessVideo.ipynb](https://github.com/InfantLab/VASC/blob/master/Step1.ProcessVideo.ipynb) from an instance of Jupyter or JupyterLab running on your local system.

This should then guide you through the process of getting OpenPose to convert each video into a set of frame by frame pose estimates. 

## Step 2 - Cleaning the data

Open your copy of [Step2.OrganiseData.ipynb](https://github.com/InfantLab/VASC/blob/master/Step2.OrganiseData.ipynb) in Jupyter.

## Step 3 - Extracting Movement

Open your copy of [Step3.ExtractMovement.ipynb](https://github.com/InfantLab/VASC/blob/master/Step3.ExtractMovement.ipynb) in Jupyter.





If you have any comments or questions, either contact [Caspar Addyman <c.addyman@gold.ac.uk](mailto:c.addyman@gold.ac.uk) or submit an [issue report](https://github.com/InfantLab/VASC/issues).
