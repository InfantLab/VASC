# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Video Actor Synchroncy and Causality (VASC)
# ## RAEng: Measuring Responsive Caregiving Project
# ### Caspar Addyman, 2020
# ### https://github.com/infantlab/VASC
#
# # Step 2: Reorganise the OpenPose JSON wire frame data
#
# This script uses output from [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) human figure recognition neural network to create labeled wireframes for each figure in each frame of a video.
#
#
# The `write_json flag` saves the people pose data using a custom JSON writer. Each JSON file has a set of coordinates and confidence scores for each person identified in the frame. For a given person there is:
#
# > An array pose_keypoints_2d containing the body part locations and detection confidence formatted as x1,y1,c1,x2,y2,c2,.... The coordinates x and y can be normalized to the range [0,1], [-1,1], [0, source size], [0, output size], etc., depending on the flag keypoint_scale (see flag for more information), while c is the confidence score in the range [0,1].
#
# <img src="keypoints_pose_25.png" alt="BODY-25 mapping" width="240"/>

# ## 2.0 - import modules and initialise variables

# +
import os                #operating system functions
import math              #simple math
import glob              #file listing
import json              #importing and exporting json files
import cv2               #computervision toolkit
import numpy as np       #tools for numerical data
import pandas as pd
import logging
import ipywidgets as widgets  #let's us add buttons and sliders to this page.
from ipycanvas import Canvas
from datetime import datetime
from pprint import pprint

import matplotlib.pyplot as plt
# %matplotlib inline

import vasc #a module of our own functions (found in vasc.py in this folder)

#turn on debugging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# %pdb on
# -

# ## 2.1 Settings?
#
# Load a json file that tells us where to find our videos and where to save the data. You should create a different settings file for each project. Then you don't need to change any other values in the script for Step 1 or Step 2.
#
# TODO - write a helper to create a settings file
#

# +
#settingsjson = "C:\\Users\\cas\\OneDrive - Goldsmiths College\\Projects\\Little Drummers\\VASC\\settings.json"
settingsjson = "E:\\little.drummer.20220111\\LD.settings.json"

try:
    with open(settingsjson) as json_file:
        settings = json.load(json_file)
        print("Existing settings.json found..")
except json.JSONDecodeError:
    logging.exception("Settings file was not valid JSON.")
except Exception as e:
        emsg = str(e)
        #show the error
        print("Error: ",emsg)
        print("No setting.json file found!\nPlease see Step 0 for instructions")
# -

# #### 2.1.1 - anonymise the videos?
#
# Setting the `anon` flag to
#
# * `True` - we will not display just the wireframes on black backround without the underlying images from the video.
# * `False` - we will *attempt to* draw video images - If videos are not available we fall back to anonymous mode

anon = settings["flags"]["anon"]

# Did we get OpenPose to include wireframes for the hands?

includeHands = settings["flags"]["includeHands"]

# ## 2.2 - Where are the data?
#
# This routine only needs to know where to find the processed data  and what are the base names. The summary information is listed in the `videos.json` file we created. The raw numerical data is in `allframedata.npz`.

# +
# where's the project data folder? (with trailing slash)
projectpath = settings["paths"]["project"]
#where are your video files?
videos_in = settings["paths"]["videos_in"]

# locations of videos and output
videos_out = settings["paths"]["videos_out"]
videos_out_openpose   = settings["paths"]["videos_out_openpose"]
videos_out_timeseries = settings["paths"]["videos_out_timeseries"]
videos_out_analyses   = settings["paths"]["videos_out_analyses"]

print(videos_in)
print(videos_out)
print(videos_out_openpose)
print(videos_out_timeseries)
print(videos_out_analyses)
# -

#retrieve the list of base names of processed videos.
videosjson = settings["paths"]["videos_out"] + '\\' + settings["filenames"]["videos_json"]
try:
    with open(videosjson) as json_file:
        videos = json.load(json_file)
        print("Existing videos.json found..")
except:
    videos = {}
    print("Creating new videos.json")

#optional - check the json
for vid in videos:
    print(vid)
    for cam in videos[vid]:
        print(videos[vid][cam])

# ### 2.2.2 Original or clean?
#
# Data from Step 1 was saveed in our "alldatanpz" file. Once cleaned, a new copy of data will be saved in "cleandatanpz"
#
# It will often take time to clean data so we save progess as we go along. To do this non-distructively we create a new
#
# The `cleaned` flag tells us whether to start from original data or from a (partially) cleaned set.

# +
# uncomment next line to manually set cleaned flag
settings["flags"]["cleaned"] = False

print("cleaned = {0}".format(settings["flags"]["cleaned"]))


# +
if not settings["flags"]["cleaned"]:
    #can reload the values without recomputing
    reloaded = np.load(videos_out_timeseries +  "\\" + settings["filenames"]["alldatanpz"])
    if includeHands:
        LH_npz = np.load(videos_out_timeseries +  "\\" + settings["filenames"]["lefthandnpz"])
        RH_npz = np.load(videos_out_timeseries +  "\\" + settings["filenames"]["righthandnpz"])
else:
    #or we can load an cleaned/partially cleaned dataset..
    reloaded = np.load(videos_out_timeseries  +  "\\" + settings["filenames"]["cleannpz"])
    if includeHands:
        LH_npz = np.load(videos_out_timeseries +  "\\" + settings["filenames"]["cleanleftnpz"])
        RH_npz = np.load(videos_out_timeseries +  "\\" + settings["filenames"]["cleanrightnpz"])

keypoints_array = np.copy(reloaded["keypoints_array"])  #an array where we clean the data.

if includeHands:
    LH = np.copy(LH_npz["keypoints_array"])  #an array where we clean the data.
    RH = np.copy(RH_npz["keypoints_array"])  #an array where we clean the data.
else:
    LH = None
    RH = None
# -


#check the shape
keypoints_array.shape

# ## Step 2.3 Clean the data
#
# We now have an numpy array called `keypoints_array` containing all the openpose numbers for all videos. Now we need to do some cleaning of the data. We provide set of tools to do this. There are several tasks we need to do.
#
# 1. Pick camera with best view of both participants - swap this to camera 1 (if multiple cameras).
# 2. You might delete sets for whom all data is too poor quality. But they can also be excluded in **Step 3** by a flag in the data spreadsheet.
# 3. Tag the adult & infant in first frame of interest. So both individuals should be in first frame.
# 4. Try to automatically tag then in subsequent frames.
# 5. Manually fix anything the automatic process gets wrong.
# 6. Exclude other detected people (3rd parties & false positives)
#
# We do all of this with the control panel below.
#

# ### Step 2.3.1: Which is best camera angle?
#
# If we have just one camera then use that. If there are multiple angles, pick the best one and swap it to be "camera1".
#

# ### Step 2.3.2: Where does the interesting data start and end?
#
# For many videos, the period of interest might start (and end) some time into the video. For now we are using the whole video .
# TODO  - We wiil give the user the opportunity to set these.
#

# +
#let's loop through the processed list and set and startframe and endframe for each video
# for the moment we'll just use the full video.
# also might specify which hand to code left or right
# In step 3 we let the user specify this per video

starttime = 0 
starttime = 5  #(in seconds)

for vid in videos:
    for cam in videos[vid]:
        videos[vid][cam]["side"] = "right"
        videos[vid][cam]["start"] = int(starttime * videos[vid][cam]["fps"]) #convert time to number of frames
        videos[vid][cam]["end"] = videos[vid][cam]["frames"]
# -

# ### Step 2.3.2: Tag the actors of interest at start
#
# We want to know which person is the adult and which is the infant in the first frame. We want the child to be Person 0 and Adult to be Person 1. The buttons below provide the choice to swap data series so that this is correct.
#
# For example, if child data starts in series 3, we pick 3 in the drop down list next to button `Swap to child (0)` and then press the button. This will swap these two series.
#
# This function operates beyond the current frame so it's possible to use it multiple times if data jumps around. However, there is a short cut for part of this process..

# ### Step 2.3.3: Fix by location - Track actors frame by frame
#
# At present OpenPose doesn't track individuals from one frame to the next (I believe they are working on this). It just labels each person in each frame. This means that Person 1 in frame 1 might become Person 2 by frame 100. Here we provide some tools that automatically trying to guess who is who. This is tricky so we also ask for human input.
#
# Once the child and adult data start in series 0 and 1 respectively, press the `Fix by location` button.
#
# If this leaves a few errors we can move slider to affected frame and use the swap series function to manually correct.

# ### Step 2.3.4: Fix by size - identify data by size of wireframes.
#
# In many of our cases of interest we have an adult and a young child interacting. Therefore, it is handy to try autolabelling based on a sorting of the size of their wireframes.
#
# Pressing the `Fix by size` button, makes person 0 the smallest person in the frame, person 1 the next smallest and so on.

# ### Step 2.3.4: Exclude other people
#
# Finally we can delete any people in background or false positives (ghosts) detected by OpenPose. We simply set these to zero.

# ## Step 2.4: CLEAN UP CONTROL PANEL
#
# Run this BIG block of code to provide controls to edit and reorganise the data.
# If anything goes wrong you can revert to the original data.
#
# This needs `ipywidgets` and `ipycanvas` to be installed. (See [Step 0](Step0.GettingStarted.ipynb))

# +
###############################################################
# All the code in this block draws our data editing control panel
###############################################################

###############################################################
## a canvas object to show current frame of the video.
canvas = Canvas(width=800, height=600)

###############################################################
## dropbown lists to select the video and the camera (if multiple angles)
## next to the video dropdown we have Delete button to remove that vid
## next to the camera dropdown we have a swap button to choose a primary camera.
vidlist = [] #used to fill dropdown options
camlist = [] #used to fill dropdown options
for vid in videos:
    vidlist.append(vid)

for cam in videos[vid]:
    camlist.append(cam)

pickvid = widgets.Dropdown(
    options= vidlist,
    value= vidlist[0],
    description='Select subject:'
)
button_exclude =  widgets.Button(description='DELETE THIS ONE!')

pickcam = widgets.Dropdown(
    options= camlist,
    value= camlist[0],
    description='Select camera:'
)
button_swapcam = widgets.Button(description="Swap this to camera1")
cambox = widgets.HBox([pickcam, button_swapcam])

###############################################################
## Who is who?
## In processed video we want child in index 0 and adult in index 1.
## so need ability to swap series and delete unwanted data.
## pressing button_swapchild swaps selected set of data to be index 0 - default for child.

button_swapchild = widgets.Button(description="Swap to child (0)")
child = widgets.Dropdown(
    options = list(range(10)),
    value= 1,
    description='Set: '
)
babybox = widgets.HBox([button_swapchild, child])

adult = widgets.Dropdown(
    options = list(range(10)),
    value= 0,
    description='Set: '
)
button_swapadult = widgets.Button(description="Swap to adult (1)")
adultbox = widgets.HBox([button_swapadult,adult])

button_remove = widgets.Button(description="Remove these data")
remove = widgets.Dropdown(
    options = list(range(10)),
    value= 2,
    description='Set: '
)
removebox = widgets.HBox([button_remove,remove])


###############################################################
## What frame is displayed in the canvas?
## all swap and delete operation work on data AFTER this frame.
## include a few buttons to adjust the frame forward or backwards slightly
slider = widgets.IntSlider(
    value=0,
    min=0,
    max=161,
    step=1,
    description='Frame:',
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d',
    layout=widgets.Layout(width='800px')
)

#buttons to adjust the slider in small increments
minus1pct = widgets.Button(description="-1%")
minus10 = widgets.Button(description="-10")
minus1 = widgets.Button(description="-1")
plus1 = widgets.Button(description="+1")
plus10 = widgets.Button(description="+10")
plus1pct = widgets.Button(description="+1%")

def minus1pct_clicked(output):
    slider.value = max(0,slider.value - 0.01 * slider.max)
def minus10_clicked(output):
    slider.value = max(0,slider.value - 10)
def minus1_clicked(output):
    slider.value = max(0,slider.value - 1)
def plus1_clicked(output):
    slider.value = min(slider.max,slider.value + 1)
def plus10_clicked(output):
    slider.value = min(slider.max,slider.value + 10)
def plus1pct_clicked(output):
    slider.value = min(slider.max,slider.value + 0.01 * slider.max)

minus1pct.on_click(minus1pct_clicked)
minus10.on_click(minus10_clicked)
minus1.on_click(minus1_clicked)
plus1.on_click(plus1_clicked)
plus10.on_click(plus10_clicked)
plus1pct.on_click(plus1pct_clicked)

adjustbox  = widgets.HBox([minus1pct,minus10,minus1,plus1,plus10,plus1pct])

###############################################################
## Action buttons
## To redraw everything it's current state, to attempt autofixing or to undo some or all our changes

button_update = widgets.Button(description="Redraw")
button_fixlocations = widgets.Button(description="Fix by location",tooltip="match each person to nearest person in next frame")
button_fixsizes = widgets.Button(description="Fix by size",tooltip="label people sequentially by size of their wireframe")
button_reset_one = widgets.Button(description="Reset this video")
button_reset_all = widgets.Button(description="Reset all")
buttonbox = widgets.HBox([button_update,button_fixlocations,button_fixsizes,button_exclude,button_reset_one,button_reset_all])
output = widgets.Output()


###############################################################
## Widget 'Event' codes
## watches each widget waiting for something to change and then executes these bits of code.

def pickvid_change(change):
    if change['name'] == 'value' and (change['new'] != change['old']):
        updateAll(True)

def pickcam_change(change):
    if change['name'] == 'value' and (change['new'] != change['old']):
        updateAll(True)

def slider_change(slider):
    updateAll(False)

def on_button_clicked(output):
    logging.info('button_update_all clicked')
    updateAll(True)

def on_reset_all(output):
    global keypoints_array
    logging.info('button_reset_all clicked')
    keypoints_array = np.copy(reloaded["keypoints_array"])
    updateAll(True)

def on_fixlocations(output):
    global keypoints_array
    logging.info('on_fixlocations')
    v = videos[pickvid.value][pickcam.value]["v"]
    c = videos[pickvid.value][pickcam.value]["c"]
    end  = videos[pickvid.value][pickcam.value]["end"]
    window = 10
    vasc.fixpeopleSeries(keypoints_array,v,c,[0,1],slider.value, end, window, includeHands, LH, RH)
    updateAll(True)

def on_fixsizes(output):
    global keypoints_array
    logging.info('on_fixsizes')
    v = videos[pickvid.value][pickcam.value]["v"]
    c = videos[pickvid.value][pickcam.value]["c"]
    N = videos[vid][cam]["maxpeople"]
    end  = videos[pickvid.value][pickcam.value]["end"]
    vasc.sortpeoplebySize(keypoints_array,v,c,N,slider.value, end, includeHands, LH, RH)
    updateAll(True)

def on_deleteparticipant(output):
    global keypoints_array
    global videos
    logging.info('on_deleteparticipant')
    for cam in videos[pickvid.value]: #loop through delete all cameras for this video.
        v = videos[pickvid.value][cam]["v"]
        c = videos[pickvid.value][cam]["c"]
        end  = videos[pickvid.value][cam]["end"]
        vasc.deleteSeries(keypoints_array,v,c,remove.value,0, end)
        if includeHands:
            vasc.deleteSeries(LH,v,c,remove.value,0, end)
            vasc.deleteSeries(RH,v,c,remove.value,0, end)
    #now remove this from videos object
    if pickvid.value in videos:
        logging.info(pickvid.value)
        del videos[pickvid.value]
    #repopulate the dropdown
    for vid in videos:
        vidlist.append(vid)
    pickvid.options = vidlist
    updateAll(True)

def on_deleteseries(output):
    global keypoints_array
    global videos
    logging.info('on_fixseries')
    v = videos[pickvid.value][pickcam.value]["v"]
    c = videos[pickvid.value][pickcam.value]["c"]
    end  = videos[pickvid.value][pickcam.value]["end"]
    vasc.deleteSeries(keypoints_array,v,c,remove.value,slider.value, end)
    if includeHands:
        vasc.deleteSeries(LH,v,c,remove.value,slider.value, end)
        vasc.deleteSeries(RH,v,c,remove.value,slider.value, end)
    updateAll(True)

def on_swapcam(output):
    global keypoints_array
    global videos
    logging.info('on_swapcam')
    print(videos[pickvid.value][pickcam.value])
    videos, keypoints_array = vasc.swapCameras(videos, keypoints_array,pickvid.value,pickcam.value,"camera1")
    updateAll(True)

def on_swapchild(output):
    global keypoints_array
    global videos
    logging.info('on_swapchild')
    v = videos[pickvid.value][pickcam.value]["v"]
    c = videos[pickvid.value][pickcam.value]["c"]
    end  = videos[pickvid.value][pickcam.value]["end"]
    vasc.swapSeries(keypoints_array,v,c,0,child.value,slider.value,end)
    if includeHands:
        vasc.swapSeries(LH,v,c,0,child.value,slider.value,end)
        vasc.swapSeries(RH,v,c,0,child.value,slider.value,end)
    updateAll(True)

def on_swapadult(output):
    global keypoints_array
    global videos
    logging.info('on_swapadult')
    v = videos[pickvid.value][pickcam.value]["v"]
    c = videos[pickvid.value][pickcam.value]["c"]
    end  = int(videos[pickvid.value][pickcam.value]["end"])
    vasc.swapSeries(keypoints_array,v,c,1,adult.value,slider.value,end)
    if includeHands:
        vasc.swapSeries(LH,v,c,1,adult.value,slider.value,end)
        vasc.swapSeries(RH,v,c,1,adult.value,slider.value,end)
    updateAll(True)


slider.observe(slider_change, 'value')
pickvid.observe(pickvid_change, 'value')
pickcam.observe(pickcam_change, 'value')
button_exclude.on_click(on_deleteparticipant)
button_swapcam.on_click(on_swapcam)
button_swapchild.on_click(on_swapchild)
button_swapadult.on_click(on_swapadult)
button_fixsizes.on_click(on_fixsizes)
button_fixlocations.on_click(on_fixlocations)
button_remove.on_click(on_deleteseries)
button_update.on_click(on_button_clicked)
button_reset_all.on_click(on_reset_all)

###############################################################
## ## functions to draw complicated stuff..
def drawOneFrame(vid, cam, frameNum):
    # which subarray of data do we need?
    v = videos[vid][cam]["v"]
    c = videos[vid][cam]["c"]
    #print("drawOneFrame v:{0}, c:{1}".format(v, c))
    if anon == True:
        #draw a black image
        frame = np.zeros((videos[vid][cam]["height"], videos[vid][cam]["width"], 3), dtype = "uint8")
    else:
        vidpath = videos[vid][cam]["fullpath"]
        frame = vasc.getframeimage(vidpath,frameNum)
    vasc.drawPoints(frame,keypoints_array[v,c,frameNum,:,:],videos[vid][cam]["maxpeople"])
    vasc.drawLines(frame,keypoints_array[v,c,frameNum,:,:],videos[vid][cam]["maxpeople"])
    vasc.drawBodyCG(frame,keypoints_array[v,c,frameNum,:,:],videos[vid][cam]["maxpeople"])
    if includeHands:
        vasc.drawHands(frame,RH[v,c,frameNum,:,:],videos[vid][cam]["maxpeople"])
        vasc.drawHands(frame,LH[v,c,frameNum,:,:],videos[vid][cam]["maxpeople"])
    #send the image to the canvas
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hiddencanvas = Canvas(width=img.shape[1], height=img.shape[0])
    hiddencanvas.put_image_data(img, 0, 0)
    canvas.draw_image(hiddencanvas,0,0,canvas.width,canvas.height)
    canvas.restore()

def drawMovementGraph(vid, cam, points, frame = 0, average = True):
    v = videos[vid][cam]["v"]
    c = videos[vid][cam]["c"]
    N = videos[vid][cam]["frames"]
    t = np.zeros([N,1])
    t[:,0]= list(range(N))
    #print("drawMovementGraph v:{0}, c:{1}".format(v, c))
    
    #variable to track the centre of gravity for each person
    ceegees = np.zeros([N,videos[vid][cam]["maxpeople"]])

    for frameNum in range(N):
        for p in range(videos[vid][cam]["maxpeople"]):
            personkeypoints = keypoints_array[v,c,frameNum,p,:]
            avx = vasc.averagePoint(personkeypoints,vasc.xs)
            if (avx > 0):
                ceegees[frameNum,p] = avx
            else:
                ceegees[frameNum,p] = None

    plt.figure(figsize=(12, 4))
    plt.plot(t,ceegees)
    plt.axvline(x=frame,c='tab:cyan')
    plt.title('Horizontal movement of people (average) over time.)')
    plt.legend([0, 1, 2, 3])
    plt.show()

###############################################################
## Handy update routine to run each time something has changed
def updateAll(forceUpdate = False):
    output.clear_output(wait = True)
    if forceUpdate:
        slider.value = 0
        slider.max = videos[pickvid.value][pickcam.value]["end"]
    with output:
        drawOneFrame(pickvid.value,pickcam.value,slider.value)
        drawMovementGraph(pickvid.value,pickcam.value,vasc.xs,slider.value,True)
        display(canvas,pickvid,cambox, babybox,adultbox,removebox, buttonbox, slider, adjustbox)
        

#draw everything for first time
updateAll(True)
output
# -

# ### Step 2.4.1: TODO - Correct for camera motion?
#
# Some video sets the camera is not fixed. Any camera movements will cause perfectly correlated movements in the pair of signals. We need to decide what (if anything) to do about this. (Not yet implemented.)
#
#

# ### Step 2.5: TODO - Interpolate missing data
#
# There are still likely to be gaps. We need to decide what to do about those.  At the moment interpolation is done by scipy in the Step 3 code.
#
# #### Step 2.5.1. TODO - autofix to cope with missing data
#
# Missing data currently confuses autofix and on it's own interpolation won't help here. Because you can't interpolate until you know who is who. Our current approach is to let autofix by location use a moving average of several previous frames.

# ### Step 2.6: TODO - Save your game
#
# Ought to be able to save the array when you half way through cleaning it. So you don't lose progress can come back another time.
# You can do this manually at the moment by running step 2.7 to save current progress and then restarting by reloading `cleandata.npz` in step 2.2.1

keypoints_array.shape

# ## Step 2.7: Save the numpy data!
#
# ### *Warning, with BIG datasets, these steps can take multiple minutes each...*
#
# Saving the data at this stage so we don't have to repeat these steps again if we reorganise or reanalyse the data.
#
# We create a compressed NumPy array `cleandata.npz` containing the person location data for all the videos.
#
# We also update the `videos.json` file with more info about the videos. in a new file called `clean.json`.
#

# +
#update the json file in the video out directory
with open(videos_out + '\\' + settings["filenames"]["clean_json"], 'w') as outfile:
    json.dump(videos, outfile)

# in the time series folder we save the data file.
#in a compressed format as it has a lot of empty values
np.savez_compressed(videos_out_timeseries + '\\' + settings["filenames"]["cleannpz"] , keypoints_array=keypoints_array)
if includeHands:
    np.savez_compressed(videos_out_timeseries + '\\' + settings["filenames"]["cleanleftnpz"] , keypoints_array=LH)
    np.savez_compressed(videos_out_timeseries + '\\' + settings["filenames"]["cleanrightnpz"] , keypoints_array=RH)

print("Data saved")


#make a note in our settings file so that we used cleandata next time
settings["flags"]["cleaned"] = True
#after each new id we save the json data
settings["lastUpdate"] = datetime.now().isoformat()
with open(settingsjson, 'w') as outfile:
    json.dump(settings, outfile)
    print('settings.json updated')
    
# -


# ## Step 2.8: Save a pandas dataframe version too.
#
# Most of our analysis will be done with SciPy which uses pandas dataframes as its main data format. So let's build a multiindex dataframe containing just the data we need.
#
# The rows will have three levels of hierarchy (video x person x BODY25-coordinate). The rows are the individual frames. So a single column will contain the complete time-series of a single dimension of a single point of one person.  So in this example:
# ```
#  rows 0-411 represent the 412 frames of data.
#
#  col 0 is x-coordinate of point 0 (nose) of infant in video 'lookit.01'
#  col 1 is y-coordinate of point 0 (nose) of infant in video 'lookit.01'
#  col 2 is openpose confidence score for how well it identified that point.
# ```
#
# <img src="multiindexdataframe.png" alt="multiindex" width="871"/>
#

# +
#optional
#can reload the clean values without recomputing steps above
reloaded = np.load(videos_out_timeseries + '\\' + settings["filenames"]["cleannpz"])
keypoints_array = reloaded["keypoints_array"] #the unprocessed data
keypoints_array.shape

#TODO reload hands
# -

keypoints_array.shape

# +
#delete all cameras except 0
#keypoints_array = np.delete(keypoints_array,np.s_[1:],1)
#delete all people except 0 & 1
keypoints_array = np.delete(keypoints_array,np.s_[2:],3)

if includeHands:
    #Same for LH & RH
    LH = np.delete(LH,np.s_[1:],1)
    LH = np.delete(LH,np.s_[2:],3)
    RH = np.delete(RH,np.s_[1:],1)
    RH = np.delete(RH,np.s_[2:],3)
# -


#truncate the timeseries - many videos are longer than we need
keypoints_array = np.delete(keypoints_array,np.s_[10000:],2)
shp = keypoints_array.shape
keypoints_array.shape


# Another save point - this array is much smaller so will load / Save quicker.

np.savez_compressed(videos_out_timeseries + '\\trimdata.npz', keypoints_array=keypoints_array)

#Another save point here if it helps.
trimmed = np.load(videos_out_timeseries + '\\trimdata.npz')
keypoints_array = trimmed["keypoints_array"] #the unprocessed data
keypoints_array.shape


# Now we reorganise the data in a multiindex pandas array and save using `pyarrow`.
# First create an empty dataframe with right shape

# +
#first list the three levels of row hierarchy
toplevel = videos.keys()
participants = ["infant","parent"]
coords = list(range(3*vasc.nPoints)) #we have 3 x 25 coordinates to store

#columns are frames
timeseries = list(range(shp[2])) #how big is third dimension of the array?

col_names = ['video','person','coord']
#row_names = ['frames']

col_index = pd.MultiIndex.from_product([toplevel,participants,coords], names=col_names)

cleandf = pd.DataFrame(columns=col_index, index = timeseries)
#cleandf.head()


# -

# Then populate the dataframe row by row.
#
# *This step is particularly SLOW*

for vid in videos:
    for p in range(2) :
        v = videos[vid]["camera1"]["v"]
        part = participants[p]
        for r in range(3*vasc.nPoints):
            cleandf[(vid, part, r)] = keypoints_array[v,0,:,p,r]

#Sort the columns into alphabetical order (helps with step 3 calculations.)
cleandf = cleandf.sort_index(axis = 1)

if includeHands:
    coords = list(range(3*vasc.handPoints)) #we have 3 x 21 coordinates to store
    col_index = pd.MultiIndex.from_product([toplevel,participants,coords], names=col_names)
    cleanLH = pd.DataFrame(columns=col_index, index = timeseries)
    cleanRH = pd.DataFrame(columns=col_index, index = timeseries)
    for vid in videos:
        for p in range(2) :
            v = videos[vid]["camera1"]["v"]
            part = participants[p]
            for r in range(3*vasc.handPoints):
                cleanLH[(vid, part, r)] = LH[v,0,:,p,r]
                cleanRH[(vid, part, r)] = RH[v,0,:,p,r]
    cleanLH = cleanLH.sort_index(axis = 1)
    cleanRH = cleanRH.sort_index(axis = 1)

# ### Finally save this to a compressed file.
#
# We use the fast `parquet` format with library `pyarrow` in order to preserve our hierarchical index in a compressed format. We save into the timeseries sub-folder.
#

settings["flags"]["cleaned"] = True

import pyarrow.parquet as pq
import pyarrow as pa

# +
pq.write_table(pa.Table.from_pandas(cleandf), videos_out_timeseries + '\\' + settings["filenames"]["cleandataparquet"])

if includeHands:
    pq.write_table(pa.Table.from_pandas(cleanLH), videos_out_timeseries + '\\' + settings["filenames"]["lefthandparquet"])
    pq.write_table(pa.Table.from_pandas(cleanRH), videos_out_timeseries + '\\' + settings["filenames"]["righthandparquet"])
















# -

#after each new id we save the json data
updateTime = datetime.now()
settings["lastUpdate"] = updateTime.isoformat()
with open(settingsjson, 'w') as outfile:
    json.dump(settings, outfile)
    print('settings.json updated')


#Optional check
print('reading parquet file:')
pqdf = pq.read_table(videos_out_timeseries + '\\cleandata.parquet').to_pandas()
print(pqdf.head())


# #### That's it.
#
# Now go onto [Step 3 - Analyse the data](Step3.AnalyseData.scipy.ipynb)
