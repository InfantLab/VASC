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
#     display_name: Python 3
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

# ## 2.1 - import modules and initialise variables

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


import matplotlib.pyplot as plt
# %matplotlib inline

import vasc #a module of our own functions (found in vasc.py in this folder)

#turn on debugging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# %pdb on   
# -

# #### 2.1.1 - anonymise the videos?
#
# Setting the `anon` flag to 
#
# * `True` - we will not display just the wireframes on black backround without the underlying images from the video. 
# * `False` - we will *attempt to* draw video images - If videos are not available we fall back to anonymous mode 

anon = True

# ## 2.2 - Where are the data?
#
# This routine only needs to know where to find the processed data  and what are the base names. The summary information is listed in the `videos.json` file we created. The raw numerical data is in `allframedata.npz`.

# +
# where's the project folder?
jupwd =  os.getcwd() + "\\"
projectpath = os.getcwd() + "\\..\\SpeaknSign\\"
#projectpath = os.getcwd() + "\\..\\lookit\\"


# locations of videos and output
videos_in = projectpath 
videos_out   = projectpath + "out"
#videos_out1 = video_out
videos_out1 = "E:\\SpeakNSign\\out"
videos_out_openpose   = videos_out + "\\openpose"
videos_out_timeseries = videos_out + "\\timeseries"
videos_out_analyses   = videos_out + "\\analyses"

print(videos_out_openpose)
print(videos_out_timeseries)
print(videos_out_analyses)
# -

#retrieve the list of base names of processed videos.
try:
    with open(videos_out + '\\videos.json') as json_file:
        videos = json.load(json_file)
        print("Existing videos.json found..")
except:
    videos = {}
    print("videos.json not found in ", videos_out)

#optional - check the json
for vid in videos:  
    print(vid)
    for cam in videos[vid]:
        print(videos[vid][cam])

#can reload the values without recomputing
#reloaded = np.load(videos_out_timeseries + '\\allframedata.npz')
reloaded = np.load(videos_out_timeseries + '\\cleandata.npz')
keypoints_original = reloaded["keypoints_array"] #the unprocessed data
#keypoints_array = np.copy(keypoints_original)  #an array where we clean the data.


reloaded

keypoints_array = keypoints_original  #an array where we clean the data.
keypoints_array.shape

# ## Step 2.3 Clean the data
#
# We now have an numpy array called `keypoints_array` containing all the openpose numbers for all videos. Now we need to do some cleaning of the data. We provide set of tools to do this. There are several tasks we need to do.
#
# 1. Pick camera with best view of both participants - swap this to camera 1 (Assu
# 2. Tag the first and last frames of interest. 
# 3. Tag the adult & infant in first frame of interest. So both individuals should be in first frame.
# 4. Try to automatically tag then in subsequent frames.
# 5. Manually fix anything the automatic process gets wrong.
# 6. Exclude other detected people (3rd parties & false positives)
#
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

#let's loop through the processed list and set and startframe and endframe for each video
# for the moment we'll just use the full video.
# TODO - we will let the use specify this per video 
for vid in videos:
    for cam in videos[vid]:
        videos[vid][cam]["start"] = 0
        videos[vid][cam]["end"] = videos[vid][cam]["frames"]

# ### Step 2.3.2: Tag the actors of interest at start
#
# We want to know which person is the adult and which is the infant in the first frame. We want the child to be Person 0 and Adult to be Person 1. The buttons below provide the choice to swap data series so that this is correct.
#
# For example, if child data starts in series 3, we pick 3 in the drop down list next to button `Swap to child (0)` and then press the button. This will swap these two series.
#
# This function operates beyond the current frame so it's possible to use it multiple times if data jumps around. However, there is a short cut for part of this process..

# ### Step 2.3.3: Track actors frame by frame
#
# At present OpenPose doesn't track individuals from one frame to the next (I believe they are working on this). It just labels each person in each frame. This means that Person 1 in frame 1 might become Person 2 by frame 100. Here we provide some tools that automatically trying to guess who is who. This is tricky so we also ask for human input. 
#
# Once the child and adult data start in series 0 and 1 respectively, press the `Auto fix` button.
#
# If this leaves a few errors we can move slider to affected frame and use the swap series function to manually correct.

# ### Step 2.3.4: Exclude other people
#
# Finally we can delete any people in background or false positives (ghosts) detected by OpenPose. We simply set these to zero.

# ## CONTROL PANEL
#
# Run this BIG block of code to provide controls to edit and reorganise the data. 
# If anything goes wrong you can revert to the original data. 
#
# This needs ipywidgets and ipycanvas to be installed. (See Step 0).

# +
canvas = Canvas(width=800, height=600)

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

#pressing this button swaps one set of data to be index 0 - default for child.
button_swapchild = widgets.Button(description="Swap to child (0)")
child = widgets.Dropdown(
    options = list(range(10)),
    value= 0,
    description='Set: '
)
babybox = widgets.HBox([button_swapchild, child])
adult = widgets.Dropdown(
    options = list(range(10)),
    value= 1,
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


slider = widgets.IntSlider(
    value=0,
    min=0,
    max=161,
    step=1,
    description='Frame:',
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)


button_update = widgets.Button(description="Redraw")
button_fixseries = widgets.Button(description="Auto fix")
button_reset_one = widgets.Button(description="Reset this video")
button_reset_all = widgets.Button(description="Reset all")
buttonbox = widgets.HBox([button_update,button_fixseries,button_exclude,button_reset_one,button_reset_all])
output = widgets.Output()

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
    keypoints_array = np.copy(keypoints_original)
    updateAll(True)

def on_fixseries(output):
    global keypoints_array
    logging.info('on_fixseries')
    v = videos[pickvid.value][pickcam.value]["v"]
    c = videos[pickvid.value][pickcam.value]["c"]
    end  = videos[pickvid.value][pickcam.value]["end"]
    vasc.fixpeopleSeries(keypoints_array,v,c,[0,1],slider.value, end)
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
    updateAll(True)

def on_swapadult(output):
    global keypoints_array
    global videos
    logging.info('on_swapadult')
    v = videos[pickvid.value][pickcam.value]["v"]
    c = videos[pickvid.value][pickcam.value]["c"]
    end  = int(videos[pickvid.value][pickcam.value]["end"])
    vasc.swapSeries(keypoints_array,v,c,1,adult.value,slider.value,end)
    updateAll(True)


slider.observe(slider_change, 'value')
pickvid.observe(pickvid_change, 'value') 
pickcam.observe(pickcam_change, 'value') 
button_exclude.on_click(on_deleteparticipant) 
button_swapcam.on_click(on_swapcam)
button_swapchild.on_click(on_swapchild)
button_swapadult.on_click(on_swapadult)
button_fixseries.on_click(on_fixseries)
button_remove.on_click(on_deleteseries)
button_update.on_click(on_button_clicked)
button_reset_all.on_click(on_reset_all)

##functions to draw complicated stuff..
def drawOneFrame(vid, cam, frameNum):
    # which subarray of data do we need?
    v = videos[vid][cam]["v"]
    c = videos[vid][cam]["c"]
    if anon == True:
        #draw a black image
        frame = np.zeros((videos[vid][cam]["height"], videos[vid][cam]["width"], 3), dtype = "uint8")
    else:
        vidpath = videos[pickvid.value][pickcam.value]["fullpath"]
        frame = vasc.getframeimage(vidpath,frameNum) 
    vasc.drawPoints(frame,keypoints_array[v,c,frameNum,:,:],videos[vid][cam]["maxpeople"])
    vasc.drawLines(frame,keypoints_array[v,c,frameNum,:,:],videos[vid][cam]["maxpeople"])
    vasc.drawBodyCG(frame,keypoints_array[v,c,frameNum,:,:],videos[vid][cam]["maxpeople"])
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

def updateAll(forceUpdate = False):
    output.clear_output(wait = True)
    if forceUpdate:
        slider.value = 0
        slider.max = videos[pickvid.value][pickcam.value]["end"]
    with output:
        display(canvas,pickvid,cambox, babybox,adultbox,removebox, slider, buttonbox)  
        drawOneFrame(pickvid.value,pickcam.value,slider.value)
        drawMovementGraph(pickvid.value,pickcam.value,vasc.xs,slider.value,True)

#draw everything for first time
updateAll(True)
output
# -

# ### Step 2.4: TODO - Correct for camera motion?
#
# Some video sets the camera is not fixed. Any camera movements will cause perfectly correlated movements in the pair of signals. We need to decide what (if anything) to do about this. (Not yet implemented.)
#
#

# ### Step 2.5: TODO - Interpolate missing data
#
# There are still likely to be gaps. We need to decide what to do about those.  At the moment interpolation is done by scipy in the Step 3 code.

# ### Step 2.6: TODO - Exclude whole video
#
# Some time the data will look too bad to use. In which case, we need to completely remove this whole set. (Not yet implemented.) 

keypoints_array.shape

# ## Step 2.7: Save the numpy data!
#
# Saving the data at this stage so we don't have to repeat these steps again if we reorganise or reanalyse the data.
#
# We create a compressed NumPy array `cleandata.npz` containing the person location data for all the videos. 
#
# We also update the `videos.json` file with more info about the videos. in a new file called `clean.json`. 

# +
#update the json file in the video out directory
with open(videos_out + '\\clean.json', 'w') as outfile:
    json.dump(videos, outfile)

# in the time series folder we save the data file. 
#in a compressed format as it has a lot of empty values
np.savez_compressed(videos_out_timeseries + '\\cleandata.npz', keypoints_array=keypoints_array)
# -

# ## Step 2.8: Save a pandas dataframe version too.
#
# Most of our analysis will be done with SciPy which uses pandas dataframes as its main data format. So let's build a multiindex dataframe containing just the data we need. 
#
# The rows will have three levels of hierarchy (video x person x BODY25-coordinate). The rows are the individual frames. So a single column will contain the complete time-series of a single dimension of a single point of one person.  So in this example: 
# ```
# rows 0-411 represent the 412 frames of data.
#
# col 0 is x-coordinate of point 0 (nose) of infant in video 'lookit.01'
# col 1 is y-coordinate of point 0 (nose) of infant in video 'lookit.01'
# col 2 is openpose confidence score for how well it identified that point.
# ```
#
# <img src="multiindexdataframe.png" alt="multiindex" width="871"/>
#

#optional
#can reload the clean values without recomputing steps above
reloaded = np.load(videos_out_timeseries + '\\cleandata.npz')
keypoints_array = reloaded["keypoints_array"] #the unprocessed data
keypoints_array.shape

#delete all cameras except 0 
keypoints_array = np.delete(keypoints_array,np.s_[1:],1)
#delete all people except 0 & 1
keypoints_array = np.delete(keypoints_array,np.s_[2:],3)
shp = keypoints_array.shape


# first create an empty dataframe with right shape

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

# +
for vid in videos:
    for p in range(2) :
        v = videos[vid]["camera1"]["v"]
        part = participants[p]
        for r in range(3*vasc.nPoints):
            cleandf[(vid, part, r)] = keypoints_array[v,0,:,p,r]

cleandf
# -

#Sort the columns into alphabetical order (helps with step 3 calculations.)
cleandf = cleandf.sort_index(axis = 1)

# Finally save this to a compressed file.
#
# We use the fast `parquet` format with library `pyarrow` in order to preserve our hierarchical index in a compressed format. We save into the timeseries sub-folder. 

# +
import pyarrow as pa
import pyarrow.parquet as pq

pq.write_table(pa.Table.from_pandas(cleandf), videos_out_timeseries + '\\cleandata.parquet')


# -

print('reading parquet file:')
pqdf = pq.read_table(videos_out_timeseries + '\\cleandata.parquet').to_pandas()
print(pqdf.head())


# #### That's it. 
#
# Now go onto [Step 3 - Analyse the data](Step3.AnalyseData.scipy)
