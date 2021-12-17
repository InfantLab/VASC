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
#
# ## RAEng: Measuring Responsive Caregiving Project
# ### Caspar Addyman, 2020
# ### https://github.com/infantlab/VASC
#
# # Step 1  Process videos using OpenPose
#
# This script uses [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) human figure recognition neural network to create labeled wireframes for each figure in each frame of a video. OpenPoseDemo will go through a video frame by frame outputing a JSON file for each frame that contains a set of coordinate points and for a wireframe for each video.

# ## 1.0 - Libraries

# +
#import the python libraries we need
import os
import sys
import time
import glob
import json
import cv2               #computervision toolkit
import logging
import numpy as np
import pandas as pd
from datetime import datetime

#turn on debugging
# %pdb on
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# -

# ### 1.1 Settings?
#
# Load a json file that tells us where to find our videos and where to save the data.  You should create a different settings file for each project. Then you don't need to change any other values in the script for Step 1 or Step 2. 
#
# `TODO - write a helper to create a settings file`

# +
settingsjson = "C:\\Users\\cas\\OneDrive - Goldsmiths College\\Projects\\Little Drummers\\VASC\\settings.json"

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

#optional - have a look at our settings to make sure they're what we expect
print(settings)

# ### 1.2 Where is OpenPose?
#
# We need the full path to your openpose directory (this is in settings)

# +
# location of openposedemo - THIS WILL BE DIFFERENT ON YOUR COMPUTER
openposepath = settings["paths"]["openpose"]

if sys.platform == "win32":
    app = "bin\\OpenPoseDemo.exe"
else:
    app = 'bin\\OpenPoseDemo.bin'

openposeapp = openposepath + app
print(openposeapp)

# + [markdown] tags=[]
# ### 1.3 Where are your videos?
#
# In the next cell you need to specify the folder with your set of video files. So that we process them. These scripts use the following directory structure. It expects your videos to be in a subfolder of your project 
#
# ```
# path\to\project\myvideos
# ```
#
# and it expects a folder `out` in the project at the same level as the videos with three subfolders for JSON files, the aggregated timeseries and the analyses. 
#
# You will need to create all three of these. Or copy them from the VASC project folder.
#
# ```
# path\to\project\out\openpose
# path\to\project\out\timeseries
# path\to\project\out\analyses
# ```

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

# ## 1.4 Load the videos 
#
# We have two ways of loading in videos. Either from an Excel file of names and paths to the videos. Or by scanning contents of a particular folder. But first 
#
# ### 1.4.1 Retrieve or create videos.json
#
# We store information about the videos we are processing in a file calle `videos.json`, located at the top level of the `out` folder. 
#
# So we see if we already have this file. Maybe we've done some processing already.
#
# If you want to start afresh just delete this single file and everything gets reprocessed from scratch.

#retrieve the list of base names of processed videos.
videosjson = settings["paths"]["videos_out"] + '\\' + settings["filenames"]["videos_json"]
try:
    with open(videosjson) as json_file:
        videos = json.load(json_file)
        print("Existing videos.json found..")
except:
    videos = {}
    print("Creating new videos.json")

# ### EITHER 
# ### 1.4.2.A Read an Excel file of videos
#
# We expect the first column of the spreadsheet to tell us the base name for each participant and columns 2 to 4 contains the full name and location of the videos.
#
# There maybe be multiple camera angles of for same session. 
#
# ```
# Subject   | camera1      | camera2        | camera3 
# subj1     | pathtocam1.1 | pathtocam1.2   | pathtocam1.3 
# subj2     | pathtocam2.1 | pathtocam2.2   | pathtocam3.3 
# ...
# ```
#
# If there aren't these columns can be blank in the spreadsheet.
#
# ```
# Subject   | camera1      | camera2        | camera3 
# subj1     | pathtocam1.1 |                |   
# subj2     | pathtocam2.1 |                |  
# ...
# ```
#
#
# The names come from the spreadsheet so we set a flag `namesfromfiles = False`.

# +
excelpath = "U:\\Caspar\\SS_CARE.xlsx"
videolist = pd.read_excel(excelpath)

namesfromfiles = False
cameras = ["camera1", "camera2", "camera3"]


# iterate through each row and select  
# 'Name' and 'Stream' column respectively. 
for ind in videolist.index :
    vid = videolist["subject"][ind]
    
    if vid in videos:
        #we already have some info about this video
        print(vid, "found.")
    else:
        #generate an structure to hold some info about this video
        videos[vid] = {}
    
    for cam in cameras:
        #do we have a column for this camera in the spreadsheet?
        if cam in videolist.columns:
            #if so what is path of video it has
            fullpath = videolist[cam][ind] 
            #maybe we processed this already 
            if cam in videos[vid] and videos[vid][cam]["fullpath"] ==  fullpath:
                print(cam, videos[vid][cam]["stemname"], "already processed")
                print("Exit code:", videos[vid][cam]["openpose"]["exitcode"])
                print("Date:", videos[vid][cam]["openpose"]["when"])
            else:
                fullname = os.path.basename(fullpath) 
                stemname, fmt = os.path.splitext(fullname)
                videos[vid][cam] = {}
                videos[vid][cam]["shortname"] = vid + "." + cam
                videos[vid][cam]["stemname"] = stemname
                videos[vid][cam]["fullname"] = fullname
                videos[vid][cam]["fullpath"] = fullpath
                videos[vid][cam]["index"] = None         #the numerical index this data will have in np.array.
                videos[vid][cam]["format"] = fmt
                videos[vid][cam]["openpose"] =  {"exitcode" : None, "when" : None} 

    
print(videos)
# -

# ### Or 
# ### 1.4.3.B Scanning all videos in particular folder 
#
# In which case we look at all videos in `videos_in` let the names of the files also provide the base names for each participant we create.  
#
# We will reference these files by the video names so myvid1.avi is found in `videos["myvid1"]`.
#
# However, in other cases we will allow for possibility of multiple camera angles so this defaults to `"camera1"`.
#
# We set a flag `namesfromfiles = True`.
#
# If your vidoes aren't appearing there might be too many or too few trailing slashes in your path.

#quick check to see if we can find videos
mp4s =     glob.glob(videos_in + "*/*.mp4", recursive = True)
print("We found %d mp4s" % len(mp4s))

#just checking i'm loooking int the right place
videos_in

# +
#first get list of videos in the inbox
avis =     glob.glob(videos_in + "*/*.avi", recursive = True)
mp4s =     glob.glob(videos_in + "*/*.mp4", recursive = True)
threegps = glob.glob(videos_in + "*/*.3gp", recursive = True)

print("We found %d avis" % len(avis))
print("We found %d mp4s" % len(mp4s))
print("We found %d 3gps" % len(threegps))

#For the moment we will manually specify what videos to process. 
#TODO generate a list of force or skip videos to automate things slightly
allvideos = []
allvideos.extend(avis)
allvideos.extend(mp4s)
allvideos.extend(threegps)

namesfromfiles = True


for thisvid in allvideos:
    #first we need base name of video for the output file name
    #we will reference these files by the video names so myvid1.avi is found in videos["myvid1"]0
    fullname = os.path.basename(thisvid)
    vid, fmt = os.path.splitext(fullname) 
    #generate an structure to hold some info about this video
    if vid in videos: 
        print(vid + " already in videos.json")
    else:
        print("Adding " + vid + " to videos.json")
        videos[vid] = {}  
        cam = "camera1"
        videos[vid][cam] = {} 
        videos[vid][cam]["shortname"] = vid + "." + cam
        videos[vid][cam]["stemname"] = vid
        videos[vid][cam]["fullname"] = fullname
        videos[vid][cam]["fullpath"] = thisvid
        videos[vid][cam]["index"] = None         #the numerical index this data will have in np.array.
        videos[vid][cam]["format"] = fmt
        videos[vid][cam]["openpose"] =  {"exitcode" : None, "when" : None} 
# -

# ### 1.5 Calling the OpenPose app
# To operate OpenPose we pass a set of parameters to the demo executable. For the full list of options see  [OpenPoseDemo](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/demo_overview.md)
#
# Our main parameters are
#
# ```
# --video        path\to\video_to_process   #input video
# --write_json   path\to\output_directory   #one json file per frame
# --write_video  path\to\output_directory   #video with identified figures
# --write_images path\to\output_directory   #one image per frame with wireframes
# --disable_blending true/false             # wireframes on black background (true) or blended on top of video (false)
#  ```
#
# Other useful params
#  ```
# --hand                #include a model and points for the hands. 
# --frame_first  100    #start from frame 100
# --display 0           #don't show the images as they are processed
#  ```
#

#should we generate and process points for hand data?
includeHands = settings["flags"]["includeHands"]

# +
#put all params in a dictionary object
params = dict()
params["write_json"] = videos_out_openpose
# params["write_images"] = videos_out_openpose  #for the moment dump images in output file - TODO name subfolder
#params["disable_blending"] = "false"
params["display"]  = "1"
if includeHands:
    params["hand"] = ""

createoutputvideo = True #do we get openpose to create a video output?
# -

# ### The main openpose loop
#
# Call the `openposedemo` app for each of the videos at a time. For each one print the full command that we use so that you can use it manually to investigate any errors. 
#
# Finally, we write a list of the processed videos to a file called `videos.json`. 
# Note that we will add other information to this file as we go through other steps. 
#
# TODO - We might use videos.json to let this processing happen in blocks. i.e. not calling for openpose if video is already processed.

# +
currdir =  os.getcwd() + "\\" #keep track of current directory so we can change back to it after processing

optstring = ""
for key in params:
    optstring += " --" + key +  ' "' + params[key] + '"' #need to quote paths 

print(optstring)


count = 0
os.chdir(openposepath)
for vid in videos:
    #first we need base name of video for the output file name
    video_outname = vid + "_output.avi"
    for cam in videos[vid]:
        print("\n\nStaring openpose processing of " + vid + "." + cam )
        if videos[vid][cam]["openpose"]["exitcode"] == 0:
            print("Already processed", videos[vid][cam]["openpose"]["when"] )
        else:
            try:
                # Log the time
                time_start = time.time()
                video = ' --video "' + videos[vid][cam]["fullpath"] + '"'
                if createoutputvideo:
                    video_out = ' --write_video "' + videos_out_openpose + '\\' + vid + cam + "_output.avi" + '"'
                else:
                    video_out = ""
                openposecommand = openposeapp + video + video_out + optstring
                print(openposecommand)
                exitcode = os.system(openposecommand)
                videos[vid][cam]["openpose"]["exitcode"] = exitcode
                videos[vid][cam]["openpose"]["optstring"] = optstring
                videos[vid][cam]["openpose"]["handdata"] = includeHands
                # Log the time again
                time_end = time.time()
                if (exitcode == 0):
                    videos[vid][cam]["index"] = count  #TODO - Use this 
                    count += 1
                    videos[vid][cam]["openpose"]["exitcode"] = 0
                    videos[vid][cam]["openpose"]["when"] = datetime.now().isoformat()
                    videos[vid][cam]["openpose"]["out"] = videos_out_openpose + '\\' + video_outname
                    print ("Done " + vid + cam)
                    print ("It took %d seconds for conversion." % (time_end-time_start))
                else:
                    videos[vid][cam]["openpose"]["exitcode"] = exitcode
                    videos[vid][cam]["openpose"]["when"] = datetime.now().isoformat()
                    print("OpenPose error. Exit code %d" % exitcode)
                    print ("Conversion failed after %d seconds." % (time_end-time_start))
            except Exception as e:
                print("Error: ", e)
                pass
    #after each new id we save the json data
    with open(videosjson, 'w') as outfile:
        json.dump(videos, outfile)
        print('videos.json updated')

        
#change the directory back
os.chdir(currdir)
# -

# ## 1.6 Gather the data into useable format.
#
# OpenPose has created one JSON file per frame of video. We want to group these up into bigger arrays. 
#
# This routine needs to know where to find the processed videos and what are the base names. These are listed in the `videos.json` file we created.

#retrieve the list of base names of processed videos.
with open(videosjson) as json_file:
    videos = json.load(json_file)

# First find out the height, width and frames per second for each video and add this to `videos.json`

# +
#make a note of some video properties

for vid in videos:
    for cam in videos[vid]:
        cap = cv2.VideoCapture(videos[vid][cam]["fullpath"]) # 0=camera
        if cap.isOpened(): 
            videos[vid][cam]["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            videos[vid][cam]["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            videos[vid][cam]["fps"] = int(cap.get(cv2.CAP_PROP_FPS))
            videos[vid][cam]["n_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
# -

#optional
#print these out to remind ourselves. 
for vid in videos:  
    print(vid, ":")
    for cam in videos[vid]:
        print(videos[vid][cam])

# ####  Extracting all the numeric data from the json files
#
# We loop through the list of names in `videos` and search for all json files associated with that name. We then extract all the coordinates and confidence scores for all identified people in each frame and store them in one big multidimensional padded array.
#
# ```
# 1st dimension - number of videos
# 2nd dimension - max number of cameras (often 1 but sometimes 2 or 3) 
# 3rd dimension - max nummber of frames
# 4th dimension - max number of people
# 5th dimension - number of values (per person) output by openpose
# ```
#
# For example, if we had the following videos 
#
# ```
# video1 - cam1 - 200 frames  - 3 people (max) 
# video1 - cam2 - 200 frames  - 3 people (max) 
# video2 - 203 frames  - 2 people (max) 
# video3 - 219 frames  - 4 people (max) 
# ```
#
# then we'd create a `3 x 2 x 219 x 4 x 75` array.
#
# First we see how big the dimensions of the array have to be. 
# And create an numpy array called `keypoints_array` big enough to hold all of this.
#
# As a sanity check we count the number of frames processed by openpose. Ought to be same as above.
#

# +
nvideos = len(videos)
maxcameras = 3
maxframes = 0
maxpeople = 15 #maximum people we might expect (large upper bound)
ncoords = 75 #the length of the array coming back from openpose x,y coords of each point plus cafs
hcoords = 63

for vid in videos:    
    for cam in videos[vid]:    
        #use glob to get all the individual json files.
        videoname = videos[vid][cam]["stemname"]
        alljson = glob.glob(videos_out_openpose + "\\" + videoname + "*.json")
        nframes = len(alljson)
        print("Video", vid, cam, "has {0} frames.".format(nframes))
        videos[vid][cam]["frames"] = nframes
        maxframes = max(maxframes,nframes)
    
    
keypoints_array = np.zeros([nvideos,maxcameras, maxframes,maxpeople,ncoords]) #big array to hold all the numbers

#also create arrays for hand data 
righthand_array = np.zeros([nvideos,maxcameras, maxframes,maxpeople,hcoords]) #big array to hold all the numbers
lefthand_array  = np.zeros([nvideos,maxcameras, maxframes,maxpeople,hcoords]) #big array to hold all the numbers

print("Initialise numpy array of size", keypoints_array.shape)
# -

# Now loop through all the videos copying the frame data into our big `keypoints_array` and also seeing how many people (max) are detected in each one. 

#update the json file in the video out directory
with open(videosjson, 'w') as outfile:
    json.dump(videos, outfile)


# +
npeople = np.zeros(maxframes)  #an array to track how many people detected per frame.
globalmaxpeople =  0
v = -1

for vid in videos: 
    v += 1  #index for this subject
    c = -1
    for cam in videos[vid]:
        c += 1  #index for this camera
        #use glob to get all the individual json files.
        alljson = glob.glob(videos_out_openpose + "\\" + videos[vid][cam]["stemname"] + "*.json") 
        i = 0
        for frame in alljson:
            with open(frame, "r") as read_file:
                data = json.load(read_file)
                j = 0
                for p in data["people"]:
                    keypoints_array[v,c,i,j,:]= p["pose_keypoints_2d"]
                    if includeHands:
                        righthand_array[v,c,i,j,:]= p["hand_right_keypoints_2d"] 
                        lefthand_array[v,c,i,j,:] = p["hand_left_keypoints_2d"] 
                    j += 1
                npeople[i] = j
                i += 1
        #end loop for this video
        people = int(max(npeople))
        print("Video", vid, cam, "has {0} people detected.".format(people))
        videos[vid][cam]["maxpeople"] = people
        videos[vid][cam]["v"] = v  #might be useful to have these indices available
        videos[vid][cam]["c"] = c
        #how many people did it contain? Is this biggest number so far?
        globalmaxpeople = max(globalmaxpeople, people)
        
    
#and just like that n videos have been reduced to a big block of people coords.
#we now truncate the array for the maximum number of people as the rest of it is all zeros

keypoints_array = np.delete(keypoints_array,np.s_[int(globalmaxpeople):],3)

if includeHands:
    righthand_array = np.delete(righthand_array,np.s_[int(globalmaxpeople):],3)
    lefthand_array  = np.delete(lefthand_array, np.s_[int(globalmaxpeople):],3)                                 

print("keypoints_array has size", keypoints_array.shape)
# -

# ## 1.7 Save the data!
#
# Saving the data at this stage so we don't have to repeat these steps again if we reorganise or reanalyse the data.
#
# We create a compressed NumPy array `allframedata.npz` containing the person location data for all the videos. 
#
# We also update the `videos.json` file with more info about the videos. 

# +
#update the json file in the video out directory
with open(videosjson, 'w') as outfile:
    json.dump(videos, outfile)

# in the time series folder we save the data file. 
#in a compressed format as it has a lot of empty values
np.savez_compressed(videos_out_timeseries + "\\" + settings["filenames"]["alldatanpz"], keypoints_array=keypoints_array)

if includeHands:
    np.savez_compressed(videos_out_timeseries + '\\' + settings["filenames"]["righthandnpz"], keypoints_array=righthand_array)
    np.savez_compressed(videos_out_timeseries + '\\' + settings["filenames"]["lefthandnpz"],  keypoints_array=lefthand_array)


# -

# #### That's it. 
#
# Now go onto [Step 2 - Organising the data](Step2.OrganiseData.ipynb)
