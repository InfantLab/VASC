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
# # Step 3: Analyse the data using statsmodels
#
# This script correlates and compares the timeseries of wireframes for the two figures in the video `["parent", "infant"]`
#
# We start by reloading the saved parquet file containing the numpy array of all [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) data from all pairs of individuals. 
#
#

# +
import sys
import os
import json
import math
import numpy as np       
import pandas as pd
import pyarrow.parquet as pq

import matplotlib.pyplot as plt
# %matplotlib inline

import logging
import ipywidgets as widgets  #let's us add buttons and sliders to this page.
from ipycanvas import Canvas

import vasc #a module of our own functions (found in vasc.py in this folder)

#turn on debugging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# %pdb on

# +
jupwd =  os.getcwd() + "\\"
projectpath = os.getcwd() + "\\..\\SpeakNSign\\"
# projectpath = os.getcwd() + "\\..\\lookit\\"

# locations of videos and output
videos_in = projectpath 
videos_out   = projectpath + "out"
#videos_out = "E:\\SpeakNSign\\out"
videos_out_openpose   = videos_out + "\\openpose"
videos_out_timeseries = videos_out + "\\timeseries"
videos_out_analyses   = videos_out + "\\analyses"
# -

# ### 3.1 Load the clean data as a DataFrame
#
# Reload the clean data file created in step 2. 

#retrieve the list of base names of processed videos.
try:
    with open(videos_out + '\\clean.json') as json_file:
        videos = json.load(json_file)
        print("Existing clean.json found..")
except:
    print("File clean.json not found.")

# +
print('reading parquet file:')
df = pq.read_table(videos_out_timeseries + '\\cleandata.parquet').to_pandas()

#sort the column names as this helps with indexing
df = df.sort_index(axis = 1)
print(df.head())
# -

# ## 3.2 Process the data 
#
# Next we set all 0 values to as missing value `np.nan` to enable interpolation.
# Then use numpy's built in `interpolate` method. 

# +
df = df.replace(0.0, np.nan)

#are we going to use all the data or a subset?
first = 501
last = 8500

df = df.truncate(before  = first, after = last)
# -

df = df.interpolate()

df.shape

print(df.head())

# ### 3.2.1 Mean movements
# We create a dictionary of the subsets of OpenPose coordinates we want to average and then call `mean` on the Pandas dataframe. e.g.
#
# ```
# meanpoints = {
#                "headx" : [0, 3, 45, 48, 51, 54],
#                "heady" : [1, 4, 46, 49, 52, 55],
#                "allx" :  [0, 3, 6, 9, ...],
#                "ally" :  [1, 4, 7, 10, ...]
#              }
# ```
#
# Then we call the `vasc.averageCoordinateTimeSeries` function to average across sets of coordinates. For a given set of videos and people. For example
#
# In:
# ```
# videos = "All"
# people = "Both"
# df2 = vasc.averageCoordinateTimeSeries(df,meanpoints,videos,people)
# df2.head
# ```
#
# Out:
# ```
# person      infant                                          parent   
# avgs         headx       heady          xs          ys       headx   
# 501     565.996600  369.840600  534.895615  398.482538  471.686200   
# 502     567.231800  369.887600  534.354198  398.706552  471.849400   
# 503     567.228600  370.159600  534.444328  398.678133  471.711600   
# 504     566.912600  369.857000  535.369536  398.551636  472.309400
# ...            ...         ...         ...         ...         ...
# ```
#

# +
meanpoints = {"head" : vasc.headxys,
              "arms" : vasc.armsxys,
              "all"  : vasc.xys}

vids = "All"
people = ["infant","parent"]

avgdf = vasc.averageCoordinateTimeSeries(df,meanpoints,vids,people)
# -

avgdf.head

# ### 3.2.2 Rolling window of movements
#
# One thing we'd like to know is if mothers move in response to infants. The raw time series are probably too noisy to tell us this so instead we look at the variance of movement over a short rolling window. 
#
# First we apply 2 second long (50 frame) rolling window to each coordinate of the body and use the stddev or variance function `std()` or `var()` . Then we take averages as in the step above. 
#
# However, this time we combine x and y coordinates as this is now a movement index.

# +
win = 50

movedf = df.rolling(window = win, min_periods = math.floor(win/2)).var()
# cut out the empty bits at the start
movedf = movedf.truncate(before  = first + 50, after = last)

vids = "All"
people = ["infant","parent"]
meanpoints = {"head" : vasc.headxys,
              "arms" : vasc.armsxys,
              "all"  : vasc.xys}

movedf = vasc.averageCoordinateTimeSeries(movedf,meanpoints,vids,people)
# -

movedf.head

# Let's create a widget to plot some graphs of the data

# +
vidlist = [] #used to fill dropdown options
for vid in videos:  
    vidlist.append(vid)
        
pickvid = widgets.Dropdown(
    options= vidlist,
    value= vidlist[0],
    description='Subject:'
)

features = []
for f in meanpoints:
    features.append(f)
    
pickfeature = widgets.Dropdown(
    options= features,
    value= features[0],
    description='Feature:'
)

linetypes = ["Mean point", "Moving Average"]
picktype = widgets.Dropdown(
    options= linetypes,
    value= linetypes[0],
    description='Line type:'
)

def pickvid_change(change):
    if change['name'] == 'value' and (change['new'] != change['old']):
        updateAll(True)
        
def pickfeature_change(change):
    if change['name'] == 'value' and (change['new'] != change['old']):
        updateAll(True)

def picktype_change(change):
    if change['name'] == 'value' and (change['new'] != change['old']):
        updateAll(True)
        
pickvid.observe(pickvid_change, 'value') 
pickfeature.observe(pickfeature_change, 'value') 
picktype.observe(picktype_change, 'value') 
button_update = widgets.Button(description="Redraw")
output = widgets.Output()


def drawGraphs(vid, feature, linetype):
    """Plot input signals"""
    plt.ion()
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].set_title('Infant')
    axarr[1].set_title('Parent')
    #axarr[0].set_xlabel('Frames')
    axarr[1].set_xlabel('Frames')

    who = ["infant","parent"]

    if linetype == linetypes[0]:
        usedf = avgdf
    else:
        usedf = movedf
        
    #to select a single column..
    infant = usedf[(vid, who[0], feature)].to_frame()
    parent = usedf[(vid, who[1], feature)].to_frame()
    n  = np.arange(usedf.shape[0])
    
    #selecting multiple columns slightly messier
    #infant = df3.loc[50:,(vid, part[0], ('head','arms', 'all'))]
    #parent = df3.loc[50:,(vid, part[1], ('head','arms', 'all'))]

    axarr[0].plot(n,infant)
    axarr[1].plot(n,parent, color='b')

    plt.show() 

def updateAll(forceUpdate = False):
    output.clear_output(wait = True)
    if forceUpdate:
        logging.info('forceUpdate')
        #slider.value = 0
        #slider.max = videos[pickvid.value][pickcam.value]["end"]
    with output:
        display(pickvid,pickfeature,picktype,button_update)  
        drawGraphs(pickvid.value,pickfeature.value,picktype.value)
    
#draw everything for first time
updateAll(True)
output
# -



# ### 3.3 Movement analysis
#
# First we run some simple correlations between the mother and infant.

infant = movedf[(vid, who[0], 'head')].to_frame()
infant.head
print(type(infant))

vid = "SS003"
movedf[(vid, who[0], 'head')].corr(movedf[(vid, who[1], 'head')]) 

who = ["infant","parent"]
parts = ["head","arms","all"]
results = pd.DataFrame(columns = ("corrHead","lagHead","corrArms","lagArms","corrAll","lagAll","DyadSynScore"),
                      index = videos)

#loop through colculate for each pair
for vid in videos:
    thisrow = []
    for part in parts:
        #to select a single column..
        pearson = movedf[(vid, who[0], part)].corr(movedf[(vid, who[1], part)])
 
        thisrow.append(pearson) #this is for correlation
        thisrow.append(None) #this is for maximum lag
    
    thisrow.append(None) #don't have DyadSynScore yet 
    results.loc[vid] = thisrow

results

excelpath = projectpath + "\\SS_CARE.xlsx"
videolist = pd.read_excel(excelpath)
videolist = videolist.set_index("subject")


videolist

# +

results["DyadSynScore"] = videolist["DyadSyn"]
results["MatSensScore"] = videolist["MatSens"]
# -

results

# +
vid = "SS095"
# Set window size to compute moving window synchrony.
r_window_size = 120
# Compute rolling window synchrony

#pearson = df3[(vid, who[0], part)].corr(df3[(vid, who[1], part)])
rolling_r = df3[(vid, who[0], parts[0])].rolling(window=500, center=True).corr(df3[(vid, who[1], parts[1])])

f,ax=plt.subplots(2,1,figsize=(14,6),sharex=True)
df3.loc[:,(vid, slice(None), parts[0])].plot(ax=ax[0])
ax[0].set(xlabel='Frame',ylabel='Movement index for parent and infant')
rolling_r.plot(ax=ax[1])
ax[1].set(xlabel='Frame',ylabel='Pearson r')
plt.suptitle("Movement rolling window correlation")
# -

rolling_r.mean()


# +
def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))

vid = "SS095"
d1 = df3[(vid, who[0], parts[0])]
d2 = df3[(vid, who[1], parts[0])]
seconds = 5
fps = 25
wholeads = who[0] + 'leads <> ' + who[1] + ' leads'
rs = [crosscorr(d1,d2, lag) for lag in range(-int(seconds*fps-1),int(seconds*fps))]
offset = np.ceil(len(rs)/2)-np.argmax(rs)
f,ax=plt.subplots(figsize=(14,3))
ax.plot(rs)
ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Offset = {offset} frames\n' + wholeads,ylim=[.0,1],xlim=[0,300], xlabel='Offset',ylabel='Pearson r')
ax.set_xticklabels([int(item-150) for item in ax.get_xticks()]);
plt.legend()
# -

# ## 3.4 Granger Causality
#
# The next thing to look at is if the movements of the infant predict the movements of the parent. This would suggest parent is responding to the infant. 
#

# +

https://towardsdatascience.com/granger-causality-and-vector-auto-regressive-model-for-time-series-forecasting-3226a64889a6

https://www.machinelearningplus.com/time-series/time-series-analysis-python/
    
https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9
    
