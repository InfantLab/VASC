# Video Actor Synchroncy and Causality (VASC)
# Caspar Addyman, Goldsmiths 2020
# support functions

import cv2
import numpy as np
import pandas as pd

# OpenPose has two main body models COCO with 18 points and BODY-25 with 25.
# The default (and better model) is BODY-25. Here we provide the labelling for
# all the points and their relationships to enable us to redraw the wireframes.
#info founnd in https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/pose/poseParameters.cpp


# what body part is that?
keypointsMapping = {0:  "Nose",
                    1:  "Neck",
                    2:  "RShoulder",
                    3:  "RElbow",
                    4:  "RWrist",
                    5:  "LShoulder",
                    6:  "LElbow",
                    7:  "LWrist",
                    8:  "MidHip",
                    9:  "RHip",
                    10: "RKnee",
                    11: "RAnkle",
                    12: "LHip",
                    13: "LKnee",
                    14: "LAnkle",
                    15: "REye",
                    16: "LEye",
                    17: "REar",
                    18: "LEar",
                    19: "LBigToe",
                    20: "LSmallToe",
                    21: "LHeel",
                    22: "RBigToe",
                    23: "RSmallToe",
                    24: "RHeel",
                    25: "Background"}

#what are coordinates of each point
POINT_COORDS = [[ 0, 1], [14,15], [22,23], [16,17], [18,19], [24,25], [26,27],
                [ 6, 7], [ 2, 3], [ 4, 5], [ 8, 9], [10,11], [12,13], [30,31],
                [32,33], [36,37], [34,35], [38,39], [20,21], [28,29], [40,41],
                [42,43], [44,45], [46,47], [48,49], [50,51]]

#Which pairs of points are connected in the wireframe?
POSE_PAIRS = [[ 1, 8], [ 1, 2], [ 1, 5], [ 2, 3], [ 3, 4], [ 5, 6], [ 6, 7],
              [ 8, 9], [ 9,10], [10,11], [ 8,12], [12,13], [13,14], [ 1, 0],
              [ 0,15], [15,17], [ 0,16], [16,18], [14,19],
              [19,20], [14,21], [11,22], [22,23], [11,24]]
              # [ 2,17], [ 5,18]
nPairs = len(POSE_PAIRS)

#What color shall we paint each point.
pointcolors =  [[255,     0,    85],
                [255,     0,     0],
                [255,    85,     0],
                [255,   170,     0],
                [255,   255,     0],
                [170,   255,     0],
                [ 85,   255,     0],
                [  0,   255,     0],
                [255,     0,     0],
                [  0,   255,    85],
                [  0,   255,   170],
                [  0,   255,   255],
                [  0,   170,   255],
                [  0,    85,   255],
                [  0,     0,   255],
                [255,     0,   170],
                [170,     0,   255],
                [255,     0,   255],
                [ 85,     0,   255],
                [  0,     0,   255],
                [  0,     0,   255],
                [  0,     0,   255],
                [  0,   255,   255],
                [  0,   255,   255],
                [  0,   255,   255]]


#will colour each person 0:9 a different colour to help us keep track
personcolors = [ [255,0,0], [0,255,0], [0,0,255],[0,255,255], [255,0,255], [255,255,0],[128,255,255], [255,128,255], [255,255,128],[0,0,0]]


#useful to have the indices of the x & y coords and the confidence scores
#recall that we get them in the order [x0,y0,c0,x1,y1,c1,x2,etc]
def xyc (coords):
    xs = [x * 3 for x in coords]
    ys = [x + 1 for x in xs]
    cs = [x + 2 for x in xs]
    return xs,ys,cs

nPoints =25
xs, ys, cs = xyc(list(range(nPoints)))
xys = xs + ys
xys.sort() # sort for clarity

#same for head, arms, legs
head = [0, 1, 15, 16, 17, 18]
headx, heady, headc = xyc(head)
headxys = headx + heady
headxys.sort()  # sort for clarity

arms =  [2, 3, 4, 5, 6, 7]
armsx, armsy, armsc = xyc(arms)
armsxys = armsx + armsy
armsxys.sort()

legs = [22, 23, 24, 11, 10, 9, 8, 12, 13, 14, 19, 20 ,21]
legsx, legsy, legsc = xyc(legs)
legsxys = legsx + legsy
legsxys.sort()

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds for conversion." % (time_end-time_start))
            break



def getkeypointcoord(keypointlist,index):
    x = index * 3
    #some shuffling around to get coords in right format for plotting.
    coords = keypointlist[x:x+2]
    coords = map(int,coords)
    coords = tuple(coords)
    return coords

def averagePoint(keypointList,indices):
    """Function to find the "centre of mass" for this person.
    It will take the average of the non-zero keypoints
    Args:
        keypointList: 1d array of keypoints.
        indices: a set of indices to average over.
    Returns:
        Average
    """
    tot = 0
    N = 0
    for i in indices:
        if keypointList[i]>0:
            tot += keypointList[i]
            N += 1
    if N > 0:
        return tot / N
    else:
        return 0  # or None?

def averageCoordinateTimeSeries(df,indices,videos = "All", people = "Both"):
    """Function to find the average of a set of coordinates from the person location time series.
    This helps track their centre of mass or the movements of the head, etc. 
    It will take the average of the non-zero keypoints
    Args:
        df: timeseries dataframe.
        video: which video set is this? Default "All" of them
        who: which person (infant or parent)? Default "Both"
        indices: if this is a list then we average over indices in list.
                 if it is a dictionary we average over indices in each item
    Returns:
        Average across each row for this subset of columns
    """
    
    if videos == "All": 
        #include all the videos
        videos = list(df.columns.levels[0])
        
    if people == "Both":
        #include parent and infant
        people = list(df.columns.levels[1])
    
    if isinstance(indices, list):
        #indices is a set of coordinates
        #so create a dictionary containing them 
        indices = {"avg": indices}
    
    #list of different averages do take for each person?
    idxs =indices.keys()
    
    col_index = pd.MultiIndex.from_product([videos,people,idxs], names=['video','person','avgs'])

    avgdf = pd.DataFrame(columns=col_index)
    #average per video per person per subset of indices
    for vid in videos:
        for pers in people:
            for subidx in indices:
                #dataframes make averaging nice and easy.
                #avg by vid by pers by subidx 
                avgdf[(vid,pers,subidx)] = df[(vid,pers)][indices[subidx]].mean(axis=1)
    
        
    return avgdf


def diffKeypoints(keypoints1,keypoints2,indices):
    """Function to find how far apart one set of points is from another.
    This is useful for seeing if we have same person labelled correctly
    from one frame to next. If any point goes out of frame (loc == 0)
    then we don't include that pair.
    Args:
        keypoints1: 1st array of keypoints.
        keypoints2: 1st array of keypoints.
        indices: a set of indices to compare over.
        aveage: do we avearge over kee
    Returns:
        diff per index (if any i)
    """
    out = []
    for i in indices:
        if keypoints1[i]>0 and keypoints2[i]>0:
            out.append(abs(keypoints1[i] - keypoints2[i]))
        else:
            out.append(np.nan)
    return out

def getframeimage(videopath,framenumber):
    cap = cv2.VideoCapture(videopath)
    cap.set(cv2.CAP_PROP_POS_FRAMES,framenumber) # Where frame_no is the frame you want
    ret, frame = cap.read() # Read the frame
    # When everything done, release the capture
    cap.release()
    if ret:
        return frame
    else:
        #TODO - what do we do now?
        return False


def drawPoints(frame, framekeypoints, people):
    for p in range(people):
        personkeypoints = framekeypoints[p,:]
        for i in range(nPoints):
            coords = getkeypointcoord(personkeypoints,i)
            if sum(coords) > 0:
                cv2.circle(frame,coords, 2, pointcolors[i], -1, cv2.LINE_AA)

def drawLines(frame, framekeypoints, people):
    for p in range(people):
        personkeypoints = framekeypoints[p,:]
        for i in range(nPairs):
            line = POSE_PAIRS[i]
            A = getkeypointcoord(personkeypoints,line[0])
            B = getkeypointcoord(personkeypoints,line[1])
            if sum(A) > 0 and sum(B) > 0:
                cv2.line(frame, (A[0], A[1]), (B[0], B[1]), pointcolors[i], 2, cv2.LINE_AA)

def drawBodyCG(frame, framekeypoints, people):
    """Function to draw centre of gravity for the points of a wireframe
    Args:
        frame: the image we are drawing to
        framekeypoints: the keypoints array.
        people: how many people are there?
    Returns:
        diff per index (if any i)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.0
    for p in range(people):
        personkeypoints = framekeypoints[p,:]
        avx = averagePoint(personkeypoints,xs)
        avy = averagePoint(personkeypoints,ys)

        cgloc  = tuple((int(avx),int(avy)))
        cv2.circle(frame,cgloc, 2, [0,0,0], -1, cv2.LINE_AA)

        txtloc = tuple((int(avx) - 50,int(avy) - 30))
        cv2.putText(frame, str(p), txtloc, font, fontScale, personcolors[p])

def swapSeries(keypoints_array,v,c,pers1,pers2,start,end):
    """helper function for swapping sections of time series. This is useful because openpose isn't
       consistent in labelling people so we need to rearrange things.
    Args:
        keypoints_array: all the data.
        v: which video? - specifies first dimension of array
        c: which camera? specifies second dimension of array
        pers1: which people to swap 1
        pers2: which people to swap 2
        start: where in time series do we start? (TODO can be blank - start at beginning)
        end: where in time series do we end? (TODO can be blank - to end)
    Returns:
        a rearranged keypoints_array
    """
    temp = np.copy(keypoints_array[v,c,start:end,pers1,:])  #temporary copy pers1 
    keypoints_array[v,c,start:end,pers1,:] = keypoints_array[v,c,start:end,pers2,:] #pers2 to pers 1
    keypoints_array[v,c,start:end,pers2,:] = temp
    
    return keypoints_array
        
def deleteSeries(keypoints_array,v,c,pers,start,end):
    """helper function for deleting time series that aren't parent or child.
    Args:
        keypoints_array: all the data.
        v: which video? - specifies first dimension of array
        c: which camera? - specifies second dimension of array
        pers: which person to delete
        start: where in time series do we start? (TODO can be blank - start at beginning)
        end: where in time series do we end? (TODO can be blank - to end)
    Returns:
        a rearranged keypoints_array
    """
    #simply set all these values to zero
    keypoints_array[v,c,start:end,pers,:] = 0
    #TODO - update the corresponding json file.
    return keypoints_array
    
    
def OLDfixpeopleSeries(keypoints_array,v,c,people, start, end):
    """ openpose isn't consistent in labelling people (person 1 could be labeled pers 2 in next frame).
    So we go through frame by frame and label people in new frame with index of nearest person from previous frame. This *should* fix things.
    Do do this we take difference in location of coordinates of person 1 in this frame with all people in next frame. Let's say this is Person 4. We swap person 4 and person 1 data in all subsequent frames effectively relabling them. Now do same for person two but only comparing to person 2 upwards in next frame. 
    Nearest means the closest averaged over set of coordinates There will often be missing points in a frame so we ought to account for that. 
    Args:
        keypoints_array: all the data.
        v: which video? - specifies first dimension of array
        c: which camera? - specifies first dimension of array
        people: list of all people we are comparing. 
        start: where in time series do we start? (TODO can be blank - start at beginning)
        end: where in time series do we end? (TODO can be blank - to end)
    Returns:
        a rearranged keypoints_array
    """
    #This may get messy!
    for f in range(start,end-1):
        print ("frame ", '{:4d}'.format(f))
        # we loop through all pairs of people 
        #person p from frame f and people p, p + 1, p + 2 from frame f + 1
        for p1 in range(len(people)-1): #first person from frame f
            r1 = people[p1] #what person is this?
            whosleft = range(p1,len(people))
            delta = {}
            for p2 in whosleft: #next person from frame f + 1
                r2 = people[p2] #what person is this?
                delta[p2] = diffKeypoints(keypoints_array[v,c,f,p1,:],keypoints_array[v,c,f+1,p2,:],xys)
                print(p1,p2,np.nanmean(delta[p2]),any(np.isfinite(delta[p2])))
            try:
                 #this next line tries finds the minimum & non-null value
                key_min = min(np.nanmean(delta.keys()), key=(lambda k: delta[k]))
            except: 
                #if that fails then default to same data series
                key_min = p1
            print (key_min)
            if key_min != p1:
                #swap the rest of series between these two 
                keypoints_array = swapSeries(keypoints_array,v,c,p1,key_min,f+1,end)
    return keypoints_array

def minimumswaps(deltas):
    """For 2x2 np.array of distances we want to 'diagonalise the minimums'. That is to say we want
    to find the smallest single entry and remove the row and column containing that then repeat the process. 
    The use case is for when openpose mislabels skeletons from one frame to the next. We take a matrix of 
    distances between the centroids and map each to its nearest from frame f to frame f+1 so we can swap them. 
    So what i really need is set of swaps. And for N x N matrix we only need N-1 swaps.
    Args:
        deltas: A 2dimensional np.array of minimal differences between pairs of elements
    Returns: 
        a set of ordered pairs to swap.
    """
    (rs,cs) = deltas.shape
    swaps = []
    for c in range(cs-1):
        #next finds finds minimal remaining value in the matrix, then finds it's row, col coordinates
        rc  = np.unravel_index(np.argmin(deltas, axis=None), deltas.shape)
        #now we set this row and col to infinity so we find next smallest remaining value
        deltas[:,rc[1]] = np.inf
        deltas[rc[0],:] = np.inf
        swaps.append(rc)
    return swaps    


def fixpeopleSeries(keypoints_array,v,c,people, start, end):
    """Openpose isn't consistent in labelling people (person 1 could be labeled pers 2 in next frame).
    So we go through frame by frame and label people in new frame with index of nearest person from previous frame. This *should* fix things.
    Do do this we take difference in location of coordinates of person 1 in this frame with all people in next frame. Let's say this is Person 4. We swap person 4 and person 1 data in all subsequent frames effectively relabling them. Now do same for person two but only comparing to person 2 upwards in next frame. 
    Nearest means the closest averaged over set of coordinates There will often be missing points in a frame so we ought to account for that. 
    Args:
        keypoints_array: all the data.
        v: which video? - specifies first dimension of array
        c: which camera? - specifies first dimension of array
        people: list of all people we are comparing. 
        start: where in time series do we start? (TODO can be blank - start at beginning)
        end: where in time series do we end? (TODO can be blank - to end)
    Returns:
        a rearranged keypoints_array
    """
    #This may get messy!
    for f in range(start,end-1):
        print ("frame ", '{:4d}'.format(f))
        # we loop through all unique pairs of people to distances between
        #person p from frame f and people p, p + 1, p + 2 from frame f + 1
        #we store these in upper diagonal of delta matrix
        deltas = np.array(np.full((len(people),len(people)),np.inf)) #all other values are infinite
        for p1 in range(len(people)): #first person from frame f
            for p2 in range(p1,len(people)): # whos left? next person from frame f + 1
                delta = diffKeypoints(keypoints_array[v,c,f,p1,:],keypoints_array[v,c,f+1,p2,:],xys)
                deltas[p1,p2] = np.nanmean(delta)
        #ok, now we know the pairwise distances
        #next we find the pairwise mimimums
        swaplist = minimumswaps(deltas)
        #finally we swap these time series from the next frame so each persons continues 
        for swap in swaplist:
            p1  = swap[0]
            p2 = swap[1]
            if p1 != p2:
                #swap the rest of series between these two 
                keypoints_array = swapSeries(keypoints_array,v,c,p1,p2,f+1,end)
    return keypoints_array
                
def swapCameras(videos, keypoints_array,vidx,cam1,cam2):
    """helper function for swapping secondary camera angle to main camera.
    Usually this means to 'camera1' but we make the routine more general. 
    We need to swap the json labels and the keypoints_array. We use the v & c indices stored in the json file.
    Args:
        keypoints_array: all the data.
        vidx: which video? - specifies first dimension of array
        cam1: which cam to swap 1
        cam2: which cam to swap 2
        start: where in time series do we start? (TODO can be blank - start at beginning)
        end: where in time series do we end? (TODO can be blank - to end)
    Returns:
        a rearranged keypoints_array
    """
    v1 = videos[vidx][cam1]["v"]
    c1 = videos[vidx][cam1]["c"]
    v2 = videos[vidx][cam2]["v"]
    c2 = videos[vidx][cam2]["c"]
    
    #swap the video info
    temp = videos[vidx][cam1]
    videos[vidx][cam1] = videos[vidx][cam2]
    videos[vidx][cam2] = temp
    
    #swap the data
    temp = np.copy(keypoints_array[v1,c1,:,:,:])  #temporary copy pers1 
    keypoints_array[v1,c1,:,:,:] = keypoints_array[v2,c2,:,:,:] #pers2 to pers 1
    keypoints_array[v2,c2,:,:,:] = temp
    
    
    #finally swap the indices of the data to reference their new positions
    #Yes, this seems weird but it is correct!
    videos[vidx][cam1]["v"] = v1
    videos[vidx][cam1]["c"] = c1
    videos[vidx][cam2]["v"] = v2
    videos[vidx][cam2]["c"] = c2
    
    return videos, keypoints_array