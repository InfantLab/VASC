# Video Actor Synchroncy and Causality (VASC)
# Caspar Addyman, Goldsmiths 2020
# functions to support 


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

def diffKeypoints(keypoints1,keypoints2,indices):
    """Function to find how far apart one set of points is from another.
    This is useful for seeing if we have same person labelled correctly
    from one frame to next. If any point goes out of frame (loc == 0)
    then we don't include that pair. 
    Args:
        keypoints1: 1st array of keypoints.
        keypoints2: 1st array of keypoints.
        indices: a set of indices to compare over.
    Returns:
        diff per index (if any i)
    """
    out = []
    for i in indices:
        if keypoints1[i]>0 and keypoints2[i]:
            out.append(keypoints1[i] - keypoints2[i])
        else:
            out.append(None)
    return out