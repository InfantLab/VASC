# Video Actor Synchroncy and Causality (VASC)
# Caspar Addyman, Goldsmiths 2020
# support functions

import cv2

# OpenPose has two main body models COCO with 18 points and BODY-25 with 25.
# The default (and better model) is BODY-25. Here we provide the labelling for
# all the points and their relationships to enable us to redraw the wireframes.
#info founnd in https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/pose/poseParameters.cpp

nPoints =25
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

xs, ys, cs = xyc(list(range(nPoints)))
#same for head
head = [0, 15, 16, 17, 18]
headx, heady, headc = xyc(head)



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
