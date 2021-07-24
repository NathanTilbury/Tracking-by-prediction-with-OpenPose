#!/usr/bin/env python
'''
To Run:
python run_tracking_by_predictionV9.0.py --model=cmu --resize=656x368 --camera=0

or

python run_tracking_by_predictionV9.0.py --model=cmu --resize=656x368 --video=test
'''
import argparse
import time
import os
import cv2
import math
import numpy as np
import sys, socket, time
from os import system
from math import sqrt

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from collections import deque 
    
'''
Variables and lists to be used in the program.

Global Variables
'''
V_SERVER = "192.168.6.90"   # nil-command
V_SERVER_PORT = 6667        # port on which nil-controller listens
tracked_human = None        # ID of the user that is controlling the screen
need_user = True            # whether or not a user has control of the Kinect
'''
The above 4 lines are taken from code provided by Dr Adrian Clark with credit to Louis Clift*

Below is the sock information.
'''
sock = socket.socket (socket.AF_INET, socket.SOCK_DGRAM)
fps_time = 0

#This is the frame buffer size.
predict = 5
'''
Creates two empty double ended queues for use later in the program.
'''
person_tracked1 = deque()
predicted_positions = deque()

def send_command (cmd):
    "Send the command to nil-controller on its listening socket."
    global sock, V_SERVER, V_SERVER_PORT
    print(cmd)
    try:
        sock.sendto (cmd.strip (), (V_SERVER, V_SERVER_PORT))
    except:
        print("Socket information is incorrect")


def get_predicted_pos(img):
    '''
    This function takes the average distance change on both the x and y axis and then use this
    information to predict where the skeleton of interest will be in the next frame.
    '''
#   holds the x and y values
    xVals = []
    yVals = []
    
#   holds each change for the x and y values.
    xAvgDist = []
    yAvgDist = []
    
#   hold the last recorded value of the trackedpersons list.
    lastVals = person_tracked1[0]
    
#   adding all the x and y values from the tracked person coords list to seperate lists to calc the average of each.
    for i in person_tracked1:
        xVals.append(int(i[0]))
        yVals.append(int(i[1]))

#   finding the the change between each set of coordinates and adding them to a new list to caluclate the average.
#   distances for x
    for xAvg in xVals:
        for i in range(predict-1):
            xAvgDist.append(xVals[i+1] - xVals[i])
#   distances for y
    for yAvg in yVals:
        for i in range(predict-1):
            yAvgDist.append(yVals[i+1] - yVals[i])


#   finding the average change for the x and y values.
    xdiff = sum(xAvgDist)/len(xAvgDist)
    ydiff = sum(yAvgDist)/len(yAvgDist)

    
#   creating the predicted x and y values
    if lastVals[0] < xdiff:
        predictx = int(lastVals[0] + xdiff)
    else:
        predictx = int(lastVals[0] - xdiff)
        
    if lastVals[1] < ydiff:
        predicty = int(lastVals[1] + ydiff)
    else:
        predicty = int(lastVals[1] - ydiff)

#   drawing the predicted postion.
    cv2.circle(img, (predictx, predicty), 4, [0,0,255], -1, 8)


#   returning the predicted coordinates to the program.
    predicted_position = [predictx, predicty]
    return predicted_position
    

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_eclid_dist(LTP, locate):
    '''
    Finding the location with shortest distance from the predicted postion of the last frame
    and returning the coordinates of the closest.
    '''
    tracked = []
    highval = [sys.float_info.max, sys.float_info.max]

    #loops through the each idenified locations (neck positons)
    for i, loc in enumerate(locate):
        p1 = loc
        try:
            # print(predicted_positions[0])
            p2 = predicted_positions[0]
        except:
            # print("ltp is",LTP)
            p2 = LTP
        euclidian_dist = sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )
        if(euclidian_dist < highval[0]):
            highval[0] = euclidian_dist
            tracked = loc

    return tracked


def run_multi_corrector(tracked, LTP, locate):
    """
    Runs the detection corrector algorithum.
    Draws the Current tracked postion.
    """
    temp_tracked = get_eclid_dist(LTP, locate)

    if(temp_tracked != tracked):
        tracked = temp_tracked
    else:
        pass

    return tracked

def run_commands(humans, image):
    """
    Uses the tracked human to then process the commands
        ---------------------------------------------------------------------------  
        Motion commands:
        ---------------------------------------------------------------------------        
        Body locations for reference:
            {0,  "Nose"},
            {1,  "Neck"},
            {2,  "RShoulder"},
            {3,  "RElbow"},
            {4,  "RWrist"},
            {5,  "LShoulder"},
            {6,  "LElbow"},
            {7,  "LWrist"},
            {8,  "MidHip"},
            {9,  "RHip"},
            {10, "RKnee"},
            {11, "RAnkle"},
            {12, "LHip"},
            {13, "LKnee"},
            {14, "LAnkle"},
            {15, "REye"},
            {16, "LEye"},
            {17, "REar"},
            {18, "LEar"},
            {19, "LBigToe"},
            {20, "LSmallToe"},
            {21, "LHeel"},
            {22, "RBigToe"},
            {23, "RSmallToe"},
            {24, "RHeel"},
            {25, "Background"}

    """ 
    global need_user, tracked_human

    if need_user:
        """
        Search for a human who's is holding up their right hand.
        This person will then become the tracked person.
        """
        for i, SOI in enumerate(humans):
            head = SOI.body_parts[0]
            leftHand = SOI.body_parts[4]
            rightHand = SOI.body_parts[7]
            
            if leftHand.y < head.y and rightHand.y > head.y:
                tracked_human = i
                print("Control given to user", i)
                need_user = False
    else:
        #get the Skeleton of interest.
        SOI = humans[tracked_human]

        #list of key locations.
        rightHand = SOI.body_parts[4]
        rightelbow = SOI.body_parts[3].x
        leftHand = SOI.body_parts[7]
        leftelbow = SOI.body_parts[6].x
        head = SOI.body_parts[0]
        MID_LOWERr = SOI.body_parts[8].y
        MID_LOWERl = SOI.body_parts[11].y
        MID_UPPER = head.y
        GROUND = MID_LOWERl - 0.05 # 0.05
        
        #Command recognition statements.
        if(leftHand.y > GROUND and rightHand.y > GROUND):
            # neutral pose - no commands sent.
            print("No command")
            cv2.putText(image, "No commands sent", (1, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, [0,120,200], 2)

        elif(leftHand.y < MID_LOWERl and leftHand.y > MID_UPPER and rightHand.y > GROUND): #
            # Left hand is stretched to the side so turn left.
            send_command ("turn_left")
            print("turn left")
            cv2.putText(image, "Turn Left", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, [0,120,200], 5)

        elif(rightHand.y < MID_LOWERr and rightHand.y > MID_UPPER and leftHand.y > GROUND):
            # Right hand is stretched to the side so turn right.
            send_command ("turn_right")
            print("turn right")
            cv2.putText(image, "Turn Right", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, [0,120,200], 5)

        elif(rightHand.y < MID_LOWERr and rightHand.y > MID_UPPER and rightelbow < rightHand.x and leftHand.y < MID_LOWERl and leftHand.y > MID_UPPER and leftelbow > leftHand.x):
            # Both hands are on hips and elbows out
            send_command ("move_backward")
            print("move backward")
            cv2.putText(image, "Move Backward", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, [0,120,200], 4)

        elif(rightHand.y < MID_LOWERr and rightHand.y > MID_UPPER and leftHand.y < MID_LOWERl and leftHand.y > MID_UPPER):
            # Both hands are extended to each side and so we move forward.
            send_command ("move_forward")
            print("move forward")
            cv2.putText(image, "Move Forward", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, [0,120,200], 4)

        elif(rightHand.y > head.y and leftHand.y < head.y):
            # left hand above the head, to release the tracking.
            print("Tracked person released.")
            cv2.putText(image, "Tracked person released.", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, [0,120,200], 3)
            tracked_human = None
            need_user = True


def main(args):
    global tracked_human, need_user
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    
    #open camera stream.
    if args.video == "0":
        camera = cv2.VideoCapture(args.camera)
    else:
        camera = cv2.VideoCapture(args.video)

    if(camera.isOpened() == False):
        print("Error opening video stream")
    
    width = int(camera.get(3))
    height = int(camera.get(4))

#   Variable used to display the FPS - this is for testing purposes.
    frame = 0
    print("The first need user is: {0} \tThe ID of the user is {1}".format(need_user, tracked_human))
    while(camera.isOpened()):
        '''
        The two lists below must be refreshed on each loop.
        '''
        current_tracked = []
        locations = []
        ret, image= camera.read()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        person_count = len(humans)
        if(ret == True):  
            try:
                for i, human in enumerate(humans):
                    #getting the x and y postions of the neck of each human.
                    neck = human.body_parts[1]
                    x = neck.x*image.shape[1]
                    y = neck.y*image.shape[0]
                    #adding the x and y position of each detected location.
                    locations.append([x,y])
                run_commands(humans, image)
            except:
                print("Tracking Error")

            #checking that there are some people to track.

            if(need_user):
                pass
            else:

                if len(locations) == 1:
                    current_tracked = locations[0]
                    print("ID ", tracked_human)
                    print("List locations: ", 0)
                    if(len(person_tracked1) < 1):
                        last_track_postion = [0,0]
                    else:
                        last_track_postion = person_tracked1[0]

                elif len(locations) > 1:
                    """
                    Runs this there are more than on set or coordinates in the frame,
                    which means there is more than one person in the frame.
                    """
                    print("tracked human", tracked_human)
                    # print(locations)
                    if len(person_tracked1)> 1:
                        last_track_postion = person_tracked1[0]
                        current_tracked = run_multi_corrector(current_tracked, last_track_postion, locations)

                    else:
                        '''
                        default to [0,0] then rely on the distance calculation.
                        '''
                        current_tracked = locations[tracked_human]
                        last_track_postion = [0,0]

                    print("final current tracked ", current_tracked)
                else:
                    pass
                
                """
                Updating the tracked person number:
                """
                #Change the tracked Human number to match the postion in the frame.
                for i, data in enumerate(locations):
                    if(data == current_tracked):
                        tracked_human = i
                """
                Add the tracked location to the list of tracked postions.
                """
                print("current tracked is:", current_tracked)
                try:
                    person_tracked1.appendleft(current_tracked)
                    cv2.circle(image, (int(person_tracked1[0][0]), int(person_tracked1[0][1])), 10, [255,100,50], -1, 8)

                    if(len(person_tracked1) >= predict):
                        predicted_pos = get_predicted_pos(image)
                        predicted_positions.appendleft(predicted_pos)
                        if len(predicted_positions) > 30:
                            '''
                            limiting the size of the array of predicted postions to prevent memory issues during long runs
                            '''
                            predicted_positions.pop()
                        person_tracked1.pop()

                        
                    #print the number 1 at the tracked position.
                    cv2.putText(image,"(1)", (int(person_tracked1[0][0]), int(person_tracked1[0][1])),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 255), 2)
                except:
                    print("current tracked is: ", current_tracked)
        
        #---------------------------------------------------------------------------  
        #Writing all the labels and displaying the frame window.
        #---------------------------------------------------------------------------          
        cv2.putText(image,"people: %f" % (person_count), (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, "Current Tracked Location", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,100,50])
        cv2.putText(image, "Predicted Location", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255])
        cv2.imshow('TF-Pose-Estimation Pose Tracking Window', image)
        fps_time = time.time()

        #break command to stop the program when ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
        
    #releasing the camera and closing the windows.
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    '''
    parsers - these allow for users to pass parameters directly into the console run command.
    '''
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime gesture recognition and tracking')

    parser.add_argument('--video', type=str, default="0",
                        help='please enter the name of the video file and the file type such as .mp4 at the end. If this is not in the same directory please provide the full file path.')

    parser.add_argument('--camera', type=int, default=0,
                help='The number is the system camera number, if you only have one camera attached then leave it as default.')

    '''
    There are two way of running the program, via the video or camera commands - if no camera or video input is detected
    it will default back to the system webcam 0.
    '''
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0. I recommend 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    main(args)