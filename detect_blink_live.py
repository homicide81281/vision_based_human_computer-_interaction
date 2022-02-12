# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
#from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import pyautogui
import math

def midPoint(point1, point2):
    midPoint = (((point1[0] + point2[0]) / 2), ((point1[1] + point2[1]) / 2))
    return midPoint


def distance(point1, point2):
    distance = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
    return distance

def click():
    pom = pyautogui.position()
    pyautogui.click(pom)

def nothing(x):
    pass

def moveCursor(direction):
    if direction=="right":
        pyautogui.moveRel(20,0,0)
    elif direction=="left":
        pyautogui.moveRel(-20,0,0)
    elif direction=="up":
        pyautogui.moveRel(0,150,0)
    elif direction=="down":
        pyautogui.moveRel(0,-150,0)


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


cv2.namedWindow('vidyashree',cv2.WINDOW_NORMAL)
cv2.createTrackbar('thresh','vidyashree',0,10,nothing)

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(args["shape_predictor"])
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
#vs = FileVideoStream('vidya.mp4').start()
#fileStream = True
vs = VideoStream(src=1).start()
#vs = VideoStream(usePiCamera=True).start()q
fileStream = False
time.sleep(1.0)
lx=0
ly=0
rx=0
ry=0
ol = (0,0)
# loop over frames from the video stream
while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    if fileStream and not vs.more():
        break

    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then.
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use theqq
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        thresh = 4#cv2.getTrackbarPos('thresh', 'srujan')
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        diff = abs(leftEAR-rightEAR)
        diff = diff*100
        print diff, thresh
        if leftEAR>rightEAR:
            if diff>thresh:
                moveCursor("left")
                print("left")
        if rightEAR>leftEAR:
            if diff > thresh:
                moveCursor("right")
                print("right")

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        LM = cv2.moments(leftEyeHull)
        RM = cv2.moments(rightEyeHull)
        if LM['m00']>0:
            lx=int((LM["m10"]/LM["m00"]+ 1e-7)*1)
            ly = int((LM["m01"] / LM["m00"] + 1e-7) * 1)
        if RM['m00']>0:
            rx=int((RM["m10"]/RM["m00"]+ 1e-7)*1)
            ry = int((RM["m01"] / RM["m00"] + 1e-7)*1)

        mid = midPoint((lx,ly),(rx,ry))
        d = distance(mid,ol)
        print d
        if d>4:
            if mid[1]<ol[1]:
                moveCursor("down")
                print("Down")
            else:
                moveCursor("up")
                print("Up")
        ol=mid
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1

        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                click()

            # reset the eye frame counter
            COUNTER = 0

        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "lEAR: {:.2f}".format(leftEAR), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "rEAR: {:.2f}".format(rightEAR), (300, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()