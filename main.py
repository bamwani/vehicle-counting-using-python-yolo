# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob
import math

files = glob.glob('output/*.png')
for f in files:
    os.remove(f)

from sort import *
tracker = Sort()
memory = {}
line1 = [(400,638), (1250, 788)]
line2 = [(120,838), (1120, 1080)]
counter1 = 0
counter2 = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input video")
ap.add_argument("-o", "--output", required=True,
    help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
    help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.35,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.25,
    help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"] , "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),
    dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

frameIndex = 0
# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    frame = adjust_gamma(frame, gamma=1.5)
    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (256, 256),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    center = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                
                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                center.append(int(centerY))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
    #print("idxs", idxs)
    #print("boxes", boxes[i][0])
    #print("boxes", boxes[i][1])
    
    dets = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x+w, y+h, confidences[i]])
            #print(confidences[i])
            #print(center[i])
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    tracks = tracker.update(dets)
    
    boxes = []
    indexIDs = []
    c = []
    
    previous = memory.copy()
    #print("centerx",centerX)
    #  print("centery",centerY)
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    if len(boxes) > 0:
        i = int(0)
        for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            # draw a bounding box rectangle and label on the image
            # color = [int(c) for c in COLORS[classIDs[i]]]
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            if indexIDs[i] in previous:
                previous_box = previous[indexIDs[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                cv2.line(frame, p0, p1, color, 3)
                
                #Speed Calculation
                y_pix_dist = int(y + (h-y)/2) - int(y2 + (h2-y2)/2)
                text_y = "{} y".format(y_pix_dist)
                x_pix_dist = int(x + (w-x)/2) - int(x2 + (w2-x2)/2)
                text_x = "{} x".format(x_pix_dist)
                #cv2.putText(frame, text_y, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 4)
                #cv2.putText(frame, text_x, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 4)
                final_pix_dist = math.sqrt((y_pix_dist*y_pix_dist)+(x_pix_dist*x_pix_dist))
                speed = np.round(1.5 * y_pix_dist,2)
                text_speed = "{} km/h".format(speed)
                cv2.putText(frame, text_speed, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                

                if intersect(p0, p1, line1[0], line1[1]):
                    counter1 += 1
#                if intersect(p0, p1, line2[0], line2[1]):
#                    counter2 += 1

            # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            #text = "{}".format(indexIDs[i])
            #cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            i += 1

    # draw line
    cv2.line(frame, line1[0], line1[1], (0, 255, 255), 4)
#    cv2.line(frame, line2[0], line2[1], (255, 0, 255), 2)

    note_text = "NOTE: Vehicle speeds are calibrated only at yellow line. speed of cars are more stable."
    cv2.putText(frame, note_text, (50,110), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
    # draw counter
    counter_text = "counter:{}".format(counter1)
    cv2.putText(frame, counter_text, (100,250), cv2.FONT_HERSHEY_DUPLEX, 4.0, (0, 0, 255), 7)
#    cv2.putText(frame, "ctr2",str(counter2), (100,400), cv2.FONT_HERSHEY_DUPLEX, 5.0, (255, 0, 255), 10)
    # counter += 1

    # saves image file
    #+cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 15,
            (frame.shape[1], frame.shape[0]), True)

        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * total))

    # write the output frame to disk
    writer.write(frame)

    # increase frame index
    frameIndex += 1

    #if frameIndex >= 4000: # limits the execution to the first 4000 frames
    #    print("[INFO] cleaning up...")
    #    writer.release()
    #    vs.release()
    #    exit()

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()