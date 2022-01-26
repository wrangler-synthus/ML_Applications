# import the necessary packages

import argparse

import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-V", "--video",default=0, help="path to video file")
ap.add_argument("-a", "--min-area", type=int, default=2100, help="minimum area")
args = vars(ap.parse_args())

# To run do python motion.py
# otherwise run as python motion.py -V video location..
# Run any video, just make sure that the default area and the width
# is set accordingly otherwise the fps would be very low and
# the video would run very slowly...

vs = cv2.VideoCapture(args["video"])
firstFrame = None

while True:
    _, frame = vs.read()
    # frame = frame if args.get("video", None) is None else frame[1]
    if frame is None:
        break
    frame = imutils.resize(frame, width=1200)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if firstFrame is None:
        firstFrame = gray
        continue
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < args["min_area"]:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("Feed", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
vs.release()
cv2.destroyAllWindows()
