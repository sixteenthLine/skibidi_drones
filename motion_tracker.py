import cv2
import numpy as np
from goturn_tracker import GOTURN_Tracker as GT

video_path = '/Users/dolomone/Desktop/video/video_batch_26.07-26.06/2024-07-26 11.49.50.mp4'
cap = cv2.VideoCapture(video_path)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
  diff = cv2.absdiff(frame1, frame2)
  gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (5,5), 0)
  _, tresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
  dilated = cv2.dilate(tresh, None, iterations = 3)
  contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    if cv2.contourArea(contour) > 500:
      print("Object has been detected.")
      GT_tracker = GT((x, y, w, h), frame2, cap)
      GT_tracker.tracking()

  frame1 = frame2
  ret, frame2 = cap.read()