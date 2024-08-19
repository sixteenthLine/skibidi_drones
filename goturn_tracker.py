import cv2
import numpy as np
from rtmpCalculate import Trigger

class GOTURN_Tracker:

    def __init__(self, bbox, frame, cap) :
      self.cap = cap
      self.bbox = bbox
      self.frame = frame
    
    def tracking(self):
        tracker = cv2.TrackerGOTURN.create()
        tracker.init(self.frame, self.bbox)
        step = 0
        while True:
            ok, next_frame = self.cap.read()
            if not ok:
                print("End of the video stream")
                break
            ok, bbox = tracker.update(next_frame)
            if step == 10:
                center, radius = Trigger.calculate_trigger(next_frame, drone_height= 50, wind_speed= 5 , wind_dir= 0)
                step = 0
                continue
            else: step +=1
            print(bbox)
            # cv2.rectangle(next_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (255, 0, 0), 2)
            # cv2.circle(next_frame, center, radius, (0, 255, 0),  1, lineType=cv2.LINE_AA)
            # cv2.imshow("Tracking", next_frame)

        self.cap.release()
        cv2.destroyAllWindows()