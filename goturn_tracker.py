import cv2
import numpy as np

class GOTURN_Tracker:

    def __init__(self, bbox, frame, cap) :
      self.cap = cap
      self.bbox = bbox
      self.frame = frame
    
    def tracking(self):
        tracker = cv2.TrackerGOTURN.create()
        tracker.init(self.frame, self.bbox)
        while True:
            ok, next_frame = self.cap.read()
            if not ok:
                print("End of the video stream")
                break
            ok, bbox = tracker.update(next_frame)
            print(bbox)
            # cv2.rectangle(next_frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.rectangle(next_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (255, 0, 0), 2)
            cv2.imshow("Tracking", next_frame)

        self.cap.release()
        cv2.destroyAllWindows()