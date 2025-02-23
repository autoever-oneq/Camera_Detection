import cv2
import numpy as np

### Debug Code For Lane Color Extraction
### Please change the name of the video
videoName = 'output1.avi'
cap = cv2.VideoCapture('video\\'+videoName)
frameWidth = 480
frameHeight = 240
frameCounter = 0

bar_init = [50, 120, 0, 130, 25, 100]

def empty(a):
  pass

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV",640,240)
cv2.createTrackbar("HUE Min","HSV",bar_init[0],179,empty)
cv2.createTrackbar("HUE Max","HSV",bar_init[1],179,empty)
cv2.createTrackbar("SAT Min","HSV",bar_init[2],255,empty)
cv2.createTrackbar("SAT Max","HSV",bar_init[3],255,empty)
cv2.createTrackbar("VALUE Min","HSV",bar_init[4],255,empty)
cv2.createTrackbar("VALUE Max","HSV",bar_init[5],255,empty)

while True:
  frameCounter += 1
  if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frameCounter = 0

  success, frame = cap.read()
  if not success:
    break
  
  img = cv2.resize(frame, (frameWidth,frameHeight))
  imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  h_min = cv2.getTrackbarPos("HUE Min","HSV")
  h_max = cv2.getTrackbarPos("HUE Max", "HSV")
  s_min = cv2.getTrackbarPos("SAT Min", "HSV")
  s_max = cv2.getTrackbarPos("SAT Max", "HSV")
  v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
  v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

  lower = np.array([h_min,s_min,v_min])
  upper = np.array([h_max,s_max,v_max])
  mask = cv2.inRange(imgHsv,lower,upper)
  result = cv2.bitwise_and(img, img, mask = mask)

  mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
  hStack = np.hstack([img,mask,result])
  
  cv2.imshow('Horizontal Stacking', hStack)
  if cv2.waitKey(1) == ord('q'):
    break
    
cap.release()
cv2.destroyAllWindows()