import cv2
import numpy as np
import LaneUtils as utils

# Get lane curve's trend
def getLaneCurve(img):
  imgCopy = img.copy()

  ### STEP 1 : Finding Lane Boundaries
  imgThres = utils.threshold(img)

  ### STEP 2 : Warping Lane Image
  h, w, c = img.shape
  points = utils.valTrackbars()
  imgWarp = utils.warpImg(img, points, w, h)
  imgWarpPoints = utils.drawPoints(imgCopy, points)

  ### STEP 3 : Get Histogram
  # utils.getHistogram(imgWarp)
  # cv2.imwrite('Thres.jpg', imgThres)

  cv2.imshow('video', img)
  cv2.imshow('Thres', imgThres)
  cv2.imshow('warp', imgWarp)
  cv2.imshow('warp points', imgWarpPoints)
  return None

if __name__ == '__main__':
  cap = cv2.VideoCapture('test_video.mp4')

  initialTrackbarVals = [100, 100, 100, 100]
  utils.initializeTrackbars(initialTrackbarVals)

  testWidth = 480
  testHeight = 240
  frameCounter = 0

  while True:
    frameCounter += 1
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      frameCounter = 0

    success, frame = cap.read()
    if not success:
      break

    img = cv2.resize(frame, (testWidth,testHeight))
    getLaneCurve(img)

    if cv2.waitKey(1) == ord('q'):
      break

  cv2.destroyAllWindows()