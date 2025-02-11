import cv2
import numpy as np
import LaneUtils as utils

# Global Variable
curveList = np.zeros((0), dtype=int)  # Empty integer array

# Get lane curve's trend
def getLaneCurve(img, direction='straight', display=2):
  imgCopy = img.copy()
  imgResult = img.copy()

  ### STEP 1 : Finding Lane Boundaries
  imgThres = utils.threshold(img)

  ### STEP 2 : Warping Lane Image
  hT, wT, c = img.shape
  # Get points using trackbars
  points = utils.valTrackbars()
  imgWarp = utils.warpImg(imgThres, points, wT, hT)
  imgWarpPoints = utils.drawPoints(imgCopy, points)

  ### STEP 3 : Calculate Gradient of Lane(= Intensity of Curve)
  # Get histogram that accumulates pixel in a column
  if display > 1:
    middlePoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.3, region=3, direction=direction)   # Center position for the current lane(bottom of image)
    curveAveragePoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.6, direction=direction)       # Average position of nearby roads
  else:
    middlePoint = utils.getHistogram(imgWarp, minPer=0.3, region=3, direction=direction)   # Center position for the current lane(bottom of image)
    curveAveragePoint = utils.getHistogram(imgWarp, minPer=0.6, direction=direction)       # Average position of nearby roads
  curveRaw = curveAveragePoint - middlePoint    # Raw target curve(biased) intensity

  ### STEP 4 : Smoothing Curve Using LPF Filter
  curve = utils.smoothingCurve(curveList, curveRaw)

  ### STEP 5 : Display
  if display > 0:
    imgInvWarp = utils.warpImg(imgWarp, points, wT, hT, inverse=True)
    imgInvWarp[0 : hT//3, 0 : wT] = 0   # Masking the top of inv Image
    imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
    
    imgLaneColor = np.full_like(img, (0, 255, 0))
    imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
    
    imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
    midY = 450

    cv2.putText(imgResult, str(curve), (wT//2-80,85), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,0))
    cv2.line(imgResult, (wT//2,midY), (wT//2+(curve*3),midY), (255,0,255), 5)
    cv2.line(imgResult, (wT//2+(curve*3),midY-25), (wT//2+(curve*3),midY), (255,0,255), 5)

    for x in range(-30, 30):
      w = wT//20
      cv2.line(imgResult, (w*x+curve//50, midY-10), (w*x+curve//50, midY+10), (0,0,255), 2)
    
    if display > 1:
      imgStacked = utils.stackImage(0.7, ([img,imgWarpPoints,imgWarp],[imgHist,imgLaneColor,imgResult]))
      cv2.imshow('ImageStack', imgStacked)
    else:
      cv2.imshow('Result', imgResult)

  ### STEP 6 : Normalization
  # Mapping [(-inf,-th],(-th,th),[th,inf)] -> [-1,0,1]
  curveThres = 50
  curve = 1 if (curve >= curveThres) else -1 if (curve <= -curveThres) else 0

  return curve

### Unit Test
if __name__ == '__main__':
  videoName = 'video_test2.mp4'
  cap = cv2.VideoCapture(videoName)

  testWidth = 480
  testHeight = 240
  frameCounter = 0

  initialTrackbarVals = [0, 0, 0, testHeight]
  utils.initializeTrackbars(initialTrackbarVals)

  while True:
    frameCounter += 1
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      frameCounter = 0

    success, frame = cap.read()
    if not success:
      break

    img = cv2.resize(frame, (testWidth,testHeight))
    curve = getLaneCurve(img, display=1)
    print(curve)

    # Quit
    if cv2.waitKey(1) == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()