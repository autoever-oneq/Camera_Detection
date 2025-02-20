import cv2
import numpy as np
import LaneUtils as utils
# import cv2.typing
# (parameter) img: cv2.typing.MatLike == np.typing.NDArray[np.uint8]

# Get lane curve's trend
class laneDetectionModule:
  def __init__(self):
    self.curveList = [np.int32(0)] * 5


  def getLaneCurve(self, img, laneDiff: np.int8=0, display: int=0):
    ### STEP 1 : Find Lane & Binarization
    imgThres = utils.threshold(img)

    ### STEP 2 : Warping Lane Image
    hT, wT, c = img.shape

    # Determine which region to warp according to the direction
    # Option 1 : Change ROI when targetLane =/= curLane
    points = np.float32([[wT*0.3,hT*0.3],[wT*0.7,hT*0.3],[wT*0.1,hT],[wT*0.9,hT]] if laneDiff == 0 \
                    else [[wT*0.2,hT*0.4],[wT*0.4,hT*0.4],[wT*0.3,hT],[wT*0.6,hT]] if laneDiff > 0 \
                    else [[wT*0.6,hT*0.4],[wT*0.8,hT*0.4],[wT*0.3,hT],[wT*0.6,hT]])
    # Option 2 : Not change ROI, but apply bias on curve
    #points = np.float32([[wT*0.3,hT*0.3],[wT*0.7,hT*0.3],[wT*0.1,hT],[wT*0.9,hT]])

    imgWarp = utils.warpImg(imgThres, points, wT, hT)
    imgWarpPoints = utils.drawPoints(img.copy(), points)

    ### STEP 3 : Calculate Gradient of Lane(= Intensity of Curve)
    # Get histogram that accumulates pixel in a column
    if display > 1:
      middlePoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.5, region=4)   # Center position for the current lane(bottom of image)
      curveAveragePoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.3, region=1)       # Average position of nearby roads
    else:
      middlePoint = utils.getHistogram(imgWarp, minPer=0.5, region=4)   # Center position for the current lane(bottom of image)
      curveAveragePoint = utils.getHistogram(imgWarp, minPer=0.3, region=1)       # Average position of nearby roads

    curveRaw = curveAveragePoint - middlePoint if not (np.isnan(middlePoint) or np.isnan(curveAveragePoint)) else 0 # Raw target curve(biased) intensity
    # When option 2 is applied, add bias
    # if laneDiff < 0:
    #   curveRaw -= 40
    # elif laneDiff > 0:
    #   curveRaw += 40

    ### STEP 4 : Smoothing Curve Using LPF Filter
    curve = utils.smoothingCurve(self.curveList, curveRaw)

    ### STEP 5 : Display
    if display > 0:
      imgInvWarp = utils.warpImg(imgWarp, points, wT, hT, inverse=True)
      #imgInvWarp[0 : hT//3, 0 : wT] = 0   # Masking the top of inv Image
      imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
      
      imgLaneColor = np.full_like(img, (0, 255, 0))
      imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
      
      imgResult = cv2.addWeighted(img, 1, imgLaneColor, 1, 0)
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

    ### STEP 6 : Normalization & Thresholding
    # Mapping [curveMin,curveMax] -> [quantizedMin,quantizedMax] ([-160,160] => [-40,40])
    thres, curveThres = np.float32(100.), np.float32(40.)
    curve = -curveThres if curve < -thres else curveThres if curve > thres else (curve * curveThres / thres)

    return curve

### Unit Test
if __name__ == '__main__':
  cap = cv2.VideoCapture('video/output.avi')

  dsize = (480, 240)  # (Width, Height)
  frameCounter = 0
  flag_run = 1
  fps = 56

  # DEBUG
  # initialTrackbarVals = [testWidth//2-120, testHeight//2, 35, testHeight]
  # utils.initializeTrackbars(initialTrackbarVals)

  laneModule = laneDetectionModule()
  while True:
    frameCounter += flag_run
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      frameCounter = 0

    if flag_run > 0:
      success, frame = cap.read()
      if not success:
        print("FAIL")
        break

    img = cv2.resize(frame, dsize=dsize)
    curve = laneModule.getLaneCurve(img, laneDiff=0, display=1)
    if frameCounter % 5 == 0:
      print(curve, laneModule.curveList)

    key = cv2.waitKey(1000 // fps)
    if key == ord('q'):
      break
    if key == ord('p'):
      flag_run ^= 1


  cap.release()
  cv2.destroyAllWindows()
