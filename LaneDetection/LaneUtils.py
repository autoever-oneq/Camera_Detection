import cv2
import numpy as np

### ----- STEP 1 : Find Lane Area & Remove Backgrounds ----- ###
def threshold(img):
  # Convert BGR to HSV
  imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  # Define range of Blackcolor(Lane) in HSV
  lowerBlack = np.array([0, 0, 0])
  upperBlack = np.array([90, 50, 255])

  # Binaryization based on lane color
  maskBlack = cv2.inRange(imgHsv, lowerBlack, upperBlack)

  return maskBlack

### ----- STEP 2 : Warping Lane Image ----- ###
def warpImg(img, points, w, h):
  # Source(origin 4 pts) -> Destination(output 4 vertices)
  source = np.float32(points)
  destination = np.float32([[0,0], [w,0], [0,h], [w,h]])

  # Get Warp Matrix
  matrix = cv2.getPerspectiveTransform(source, destination)
  #inv_matrix = cv2.getPerspectiveTransform(destination, source)
  imgWarp = cv2.warpPerspective(img, matrix, (w,h))
  return imgWarp

def empty(a):
  pass

def initializeTrackbars(initialTrackbarVals, wT=480, hT=240):
  cv2.namedWindow("Trackbars")
  cv2.resizeWindow("Trackbars", 360, 240)
  cv2.createTrackbar("Width Top", "Trackbars", initialTrackbarVals[0], wT//2, empty)
  cv2.createTrackbar("Height Top", "Trackbars", initialTrackbarVals[1], hT, empty)
  cv2.createTrackbar("Width Bottom", "Trackbars", initialTrackbarVals[2], wT//2, empty)
  cv2.createTrackbar("Height Bottom", "Trackbars", initialTrackbarVals[3], hT, empty)

def valTrackbars(wT=480, hT=240):
  widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
  heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
  widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
  heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
  
  points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop), (widthBottom, heightBottom), (wT-widthBottom, heightBottom)])

  return points

def drawPoints(img, points):
  for x in range(4):
    cv2.circle(img, (int(points[x][0]),int(points[x][1])), 15, (0,0,255), cv2.FILLED)
  return img

### ----- STEP 3 : Get Histogram ----- ###
def getHistogram(img, minPer=0.2, display = False):
  histValues = np.sum(img,axis=0)
  #print(histValues)
  
  maxValue = np.max(histValues)
  minValue = minPer * maxValue

  indexArray = np.where(histValues >= minValue)
  basePoint = int(np.average(indexArray))

  if display:
    imgHist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for x, intensity in enumerate(histValues):
      cv2.line(imgHist, (x,img.shape[0]), (x,img.shape[0]-intensity//255), (255,0,255), 1)
      cv2.circle(imgHist, (basePoint, img.shape[0]), 20, (0,255,255))
    return basePoint, imgHist

  return basePoint