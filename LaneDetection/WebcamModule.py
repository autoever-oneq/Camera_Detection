import cv2

default_h, default_w = 480, 640
def webcamInit(size=(default_w,default_h)):
  cap = cv2.VideoCapture(1) # 외장카메라 ID index 값 필요
  if not cap.isOpened():
    print("Cannot open video device")
    return cap

  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, default_h)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, default_w)
  return cap

def captureImg(cap: cv2.VideoCapture, display: bool=False, size: tuple=(default_h,default_w)):
  success, img = cap.read()

  if success:
    img = cv2.resize(img, size)
    if display:
      cv2.imshow('IMG', img)

  return success, img

### Unit Test
if __name__ == '__main__':
  from LaneDetectionModule import getLaneCurve

  cap = webcamInit()
  while cap.isOpened():
    success, img = captureImg(cap, display=True)
    if not success:
      print("Video Capture Failed")
      continue

    curveVal = getLaneCurve(img, display=2)
    print(curveVal)
    
    if cv2.waitKey(1) == ord('q'):
      break
  
  cap.release()
  cv2.destroyAllWindows()