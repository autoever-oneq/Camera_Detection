import cv2

dsize = (640, 480)  # (Width, Height)

class webcamModule:
  def __init__(self, size: tuple=dsize):
    self.cap = cv2.VideoCapture(1) # 외장카메라 ID index 값 필요
    if not cap.isOpened():
      print("Cannot open video device")
      return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])


  def captureImg(cap: cv2.VideoCapture, display: bool=False, size: tuple=dsize):
    success, img = cap.read()

    if success:
      img = cv2.resize(img, size)
      if display:
        cv2.imshow('IMG', img)

    return success, img

### Unit Test
if __name__ == '__main__':
  from LaneDetectionModule import laneDetectionModule

  #cap = webcamModule()
  cap = cv2.VideoCapture(f'video/camera9.mp4')
  laneModule = laneDetectionModule()

  while cap.isOpened():
    #success, img = cap.captureImg(cap, display=True)
    success, img = cap.read()
    
    if not success:
      print("Video Capture Failed")
      break

    img = cv2.resize(img, dsize=dsize)
    curveVal = laneModule.getLaneCurve(img, curLane=0, targetLane=0, display=1)
    print(curveVal)
    
    if cv2.waitKey(1) == ord('q'):
      break
  
  cap.release()
  cv2.destroyAllWindows()