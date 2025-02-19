import cv2

dsize = (640, 480)  # (Width, Height)

class webcamModule:
  def __init__(self, size: tuple=dsize):
    self.cap = cv2.VideoCapture(f'nvarguscamerasrc ! 
                                video/x-raw(memory:NVMM), width=3264, height=1848, format=(string)NV12, framerate=(fraction)28/1 !
                                nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert !
                                video/x-raw, width={size[0]}, height={size[1]}, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)
    if not self.cap.isOpened():
      print("Cannot open video device")
      return
    
    #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])


  def captureImg(self, display: bool=False, size: tuple=dsize):
    success, img = self.cap.read()

    if success:
      img = cv2.resize(img, size)
      if display:
        cv2.imshow('IMG', img)

    return success, img


  #def __exit__(self):
  #  self.cap.release()
  #  cv2.destroyAllWindows()

### Unit Test
if __name__ == '__main__':
  from LaneDetectionModule import laneDetectionModule

  webcam = webcamModule(size=(816,462))
  laneDetection = laneDetectionModule()
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  #out = cv2.VideoWriter('output.avi', fourcc, 28.0, dsize)

  try:
    while webcam.cap.isOpened():
      success, img = webcam.captureImg(display=True)

      if not success:
        print("Video Capture Failed")
        break

      #out.write(img)

      img = cv2.resize(img, dsize=dsize)
      curveVal = laneDetection.getLaneCurve(img, curLane=0, targetLane=0)
      print(curveVal)

      if cv2.waitKey(1) == ord('q'):
        break

  finally:
    cv2.destroyAllWindows()
    webcam.cap.release()