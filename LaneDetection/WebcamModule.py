import cv2
from LaneDetectionModule import getLaneCurve

cap = cv2.VideoCapture(0)

def getImg(display=False, size=(480,240)):
  success, img = cap.read()

  if success:
    img = cv2.resize(img, size)
    if display:
      cv2.imshow('IMG', img)

  return success, img

### Unit Test
if __name__ == '__main__':
  while True:
    success, img = getImg(display=True)
    if not success:
      continue

    maxVal = 0.5
    curveVal = getLaneCurve(img, display=1)

    curveVal = maxVal if (curveVal > maxVal) else -maxVal if (curveVal < -maxVal) else curveVal
    
    # Motor의 물리적 차이 존재 -> 두 모터의 서로 다른 sensitivity
    # Band를 두어 0 근처에서 작은 값에 의한 트월킹 방지
    sensitivity = 1.3
    if curveVal > 0:
      sensitivity = 1.7
      curveVal = 0 if curveVal < 0.05 else curveVal
    else:
      curveVal = 0 if curveVal > -0.08 else curveVal
    
    print(curveVal)