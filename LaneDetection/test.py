import cv2

image = cv2.imread("lane.jpg",cv2.IMREAD_COLOR)
cv2.imshow("TEST",image)
cv2.waitKey(0)
cv2.destroyAllWindows()