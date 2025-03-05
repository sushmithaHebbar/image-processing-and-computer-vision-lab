import numpy as np
import cv2
img=cv2.imread("nature-3151869_640.jpg",cv2.IMREAD_COLOR)
width=400
height=200
re_image=cv2.resize(img,(width,height))
cv2.imwrite('re_iamge.jpg',re_image)
cv2.imshow("re-image",re_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
img.shape