import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils 

#1.reading the image
img=cv2.imread("nature-3151869_640.jpg",cv2.IMREAD_COLOR )
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
	
#----------------------------------------------------------------------------------

# #1 Rotating the image clockwise
# image=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
# cv2.imshow("image",image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #.Rotating the image counterclockwise
# image=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE) #270DEG
# cv2.imshow("image",image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #----------------------------------------------------------------------------------


# #2.translation of the image
# M=np.float32([[1,0,100],[0,1,50]]) #xaxis=100 (70units) and y axis=50
# width=img.shape[1]
# height=img.shape[0]
# dst=cv2.warpAffine(img,M,(width,height))
# cv2.imshow("image",dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #----------------------------------------------------------------------------------

# #3.scaling the image increasing and decreasing
# half = cv2.resize(img, (0, 0), fx = 0.1, fy = 0.1)
# bigger = cv2.resize(img, (1050, 1610))

# stretch_near = cv2.resize(img, (780, 540), 
#                interpolation = cv2.INTER_LINEAR)    #cv2.INTER_LINEAR: This is primarily used when zooming is required.
#                                                     # This is the default interpolation technique in OpenCV.


# Titles =["Original", "Half", "Bigger", "Interpolation Nearest"]
# images =[img, half, bigger, stretch_near]
# count = 4

# for i in range(count):
#     plt.subplot(2, 2, i + 1)
#     plt.title(Titles[i])
#     plt.imshow(images[i])

# plt.show()

#----------------------------------------------------------------------------------

# which rotates our image the specified number of angle degrees about the center of the image.
# loop over the rotation angles again, this time ensuring
# no part of the image is cut off
for angle in np.arange(0, 360, 15):
	rotated = imutils.rotate_bound(img, angle)
	cv2.imshow("Rotated (Correct)", rotated)
	cv2.waitKey(100) 
      


# which rotates our image the specified number of angle degrees about the center of the image.
# # then display the rotated image to screen.
for angle in np.arange(0, 360, 15):
    rotated = imutils.rotate(img, angle)
    cv2.imshow("Rotated (Problematic)", rotated)
    cv2.waitKey(100)