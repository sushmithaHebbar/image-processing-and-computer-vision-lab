
# import os
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# # results = [] 
# # for dirpath, dirnames, filenames in os.walk("."):
# #     for filename in [f for f in filenames if f.endswith('.JPG')]: # to loop over all images you have on the directory
# #         img = cv2.imread("nature-3151869_640.jpg")
# #         avg_color_per_row = np.average(img, axis=0)
# #         avg_color = np.average(avg_color_per_row, axis=0)
# #         results.append(avg_color)
# # np_results = np.array(results) # to make results a np array
# # plt.hist(np_results)
# # plt.show() # to show the histogram

# # img = cv2.imread('nature-3151869_640.jpg')
# # avg_color_per_row = np.average(img, axis=0)
# # avg_color = np.average(avg_color_per_row, axis=0)
# # print sum(avg_color)/3
# img = plt.imread("nature-3151869_640.jpg")
# img = plt.imshow("nature-3151869_640.jpg")
# plt.hist(n_img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])


import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
  
# Load the image 
image = cv2.imread('nature-3151869_640.jpg') 
  
#Plot the original image 
plt.subplot(1, 2, 1) 
plt.title("Original") 
plt.imshow(image) 
  
# Adjust the brightness and contrast 
# Adjusts the brightness by adding 10 to each pixel value 
brightness = 10 
# Adjusts the contrast by scaling the pixel values by 2.3 
contrast = 2.3  
image2 = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness) 
  
#Save the image 
cv2.imwrite('modified_image.jpg', image2) 
#Plot the contrast image 
plt.subplot(1, 2, 2) 
plt.title("Brightness & contrast") 
plt.imshow(image2) 
plt.show()