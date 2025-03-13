# import cv2
# from matplotlib import pyplot as plt
# import numpy as np

# def enhancement(img, brightness=10, contrast=2.3):
#     image2 = cv2.addWeighted(img, contrast, np.zeros(img.shape, img.dtype), 0, brightness)
#     return image2

# def low_contrast_segmentation(img, threshold=30):
#     _, segmented_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), threshold, 255, cv2.THRESH_BINARY)
#     return segmented_img

# # Read the input colored image
# img = cv2.imread('nature-3151869_640.jpg')

# # Enhance the image
# enhanced_img = enhancement(img, brightness=3, contrast=0.8)

# # Segment the low contrast areas
# segmented_img = low_contrast_segmentation(img, threshold=30)

# # Calculate histograms for each color channel
# color = ('b', 'g', 'r')
# histr = []
# for i, col in enumerate(color):
#     histr.append(cv2.calcHist([img], [i], None, [256], [0, 256]))

# # Create a figure to display the images and plots
# plt.figure(figsize=(18, 6))

# # Display the original image
# plt.subplot(1, 3, 1)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
# plt.axis('on')

# # Display the enhanced image
# plt.subplot(1, 3, 2)
# plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
# plt.title('Enhanced Image')
# plt.axis('on')

# # Display the segmented image
# plt.subplot(1, 3, 3)
# plt.imshow(segmented_img, cmap='gray')
# plt.title('Segmented Low Contrast Area')
# plt.axis('on')

# # Create a new figure for the histograms
# plt.figure(figsize=(12, 6))

# # Plot the histograms for each color channel
# for i, col in enumerate(color):
#     plt.plot(histr[i], color=col)
# plt.title('Histogram')
# plt.xlabel('Pixel values')
# plt.ylabel('Frequency')

# # Show the combined plots
# plt.show()


#Import the necessary libraries 
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
