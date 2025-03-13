import cv2
import matplotlib.pyplot as plt
import numpy as np

def enhancement(img, brightness=10, contrast=2.3):
    image2 = cv2.addWeighted(img, contrast, np.zeros(img.shape, img.dtype), 0, brightness)
    cv2.imshow("Enhanced image", image2)
    cv2.waitKey(0)

def histogram(img):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(img, (32, 32))
    plt.hist(resized_img.ravel(), 256, [0, 255])
    plt.title('Histogram')
    plt.xlabel("Pixel values")
    plt.ylabel("Number of pixels")
    
    print(grey_img)
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([resized_img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()

if __name__ == "__main__":
    img = cv2.imread("nature-3151869_640.jpg")
    enhancement(img, 3, 0.8)
    histogram(img)

















# import cv2
# from matplotlib import pyplot as plt
# import numpy as np

# def enhancement(img, brightness=10, contrast=2.3):
#     image2 = cv2.addWeighted(img, contrast, np.zeros(img.shape, img.dtype), 0, brightness)
#     return image2

# def low_contrast_segmentation(img, threshold=30):
#     _, segmented_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
#     return segmented_img

# # Read the input image
# img = cv2.imread('nature-3151869_640.jpg', 0)

# # Enhance the image
# enhanced_img = enhancement(img, brightness=3, contrast=0.8)

# # Segment the low contrast areas
# segmented_img = low_contrast_segmentation(img, threshold=30)

# # Find the frequency of pixels in range 0-255
# histr = cv2.calcHist([img], [0], None, [256], [0, 256])

# # Create a figure to display the images and plot
# plt.figure(figsize=(18, 6))

# # Display the original image
# plt.subplot(1, 3, 1)
# plt.imshow(img, cmap='gray')
# plt.title('Original Image')
# plt.axis('off')

# # Display the enhanced image
# plt.subplot(1, 3, 2)
# plt.imshow(enhanced_img, cmap='gray')
# plt.title('Enhanced Image')
# plt.axis('off')

# # Display the segmented image
# plt.subplot(1, 3, 3)
# plt.imshow(segmented_img, cmap='gray')
# plt.title('Segmented Low Contrast Areas')
# plt.axis('off')

# # Display the histogram plot in a new figure
# plt.figure(figsize=(12, 6))
# plt.plot(histr)
# plt.title('Histogram')
# plt.xlabel('Pixel values')
# plt.ylabel('Frequency')

# # Show the combined plots
# plt.show()
