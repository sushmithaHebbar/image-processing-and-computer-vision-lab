import numpy as np 

# Importing Matplotlib for displaying images and plots.
import matplotlib.pyplot as plt  

# Importing OpenCV, a library for working with images and videos.
import cv2 
 
# Reading an image from the given file
img = cv2.imread("cat.jpg")

# Displaying the image in a new window with the title "image"
cv2.imshow("image", img)  

# Waiting indefinitely until any key is pressed (this keeps the window open)
cv2.waitKey(0)  

# Closing all OpenCV windows after a key is pressed
cv2.destroyAllWindows()