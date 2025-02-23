# Import NumPy for numerical operations.
import numpy as np 
# Import Matplotlib for displaying images.
import matplotlib.pyplot as plt  
# Import OpenCV for image processing.
import cv2  
 # Import Imutils for easy image transformations.
import imutils 


img = cv2.imread("cat.jpg")  # Load the image.
cv2.imshow("image", img)  # Display the original image.
cv2.waitKey(0)  # Wait for a key press.
cv2.destroyAllWindows()  # Close all windows.

#By using imutils we can rotate the image by any angle.Here given 60 degree.
rotate60=imutils.rotate(img,angle=60)
cv2.imshow("image",rotate60)
cv2.waitKey(0)
cv2.destroyAllWindows()

#By using imutils we can rotate the image by any angle.Here given 90 degree and displaying by using the command imshow.
rotate90=imutils.rotate(img,angle=90)
cv2.imshow("image",rotate90)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Ploting the image using matplotlib library .So that it will be displayed in the x and y axis.
plt.imshow(img)
plt.waitforbuttonpress()
plt.close('all')

#Rotating the image by 90 degree clockwise and counter clockwise using OpenCv library.
image=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

image=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE) #270DEG
cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Resizing the image by using the resize function in OpenCv library. width and height are given as 400 and 200 respectively.
width=400
height=200
re_image=cv2.resize(img,(width,height))
cv2.imwrite('re_iamge.jpg',re_image)
cv2.imshow("re-image",re_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Translating the image by using the warpAffine function in OpenCv library.
#Moving the image by 100 pixels in x direction and 50 pixels in y direction.
M=np.float32([[1,0,100],[0,1,50]])

#The command img.shape is used to get the width and height of the image.
width=img.shape[1]
height=img.shape[0]

#The warpAffine function is used to move the image by the given pixels.
dst=cv2.warpAffine(img,M,(width,height))
cv2.imshow("image",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# To convert the image to Matrix ,Here OpenCV loads images as NumPy arrays by default
matrix = np.array(img)  
# Print the matrix and Displaying the image shape.
print("Image as a matrix:\n", matrix)
print("Matrix shape (Height, Width, Channels):", matrix.shape)


#To convert the image to a grayscale image, we can use the cvtColor function in OpenCV.
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
print("Grayscale Matrix:\n", gray_img)  # Print matrix values
print("Grayscale Shape:", gray_img.shape)  # Check matrix shape
cv2.imshow("Grayscale Image", gray_img)  # Display image
cv2.waitKey(0)
cv2.destroyAllWindows()

#extracting the blue,green and red channels from the image.

blue_channel = img[:, :, 0]  # Extract blue channel
green_channel = img[:, :, 1]  # Extract green channel
red_channel = img[:, :, 2]  # Extract red channel

cv2.imshow("Blue Channel", blue_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()