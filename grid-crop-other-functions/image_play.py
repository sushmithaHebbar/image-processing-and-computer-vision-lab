import cv2.mat_wrapper
import numpy as np
import matplotlib.pyplot as plt
import cv2
#1.reading the image
img=cv2.imread("nature-3151869_640.jpg",cv2.IMREAD_COLOR )
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#----------------------------------------------------------------------------------

#2.displaying the image using ploting
plt.imshow(img)
plt.waitforbuttonpress()
plt.close('all')

#----------------------------------------------------------------------------------

#3.Rotating the image clockwise
image=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#----------------------------------------------------------------------------------

#4.Rotating the image counterclockwise
image=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE) #270DEG
cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#----------------------------------------------------------------------------------

#5.resizing the image nothing but zooming the image
width=400
height=200
re_image=cv2.resize(img,(width,height))
cv2.imwrite('re_iamge.jpg',re_image)
cv2.imshow("re-image",re_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#----------------------------------------------------------------------------------

#6.translation of the image
M=np.float32([[1,0,100],[0,1,50]])
width=img.shape[1]
height=img.shape[0]
dst=cv2.warpAffine(img,M,(width,height))
cv2.imshow("image",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

#----------------------------------------------------------------------------------

#7.scaling the image increasing and decreasing
half = cv2.resize(img, (0, 0), fx = 0.1, fy = 0.1)
bigger = cv2.resize(img, (1050, 1610))

stretch_near = cv2.resize(img, (780, 540), 
               interpolation = cv2.INTER_LINEAR)


Titles =["Original", "Half", "Bigger", "Interpolation Nearest"]
images =[img, half, bigger, stretch_near]
count = 4

for i in range(count):
    plt.subplot(2, 2, i + 1)
    plt.title(Titles[i])
    plt.imshow(images[i])

plt.show()

#----------------------------------------------------------------------------------

#8.croping the image
print(type(img)) 
# Shape of the image 
print("Shape of the image", img.shape) 
# [rows, columns] 
crop = img[50:180, 100:300]   
cv2.imshow('original', img) 
cv2.imshow('cropped', crop) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

#----------------------------------------------------------------------------------

#9.brightness and contrast the image
# define the a and b to control the contrast and brightness
a = 1.5 # Contrast control
b = 10 # Brightness control

# call convertScaleAbs function
adjusted = cv2.convertScaleAbs(img, alpha=a, beta=b)

# display the output image
cv2.imshow('adjusted', adjusted)
cv2.waitKey()
cv2.destroyAllWindows()

#----------------------------------------------------------------------------------

##10.bluring the image using different methods
blurImg = cv2.blur(img,(10,10)) 
cv2.imshow('blurred image',blurImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Averaging .this can also change the kernel size 
avging = cv2.blur(img,(10,10))
cv2.imshow('Averaging',avging)
cv2.waitKey(0)

# Gaussian Blurring
gausBlur = cv2.GaussianBlur(img, (5,5),0) 
cv2.imshow('Gaussian Blurring', gausBlur)
cv2.waitKey(0)

# Median blurring
medBlur = cv2.medianBlur(img,5)
cv2.imshow('Media Blurring', medBlur)
cv2.waitKey(0)

# Bilateral Filtering
bilFilter = cv2.bilateralFilter(img,9,75,75)
cv2.imshow('Bilateral Filtering', bilFilter)
cv2.waitKey(0)
cv2.destroyAllWindows()

#----------------------------------------------------------------------------------

#11.spliting the image using color 
b,g,r = cv2.split(img) 
#display the original image:
cv2.imshow("Original Image",img)
# Displaying Blue channel image 
# Blue colour is highlighted the most 
cv2.imshow("Model Blue Image", b) 
  
# Displaying Green channel image 
# Green colour is highlighted the most 
cv2.imshow("Model Green Image", g) 
  
# Displaying Red channel image 
# Red colour is highlighted the most 
cv2.imshow("Model Red Image", r) 
cv2.waitKey(0)

#----------------------------------------------------------------------------------
#Mereging the image
image_merge = cv2.merge([r, g, b]) 
cv2.imshow("RGB_Image", image_merge) 
cv2.waitKey(0)
