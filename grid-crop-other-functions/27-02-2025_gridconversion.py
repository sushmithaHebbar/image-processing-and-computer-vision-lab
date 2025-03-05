import cv2
import numpy as np
import os
# Define the file path
file_path = 'nature-3151869_640.jpg'

# Check if the file exists at the specified location
if not os.path.isfile(file_path):
    print(f"Error: The file at {file_path} does not exist. Please check the file path.")
else:
    # Load the image
    img = cv2.imread(file_path)

    # Check if the image was loaded correctly
    if img is None:
        print(f"Error: Could not read the image at {file_path}. Check the file integrity.")
    else:
        # Get the dimensions of the image
        h, w, channels = img.shape

        # Calculate the midpoint to divide the image into quadrants
        half_h = h // 2
        half_w = w // 2

        # Split the image into four quadrants
        top_left = img[:half_h, :half_w]  # Top-left quadrant
        top_right = img[:half_h, half_w:]  # Top-right quadrant
        bottom_left = img[half_h:, :half_w]  # Bottom-left quadrant
        bottom_right = img[half_h:, half_w:]  # Bottom-right quadrant

        # Resize the quadrants 
        top_left_resized = cv2.resize(top_left, (200, 200))
        top_right_resized = cv2.resize(top_right, (200, 200))
        bottom_left_resized = cv2.resize(bottom_left, (200, 200))
        bottom_right_resized = cv2.resize(bottom_right, (200, 200))

        # Join the quadrants back together
        top_row = np.hstack((top_left_resized, top_right_resized))  # Combine the top row
        bottom_row = np.hstack((bottom_left_resized, bottom_right_resized))  # Combine the bottom row
        final_image = np.vstack((top_row, bottom_row))  # Combine the top and bottom rows

        # Display the quadrants and the final image
        cv2.imshow('Top Left', top_left_resized)
        cv2.imshow('Top Right', top_right_resized)
        cv2.imshow('Bottom Left', bottom_left_resized)
        cv2.imshow('Bottom Right', bottom_right_resized)
        cv2.imshow('Final Image', final_image)  # Display the reconstructed image

        # save the final image
        cv2.imwrite('final_image.jpg', final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
