import os 
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import code2flow
import pytesseract
import time

def orient_vertical(img):
    ''' Rotates an image to be vertically oriented if its width is greater than its height.
    Args: img (numpy.ndarray): Input image as a NumPy array.
    Returns: numpy.ndarray: Vertically oriented image (rotated if necessary). '''

    width = img.shape[1]
    height = img.shape[0]
    if width > height:
        rotated = imutils.rotate_bound(img, angle=90)
        if verbosity > 0: print("image rotated")
        if verbosity > 1: 
            # Display the result
            cv2.imshow('Vertically Oriented Image', rotated)
            cv2.waitKey(2000)
            cv2.destroyWindow('Vertically Oriented Image')
    else:
        rotated = img.copy()

    return rotated


def sharpen_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    dilated = cv2.dilate(blurred, rectKernel, iterations=2)
    edged = cv2.Canny(dilated, 75, 200, apertureSize=3)
    if verbosity > 2:
        plt.figure(figsize=(10, 10))

        plt.subplot(1, 4, 1)  # 1 row, 3 columns, 1st subplot
        plt.imshow(gray, cmap='gray')
        plt.title('Gray')

        plt.subplot(1, 4, 2)  # 1 row, 3 columns, 1st subplot
        plt.imshow(blurred, cmap='gray')
        plt.title('Blurred')

        plt.subplot(1, 4, 3)  # 1 row, 3 columns, 2nd subplot
        plt.imshow(dilated, cmap='gray')
        plt.title('Dilated')

        plt.subplot(1, 4, 4)  # 1 row, 3 columns, 3rd subplot
        plt.imshow(edged, cmap='gray')
        plt.title('Edged')

        plt.show(block=False)

        plt.pause(3)
        plt.close()

    return edged


def binarize(img, threshold):
    threshold = np.mean(img)
    thresh, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(binary, rectKernel, iterations=2)
    if verbosity > 2: 
        print(f"cv2.threshold is {thresh}, average threshold of the image was {threshold}." )
        # Display the result
        cv2.imshow('Dilated Edges', dilated)
        cv2.waitKey(2000)
        cv2.destroyWindow('Dilated Edges')
    return dilated


def find_receipt_bounding_box(binary, img):
    # Predefine rect to the edges of the image 
    # (rect is a rotated rectangle, represented by center_x and center_y, width and height, angle)
    height, width = img.shape[:2]
    rect = ((width/2, height/2), (width, height), 0)
    
    # Predefine largest_cnt
    largest_cnt = np.array([[[0, 0]], [[width, 0]], [[width, height]], [[0, height]]])
    
    # Find contours on the image
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Predefine boxed
    boxed = img
    if contours:
        # If contours are found, find the largest one
        largest_cnt = max(contours, key=cv2.contourArea)
        
        # Get the minimum area rectangle that encloses the largest contour
        rect = cv2.minAreaRect(largest_cnt)
        
        # Get the corners of the minimum area rectangle
        box = np.intp(cv2.boxPoints(rect))
        
        # Draw the minimum area rectangle on a copy of the original image
        boxed = cv2.drawContours(img.copy(), [box], 0, (255, 255, 255), 20)
    if verbosity > 2:
        # Display the result
        cv2.imshow('Boxed', boxed)
        cv2.waitKey(2000)
        cv2.destroyWindow('Boxed')          
    return boxed, largest_cnt, rect



def find_tilt_angle(rect):
    angle = rect[2]  # Find the angle of vertical line
    if verbosity > 1: print("Angle_0 = ", round(angle, 1))
    if angle < -45:
        angle += 90 # If the angle is less than -45, 90 is added to correct the angle
    elif angle > 45:
        angle -= 90 # If the angle is greater than 45, 90 is substracted to correct the angle

    uniform_angle = abs(angle) # If the angle is between -45 and 45, the absolute value of the angle is taken
    if verbosity > 1: 
        print("Angle_1:", round(angle, 1))
        print("Uniform angle = ", round(uniform_angle, 1))
    return uniform_angle


def adjust_tilt(img, angle):
    '''correct conditions and angels are to be verified'''
    if angle >= 5 and angle < 80:
        rotated_angle = 0
    elif angle < 5:
        rotated_angle = angle
    else:
        rotated_angle = 270+angle
    tilt_adjusted = imutils.rotate(img, rotated_angle)
    delta = 360-rotated_angle
    return tilt_adjusted, delta


def crop(img, largest_contour):
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = img[y:y+h, x:x+w]
    return cropped

# UNUSED FUNCTION??
def view_gray_imgs(img1, img2, title1='', title2=''):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img1)
    try:
        ax[0].set_title(f"{title1}\n{img1.shape}")
    except:
        ax[0].set_title(f"{title1}\n{img1.size}")
    ax[1].imshow(img2, cmap='gray')
    try:
        ax[1].set_title(f"{title2}\n{img2.shape}")
    except:
        ax[1].set_title(f"{title2}\n{img2.size}")
    plt.show()
    plt.close()


def enhance_txt(img):
    width = img.shape[1]
    height = img.shape[0]
    width_1 = int(width*0.05)
    width_2 = int(width*0.95)
    height_1 = int(height*0.05)
    height_2 = int(height*0.95)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    region_of_interest = gray[height_1:height_2, width_1:width_2]  # 95% of center of the image
    threshold = np.mean(region_of_interest) * 0.85  # % of average brightness

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    edged = 255 - cv2.Canny(blurred, 100, 150, apertureSize=7)

    thresh, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    #Display results
    if verbosity > 1:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(1, 5, 1) 
        plt.imshow(region_of_interest, cmap='gray')
        plt.title('Region of Interest')

        plt.subplot(1, 5, 2)  
        plt.imshow(gray, cmap='gray')
        plt.title('Gray')

        plt.subplot(1, 5, 3) 
        plt.imshow(blurred, cmap='gray')
        plt.title('Blurred')

        plt.subplot(1, 5, 4)  
        plt.imshow(edged, cmap='gray')
        plt.title('Edged')

        plt.subplot(1, 5, 5)  # 1 row, 3 columns, 3rd subplot
        plt.imshow(binary, cmap='gray')
        plt.title('Binary')

        #plt.show()
        plt.show(block=False)

        plt.pause(5)
        plt.close()

    return binary

def find_images(folder):
    for file in os.listdir(folder):
        full_path = os.path.join(folder, file)
        if os.path.isfile(full_path):
            # yield gives only one value (path) each time the function find_images is called
            yield full_path  

def write_to_csv(text_elements, output_path):
    """
    Function to write text elements to a CSV file.
    Args:
        text_elements (dict): A dictionary containing the text elements.
        csv_path (str): The path to the CSV file.
    Returns:
        None
    """
    # Create a DataFrame from the text elements
    df = pd.DataFrame(text_elements)
    # Join filename for csv
    csv_path = output_path + 'output.csv'
    # Write the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)

def write_to_txt(output_path, filename, txt):
    ''' Writes the OCR text as plain text into a file
    Args: 
    output_path (str): directory or filename to store the plain text
    filename (str): name of the current image file, if not other name is given in output_path the filename is taken with .txt to store the text
    txt (str): plain text from OCR
    Returns: None '''
    # if output_path is a directory, use it a output-directory
    if os.path.isdir(output_path) :
        basename = os.path.basename(filename)
        out_filename = os.path.join(output_path, basename + ".txt")
    else: # if output_path is not a directory, use it as filename for the text
        out_filename = output_path
    # write the text into the file
    with open(out_filename, "w") as fp:
            fp.write(txt)

    return
   


def process_receipt(filename, lang, verb, output_path):
    # make verbosity a global variable for all funtions
    global verbosity
    verbosity = verb
    # Read raw image
    raw_img = cv2.imread(filename)

    # View raw image in RGB
    raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    if verbosity > 1: 
        # Display the result
        cv2.imshow('raw RGB', raw_rgb)
        cv2.waitKey(2000)
        cv2.destroyWindow('raw RGB')

    # Rotate
    rotated = orient_vertical(raw_img)

    # Detect edge
    edged = sharpen_edge(rotated)
    binary = binarize(edged, 100)
    boxed, largest_cnt, rect = find_receipt_bounding_box(binary, rotated)
    boxed_rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
    if verbosity > 2: 
        # Display the result
        cv2.imshow('Boxed RGB', boxed_rgb)
        cv2.waitKey(2000)
        cv2.destroyWindow('Boxed RGB')

    # Adjust tilt
    angle = find_tilt_angle(rect)
    tilted, delta = adjust_tilt(boxed, angle)
    if verbosity > 1:
        print(f"{round(delta,2)} degree adjusted towards right.")

    # Crop
    cropped = crop(tilted, largest_cnt)
    if verbosity > 2: 
        # Display the result
        cv2.imshow('Cropped', cropped)
        cv2.waitKey(2000)
        cv2.destroyWindow('Cropped')

    # Enhance text on the image for better OCR
    enhanced = enhance_txt(cropped)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    #enhanced_path = 'preprocessed/enhanced.jpg'
    basename = os.path.basename(filename)
    enhanced_path = os.path.join('preprocessed/enh_' + basename)
    cv2.imwrite(enhanced_path, enhanced_rgb)
    #plt.imsave(enhanced_path, enhanced_rgb)

    # Run OCR
    #print(pytesseract.image_to_osd(raw_rgb))
    options_all = f"--psm 4 --oem 2"
    txt = pytesseract.image_to_string(enhanced_rgb, lang=lang, config=options_all)

    # Save output txt
    write_to_txt(output_path, filename, txt)
    ##write_to_csv(txt, output_path)
    txt_path = 'output/enhanced.txt'
    with open(txt_path, 'w') as f:
        f.write(txt)
        f.close()

    # save output image (if verbosity =2)
    if verbosity > 0:
        cv2.imwrite('output_image.jpg', enhanced_rgb)

    # inished the image and close all windows
    cv2.destroyAllWindows()
    if verbosity > 0: print(f"{filename} processed")

    # Generate and save flowchart
    #code2flow.code2flow(['run.py', 'functions.py'], 'flowchart/out.png')
        
