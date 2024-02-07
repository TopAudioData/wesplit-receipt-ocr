import os 
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import code2flow
import pytesseract


def orient_vertical(img):
    width = img.shape[1]
    height = img.shape[0]
    if width > height:
        rotated = imutils.rotate_bound(img, angle=270)
    else:
        rotated = img.copy()

    return rotated


def sharpen_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    dilated = cv2.dilate(blurred, rectKernel, iterations=2)
    edged = cv2.Canny(dilated, 75, 200, apertureSize=3)
    return edged


def binarize(img, threshold):
    threshold = np.mean(img)
    thresh, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, rectKernel, iterations=2)
    return dilated


def find_receipt_bounding_box(binary, img):
    global rect

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    largest_cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_cnt)
    box = np.intp(cv2.boxPoints(rect))
    boxed = cv2.drawContours(img.copy(), [box], 0, (255, 255, 255), 20)
    return boxed, largest_cnt


def find_tilt_angle(largest_contour):
    angle = rect[2]  # Find the angle of vertical line
    print("Angle_0 = ", round(angle, 1))
    if angle < -45:
        angle += 90
        print("Angle_1:", round(angle, 1))
    else:
        uniform_angle = abs(angle)
    print("Uniform angle = ", round(uniform_angle, 1))
    return rect, uniform_angle


def adjust_tilt(img, angle):
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
    w = img.shape[1]
    h = img.shape[0]
    w1 = int(w*0.05)
    w2 = int(w*0.95)
    h1 = int(h*0.05)
    h2 = int(h*0.95)
    ROI = img[h1:h2, w1:w2]  # 95% of center of the image
    threshold = np.mean(ROI) * 0.98  # % of average brightness

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    edged = 255 - cv2.Canny(blurred, 100, 150, apertureSize=7)

    thresh, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

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


def process_receipt(filename, lang, verbosity, output_path):
    # Read raw image
    raw_img = cv2.imread(filename)

    # View raw image
    raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    if verbosity > 0:
        plt.imshow(raw_rgb)
        plt.show()

    # Rotate
    rotated = orient_vertical(raw_img)

    # Detect edge
    edged = sharpen_edge(rotated)
    binary = binarize(edged, 100)
    boxed, largest_cnt = find_receipt_bounding_box(binary, rotated)
    boxed_rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)

    # Adjust tilt
    rect, angle = find_tilt_angle(largest_cnt)
    tilted, delta = adjust_tilt(boxed, angle)
    print(f"{round(delta,2)} degree adjusted towards right.")

    # Crop
    cropped = crop(tilted, largest_cnt)

    # Enhance txt
    enhanced = enhance_txt(cropped)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    enhanced_path = 'preprocessed/enhanced.jpg'
    plt.imsave(enhanced_path, enhanced_rgb)

    # Run OCR
    options_all = f"--psm 1 --oem 1"
    txt = pytesseract.image_to_string(enhanced_path, lang=lang, config=options_all)

    # Save output txt
    write_to_csv(txt, output_path)
    txt_path = 'output/enhanced.txt'
    with open(txt_path, 'w') as f:
        f.write(txt)
        f.close()

    # Generate and save flowchart
    #code2flow.code2flow(['run.py', 'functions.py'], 'flowchart/out.png')
        
