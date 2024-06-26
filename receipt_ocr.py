import os 
import sys
import argparse
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from functions import orient_vertical, sharpen_edge, binarize, \
      find_receipt_bounding_box, find_tilt_angle, adjust_tilt, crop, \
          enhance_txt, find_images, process_receipt, write_to_csv
import code2flow
import pytesseract

# Catch the arguments given with calling the python-script
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", 
                    default="raw",
                    help="file or directory from which images will be read")

parser.add_argument("-l", "--lang", type=str,
                    default="deu",
                    help="language to use for OCR")
parser.add_argument("-out", "--output", type=str,
                    default="output",
                    help="output path for OCR recognized text. \
                         Could be a directory or a filename \
                         default path is 'output' directory)")
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2, 3],
                    default=1,
                    help="increase output verbosity")
options = parser.parse_args()


# Read the input file(s).
if not os.path.exists(options.input):
    filenames = 'raw'   # if input path is not available, use folder 'raw'
if os.path.isfile(options.input):
    filenames = [options.input] # if input path is a single file, use it
elif os.path.isdir(options.input): # if input is a folder,
    filenames = list(find_images(options.input)) # catch the filenames of the folder by this function
else:
    parser.error("input file not found: "+options.input)

# If input is a directory, make sure that the output dir exists.
# If input is a single file than use the output directory
if len(filenames) > 1 and not os.path.isdir(options.output):
    parser.error("input is a directory, but output directory not found or not a directory: "+options.output)


for filename in filenames:  # make enhancement and ocr for each file
    #print(verbosity)
    if options.verbosity > 0: print(f"start to process: {filename}")
    try:
        receipt = process_receipt(filename, options.lang, options.verbosity, options.output)
    except RuntimeError as timeout_error:
        sys.stderr.write('Skipping {}, as it took too long to process'.format(filename))
        continue


if options.verbosity > 0: print("Processing {} receipts from {}".format(len(filenames), options.input))

