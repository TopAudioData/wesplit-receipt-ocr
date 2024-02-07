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

parser = argparse.ArgumentParser()
parser.add_argument("input", help="file or directory from which images will be read")
parser.add_argument("-lang", type=str,
                    default="deu",
                    help="language to use for OCR")
parser.add_argument("-out", "--output", type=str,
                    help="output directory for OCR recognized text (default is to 'output')")
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2],
                    default=1,
                    help="increase output verbosity")
options = parser.parse_args()

# Make sure that the output dir exists.
if options.output and not os.path.isdir(options.output):
    parser.error("output dir not found or not a dir: "+options.output)


# Read the input file(s).
if not os.path.exists(options.input):
    filenames = 'raw'
if os.path.isfile(options.input):
    filenames = [options.input]
elif os.path.isdir(options.input):
    filenames = list(find_images(options.input))
else:
    parser.error("input file not found: "+options.input)

for filename in filenames:
    try:
        receipt = process_receipt(filename, options.lang, options.verbosity, options.output)
    except RuntimeError as timeout_error:
        sys.stderr.write('Skipping {}, as it took too long to process'.format(filename))
        continue


if options.verbosity > 0:
    print("Processing {} receipts from {}".format(len(filenames), options.input))

