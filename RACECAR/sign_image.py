import sys

sys.path.insert(0, "../library")

import cv2 as cv
import numpy as np
import racecar_utils as rc_utils
import skimage.transform

RED_HSV = ((1, 0, 0), (30, 255, 255))

Min_Area = 10

Image_Width = 640
Image_Height = 480

def crop(input_image, sides):
    red_contours = rc_utils.find_contours(input_image, RED_HSV[0], RED_HSV[1])
    red_contour = rc_utils.get_largest_contour(red_contours, Min_Area)

    contour_center = rc_utils.get_contour_center(red_contour)

    if red_contour is None:
        return

    else:
        if contour_center[0] - sides > 0 and contour_center[0] + sides < Image_Height and contour_center[1] - sides > 0 and contour_center[1] + sides < Image_Width:
            cropped = rc_utils.crop(input_image, (contour_center[0] - sides, contour_center[1] - sides), (contour_center[0] + sides, contour_center[1] + sides))
            return skimage.transform.resize(cropped, (32, 32), mode='constant')
        if contour_center[0] - sides < 0:
            height_start = 0
        elif contour_center[0] + sides >= Image_Height:
            height_end = Image_Height - 1
        if contour_center[1] - sides < 0:
            width_start = 0
        elif contour_center + sides >= Image_Width:
            width_end = Image_Width - 1
    
        cropped = rc_utils.crop(input_image, (height_start, width_start), (height_end, width_end))
        return skimage.transform.resize(cropped, (32, 32), mode='constant')
    