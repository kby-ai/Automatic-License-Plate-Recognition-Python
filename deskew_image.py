import numpy as np
import cv2
import math
from math import *
from scipy import ndimage
import matplotlib.pyplot as plt

def assert_lines(lines):
    
    ''' 
        params [lines] - lines returned from cv2.HoughLinesP()
        
        return value [truth statement] - checking if lines are horizontal or not
                                            we need to avoid horizontal lines
    '''
    
    for x1, y1, x2, y2 in lines[0]:
        return (x2-x1 == 0 or y2-y1 == 0)


def detectPlates(image):
		minPlateW = 50
		minPlateH = 15

		# initialize the rectangular and square kernels to be applied to the image,
		# then initialize the list of license plate regions
		rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
		squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

		regions = []

		# convert the image to grayscale, and apply the blackhat operation
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

		# find regions in the image that are light
		light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKernel)
		light = cv2.threshold(light, 50, 255, cv2.THRESH_BINARY)[1]

		# compute the Scharr gradient representation of the blackhat image and scale the
		# resulting image into the range [0, 255]
		gradX = cv2.Sobel(blackhat,
			ddepth = cv2.CV_32F,
			dx = 1, dy = 0, ksize = -1)
		gradX = np.absolute(gradX)
		(minVal, maxVal) = (np.min(gradX), np.max(gradX))
		gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

		# blur the gradient representation, apply a closing operating, and threshold the
		# image using Otsu's method
		gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
		gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
		thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

		# perform a series of erosions and dilations on the image
		thresh = cv2.erode(thresh, None, iterations = 2)
		thresh = cv2.dilate(thresh, None, iterations = 2)

		# take the bitwise 'and' between the 'light' regions of the image, then perform
		# another series of erosions and dilations
		thresh = cv2.bitwise_and(thresh, thresh, mask = light)
		thresh = cv2.dilate(thresh, None, iterations = 2)
		thresh = cv2.erode(thresh, None, iterations = 1)

        # cv2.imwrite("kk2.jpg", thresh)

		# find contours in the thresholded image
		cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# loop over the contours
		for c in cnts:
			# grab the bounding box associated with the contour and compute the area and
			# aspect ratio
			(w, h) = cv2.boundingRect(c)[2:]
			aspectRatio = w / float(h)

			# calculate extent for additional filtering
			shapeArea = cv2.contourArea(c)
			boundingboxArea = w * h
			extent = shapeArea / float(boundingboxArea)
			extent = int(extent * 100) / 100

			# compute the rotated bounding box of the region
			rect = cv2.minAreaRect(c)
			box = cv2.boxPoints(rect)

			# ensure the aspect ratio, width, and height of the bounding box fall within
			# tolerable limits, then update the list of license plate regions
			if (aspectRatio > 3 and aspectRatio < 6) and h > minPlateH and w > minPlateW and extent > 0.50:
				print("box", box)
				regions.append(box)

		# return the list of license plate regions
		return regions

def skew_correct(image):
    
    ''' 
        params: [image] - rgb image containing ROI

        return value: [rotated] - rotated image with normalized tilt angle.
                                    in other words, deskewed image
    '''

    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    not_image = cv2.bitwise_not(gray)

    
    # blur the image to remove noise
    blur = cv2.GaussianBlur(not_image, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, 20, minLineLength=20, maxLineGap=5)
    if lines is None:
        return image

    ''' 
        this statement checks two cases:

            1. If there are no lines detected - if not, change function parameters 
                                from cv2.HoughLinesP(), look for shorter lines
            
            2. If the lines detected are horizontal, horizontal lines will have an angle of 0
    '''
    if lines is None or assert_lines(lines):
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=20, maxLineGap=5)
        if lines is None or assert_lines(lines):
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=20, maxLineGap=5)

    angles_ver = []
    angles_hor = []

    # loop to find angle of line
    for line in lines:
        for x1, y1, x2, y2 in line:
            if not (x2-x1 == 0 or y2-y1 == 0):
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                if math.fabs(angle) < 45:
                    angles_hor.append(angle)
                else:
                    angles_ver.append(angle)  
    

    if len(angles_hor) > 0:
        median_angle = np.median(angles_hor)
        rotated = ndimage.rotate(image, median_angle)
    
        shift = math.tan(median_angle * np.pi/180) * image.shape[1]    

        input_pts = np.float32([[0,0], [image.shape[1]-1,0], [0,image.shape[0]-1]])
        output_pts = np.float32([[0,0], [image.shape[1]-1, -shift], [0,image.shape[0]-1]])
 
        M= cv2.getAffineTransform(input_pts , output_pts)
 
        # Apply the affine transformation using cv2.warpAffine()
        rotated = cv2.warpAffine(image, M, (image.shape[1],image.shape[0]))
    else:
        rotated = image

    if len(angles_ver) > 0:
        median_angle = np.median(angles_ver)
        if math.fabs(median_angle) > 70:
            shift = image.shape[0] / math.tan(median_angle * np.pi/180)

            input_pts = np.float32([[0,0], [image.shape[1]-1,0], [0,image.shape[0]-1]])
            output_pts = np.float32([[shift,0], [image.shape[1]-1, 0], [0,image.shape[0]-1]])

            M= cv2.getAffineTransform(input_pts , output_pts)
    
            # Apply the affine transformation using cv2.warpAffine()
            rotated = cv2.warpAffine(rotated, M, (image.shape[1],image.shape[0]))
   
    return rotated       
