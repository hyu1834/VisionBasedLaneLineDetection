################################################################################
# This file contains function for easy use of openCV feature				   #
# Python Version: 3															   #
# OpenCV Version: 3.3														   #
# NumPy Version: 1.13.3
################################################################################

# Standard Modules
import math

# 3rd Party Modules
import cv2
import numpy as np

# Local Modules


def open_image(image_path, flag = cv2.IMREAD_COLOR):
	image = cv2.imread(image_path, flag)
	return image

def open_video(video_path):
	video = cv2.VideoCapture(video_path)
	return video

def show_image(window_name, frame, delay = 100):
	cv2.imshow(window_name, frame)
	cv2.waitKey(delay)

def show_images(frames, delay = 100):
	for window_name, frame in frames:
		cv2.imshow(window_name, frame)
	cv2.waitKey(delay)


################################################################################
#						Following functions are taken from 					   #
#			UDacity Self Driving Car Engineer Nenodegree Project 1			   #
################################################################################
def grayscale(img):
	"""Applies the Grayscale transform
	This will return an image with only one color channel
	but NOTE: to see the returned image as grayscale
	you should call plt.imshow(gray, cmap='gray')"""
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
def canny(img, low_threshold, high_threshold):
	"""Applies the Canny transform"""
	return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
	"""Applies a Gaussian Noise kernel"""
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
	"""
	Applies an image mask.
	
	Only keeps the region of the image defined by the polygon
	formed from `vertices`. The rest of the image is set to black.
	"""
	#defining a blank mask to start with
	mask = np.zeros_like(img)   
	
	#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
		
	#filling pixels inside the polygon defined by "vertices" with the fill color	
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	
	#returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
	"""
	`img` should be the output of a Canny transform.
		
	Returns an image with hough lines drawn.
	"""
	lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
							minLineLength=min_line_len, maxLineGap=max_line_gap)
	# print("Hough lines: ", lines)
	line_img = np.zeros(img.shape, dtype=np.uint8)
	draw_lines(line_img, lines)
	return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
	"""
	`img` is the output of the hough_lines(), An image with lines drawn on it.
	Should be a blank image (all black) with lines drawn on it.
	
	`initial_img` should be the image before any processing.
	
	The result image is computed as follows:
	
	initial_img * α + img * β + λ
	NOTE: initial_img and img must be the same shape!
	"""
	return cv2.addWeighted(initial_img, α, img, β, λ)

################################################################################

def hough_transformation(mask, rho, theta, threshold, min_line_len, max_line_gap):
	# Hough detection
	lines = cv2.HoughLinesP(mask, rho, theta, threshold, np.array([]),
							minLineLength=min_line_len, maxLineGap=max_line_gap)
	
	return lines


def draw_lines(img, lines, red = 255, green = 0, blue = 0, width = 10):
	for line in lines:
		for x1, y1, x2, y2 in line:
			cv.line(img, (x1, y1), (x2, y2), (blue, green, red), width)
# 	draw_lines(frame, lines)
# 	return frame

# def draw_lines(img, lines, red = 255, green = 0, blue = 0):
# 	for line in lines:
# 		print(line)
# 		for x1, y1, x2, y2 in line:
# 			cv2.line(img, (x1, y1), (x2, y2), (blue, green, red), 10)