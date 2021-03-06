################################################################################
# This file contains main functions and lane line detection pipeline		   #
# Python Version: 3															   #
# Library Required:															   #
# 	OpenCV Version: 3.3														   #
# 	Matplotlib:																   #
#	Numpy: 1.13.3															   #
################################################################################

# Standard Modules
import sys
from enum import Enum

# 3rd Party Modules

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
import cv2

# Local Modules
import opencv_utils as opencv
import matplotlib_utils as matplot
import math_utils as math

class Input_Type(Enum):
	IMAGE = 0,
	VIDEO = 1

class Options():
	def __init__(self):
		self.input_type = Input_Type.IMAGE
		self.input_path = ""

	def __str__(self):
		return "Input Type: %s\nInput Path: %s\n"%(self.input_type, self.input_path)


def help():
	help_text = (
'''
NAME:
	LaneLineDetection - Module to perform lane line detection on videos or images

SYNOPSIS:
	LaneLineDetection [OPTIONS] <FILE>

OPTIONS:
	-v, --video
		Specify the input file is Video Stream

	-i, --image
		Specify the input file is Image (Default)
''')
	print(help_text)

def options_parser():
	options = Options()

	index = 1
	while(index < len(sys.argv)):
		if(sys.argv[index] == '-v' or sys.argv[index] == '--video'):
			options.input_type = Input_Type.VIDEO
		elif(sys.argv[index] == '-i' or sys.argv[index] == '--image'):
			options.input_type = Input_Type.IMAGE
		else:
			options.input_path = sys.argv[index]

		index += 1

	return options

def compute_full_lane_line(hough_lines, img_height):
	# First determine the slope of each line segment
	# and seperate line segment into left and right laneline
	left_lane_line_segments = []
	right_lane_line_segments = []
	left_lane_line_points = []
	right_lane_line_points = []

	for line in hough_lines:
		for x1, y1, x2, y2 in line:
			m = math.slope(x1, y1, x2, y2)
			b = y1 - m * x1
			# if(abs(m) > 0.4):
			if(m < 0):
				# left_lane_line_segments.append((x1, y1, x2, y2))
				left_lane_line_points.append((x1, y1))
				left_lane_line_points.append((x2, y2))
			else:
				# right_lane_line_segments.append((x1, y1, x2, y2))
				right_lane_line_points.append((x1, y1))
				right_lane_line_points.append((x2, y2))
					
	if(len(left_lane_line_points) > 0 and len(right_lane_line_points) > 0):
		# Compute least square fit line
		m_left, b_left = math.least_square_regession(left_lane_line_points)
		m_right, b_right = math.least_square_regession(right_lane_line_points)

		# Compute left and right least square line intersection
		intersect_x, intersect_y = math.line_intersection(m_left, b_left, m_right, b_right)
		intersect_x = int(round(intersect_x))
		intersect_y = int(round(intersect_y))

		# return left_lane_line_points
		return [(intersect_x, intersect_y, int(round((img_height - b_left) / m_left)), img_height),
				(intersect_x, intersect_y, int(round((img_height - b_right) / m_right)), img_height)]

	return []

################################################################################
# This function perform land line detection on given frame 					   #
# Detection pipeline 														   #
#	- Load frame 															   #
#	- Convert frame to grayscale 											   #
#	- Apply Gaussian Filter on interested region							   #
#	- Perform Canny Edge Detection on interested region  					   #
#		Note: We must apply Edge detection before region of interest 		   #
#			  Or else the edge of the region will be detected as edges 		   #
#			  Because the edge of region of interest is the strongest changes  #
#			  in the graident map 											   #
#	- Define region of interest 											   #
#	- Perform Hough Transformation 											   #
#	- Overlay line on original frame 										   #
################################################################################
def frame_lane_line_detection(frame):
	if(frame is None):
		return False
	# Variables for easy access
	frame_shape = frame.shape
	frame_width = frame_shape[1]
	frame_height = frame_shape[0]
	ignore_mask_color = 255
	rho = 1 # distance resolution in pixels of the Hough grid
	theta = np.pi/180 # angular resolution in radians of the Hough grid
	threshold = 1     # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 40 #minimum number of pixels making up a line
	max_line_gap = 20    # maximum gap in pixels between connectable line segments

	# Convert the frame to gray scale
	gray_frame = opencv.grayscale(frame)

	# Apply Gaussian Filter on the region of interest
	gaussian_frame = opencv.gaussian_blur(gray_frame, 5)

	# Apply Canny Edge Detection
	canny_edges_frame = opencv.canny(gaussian_frame, 60, 180)

	# Extract the region of interest
	vertices = np.array([[(50, frame_height), (470, 310), (490, 310), (frame_width - 50, frame_height)]], dtype = np.int32)
	region_of_interest_frame = opencv.region_of_interest(canny_edges_frame, vertices)

	# Apply Hough Transformation
	hough_lines = opencv.hough_transformation(region_of_interest_frame, 
											  rho, theta, threshold, 
											  min_line_length, max_line_gap)

	full_lane_lines = compute_full_lane_line(hough_lines, frame_height)
	land_line_mask = opencv.draw_lines(np.copy(frame)*0, full_lane_lines)
	# lane_line_mask = opencv.draw_polyline(np.copy(frame) * 0, full_lane_lines)
	
	lane_line_frame = opencv.weighted_img(frame, land_line_mask)


	# Here we will show the images for debugging purpose
	opencv.show_images([("Original Frame", frame), 
						# ("Gray Frame", gray_frame), 
						# ("Gaussian", gaussian_frame),
						("Canny Edges", canny_edges_frame),
						("Region", region_of_interest_frame),
						# ("Hough", hough_lines_frame),
						("Lane Line", lane_line_frame)
						], 0)

	return True

def process_video(video_path):
	print("Processing Video")
	video = opencv.open_video(video_path)

	while(video.isOpened()):
		ret, frame = video.read()
		if not frame_lane_line_detection(frame):
			break

	video.release()
	cv2.destroyAllWindows()			


def process_image(image_path):
	print("Processing Image")

	image = opencv.open_image(image_path)

	if not image is None:
		print("Image Resolution: %sx%s"%(image.shape[1], image.shape[0]))
		frame_lane_line_detection(image)

	cv2.destroyAllWindows()

def main():
	options = options_parser()
	print(options)
	if(options.input_path == ""):
		print("Error: No Input File Provided")
		return

	if(options.input_type == Input_Type.VIDEO):
		process_video(options.input_path)
	else:
		process_image(options.input_path)


if __name__ == '__main__':
	main()
