################################################################################
# This file contains function for easy use of Math computation  			   #
# Python Version: 3															   #
# 												   							   #
################################################################################

# Standard Modules
import math

# 3rd Party Modules


# Local Modules


def slope(x1, y1, x2, y2):
	return ((y2-y1) / (x2 - x1))

def least_square_regession(points):
	sum_x = 0
	sum_y = 0
	sum_x_sq = 0
	sum_xy = 0

	for x, y in points:
		sum_x += x
		sum_y += y
		sum_x_sq += x * x
		sum_xy += x * y

	N = len(points)
	m = ((N * sum_xy) - (sum_x * sum_y)) / ((N * sum_x_sq) - (sum_x * sum_x))
	b = ((sum_y - m * sum_x) / N)

	return (m, b)

def line_intersection(m1, b1, m2, b2):
	# parallel line
	if(m1 == m2):
		return (sys.maxint, sys.maxint)

	x = (b2 - b1) / (m1 - m2)
	y = (m1 * x) + b1

	return (x, y)