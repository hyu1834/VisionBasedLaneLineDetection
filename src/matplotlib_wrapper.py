################################################################################
# This file contains function for easy use of Matplotlib feature			   #
# Python Version: 3															   #
# Matplotlib Version: 2.1.1													   #
################################################################################

# Standard Modules
import math

# 3rd Party Modules
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Local Modules


def show_image(image):
	plt.imshow(image)
	plt.show()

def show_gray_image(image):
	plt.imshow(image, cmap='gray')
	plt.show()
