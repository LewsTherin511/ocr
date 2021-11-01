import cv2.cv2 as cv2
import pytesseract
from matplotlib import pyplot as plt
import numpy as np


def main():

	custom_config = r'--oem 3 --psm 4'

	# frame = cv2.imread('data/white_on_black.jpg')
	# frame_black = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	# ret, frame_black = cv2.threshold(frame_black,128,255,cv2.THRESH_BINARY)
	#
	# frame_white = cv2.bitwise_not(frame_black)
	# frame_black = cv2.resize(frame_black, (640,480))
	# frame_white = cv2.resize(frame_white, (640,480))
	# cv2.imshow('mostly black', frame_black)
	# cv2.imshow('mostly white', frame_white)
	# result_pre = pytesseract.image_to_string(frame_white, config=custom_config)
	# print(result_pre)

	frame_black = cv2.imread('data/thresholded/frame_thresh_1260.png', 0)
	frame_white = cv2.imread('data/thresholded/frame_thresh_15.png', 0)
	for image in frame_black, frame_white:
		avg_color_per_row = np.average(image, axis=0)
		avg_color = np.average(avg_color_per_row, axis=0)
		color = 'black' if avg_color < 128 else 'white'
		print(f'Color: {color}')



	key = cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()