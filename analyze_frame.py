import cv2
import numpy as np


def main():

	frame = cv2.imread('data/screenshot_01.png')
	height, width, channels = frame.shape
	frame_crop = np.concatenate ( (np.concatenate( (frame[0:int(height/3), 0:int(width/3)], frame[0:int(height/3), -int(width/3):]),axis=1),
								   np.concatenate( (frame[-int(height/3):, 0:int(width/3)], frame[-int(height/3):, -int(width/3):]),axis=1)), axis=0)

	frame_gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
	ret, frame_thresh = cv2.threshold(frame_gray, 100, 255, cv2.THRESH_BINARY_INV)
	contours_list, hierarchy = cv2.findContours(frame_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	check = 0
	for contour in contours_list:
		x, y, w, h = cv2.boundingRect(contour)
		if (2*h<w<4*h) and (w*h>5000):
			check += 1
			frame_crop = cv2.drawContours(frame_crop, contour, -1, (0, 255, 0), 2)

	# print(check)


	# to_remove = []
	# for i, contour in enumerate(contours_list):
	# identify rect around each contour
	# 	x, y, w, h = cv2.boundingRect(contour)
	# 	if 2*h>w:
	# 		to_remove.append(contour)
	# print(to_remove)
	# contours_list_filtered = [cnt for cnt in contours_list if cnt not in to_remove]
	# print(f'total: {len(contours_list)}')
	# print(f'remove: {len(to_remove)}')
	# 	## if height of the contour between 40% and 80% of total height, identify it as a character
	# 	# if ((h > 0.4*height) and (h<0.8*height)):
	#
	#
	cv2.imshow('output_crop', frame_crop)
	key = cv2.waitKey(0)



	# order them based on x coordinate of b-boxes (cv2.boundingRect[0])
	# 'key=lambda ctr' means that ctr is iterating over every element of the list
	# cv2.boundingRect(ctr)[0] is the 0-th component (x coordinate) of the b-box around ctr
	# contours_list = sorted(contours_list, key=lambda ctr: cv2.boundingRect(ctr)[0])



	# print(f"Found {len(contours_list)} contours.")




if __name__ == '__main__':
	main()