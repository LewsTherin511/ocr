import cv2
import numpy as np


def main():

	cap = cv2.VideoCapture('data/top-100-shots-rallies-2018-atp-season.mp4')
	while True:
		ret, frame = cap.read()
		key = cv2.waitKey(1)
		if (not ret) or key == ord('q'):
			break

		height, width, channels = frame.shape
		frame_crop = np.concatenate ( (np.concatenate( (frame[0:int(height/3), 0:int(width/3)], frame[0:int(height/3), -int(width/3):]),axis=1),
									   np.concatenate( (frame[-int(height/3):, 0:int(width/3)], frame[-int(height/3):, -int(width/3):]),axis=1)), axis=0)

		frame_gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
		frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
		ret, frame_thresh = cv2.threshold(frame_gray, 20, 255, cv2.THRESH_BINARY_INV)
		contours_list, hierarchy = cv2.findContours(frame_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		for contour in contours_list:
			x, y, w, h = cv2.boundingRect(contour)
			if (2*h<w<4*h) and (8000<w*h<60000):
				frame_crop = cv2.drawContours(frame_crop, contour, -1, (0, 255, 0), 2)





		cv2.imshow('output_crop', frame_crop)








if __name__ == '__main__':
	main()


# to_remove = []
# for i, contour in enumerate(contours_list):
## identify rect around each contour
# x, y, w, h = cv2.boundingRect(contour)
# if 2*h>w:
# 	to_remove.append(contour)
# contours_list_filtered = [cnt for cnt in contours_list if cnt not in to_remove]
# print(f'total: {len(contours_list)}')
# print(f'remove: {len(to_remove)}')
# print(len(to_remove))
## if height of the contour between 40% and 80% of total height, identify it as a character
# if ((h > 0.4*height) and (h<0.8*height)):
