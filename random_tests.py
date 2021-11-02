import numpy as np
import cv2


def main():
	empty_BGR = np.zeros((10,10,3), dtype=np.uint8)

	empty_BGR[:,:,2] = 255

	# print(empty.shape)
	# print(type(empty[0][0][0]))
	# cv2.imwrite('red.png', empty)

	# red_again = cv2.imread('red.png',0)
	# cv2.imshow('out1', empty)
	# cv2.imshow('out2', red_again)



	empty_RGB = cv2.cvtColor(empty_BGR, cv2.COLOR_BGR2RGB)
	# cv2.imshow('last', empty_new)

	final = np.concatenate([empty_BGR, empty_RGB], axis=1)
	cv2.imshow('final', final)
	cv2.waitKey(0)




if __name__ == '__main__':
	main()