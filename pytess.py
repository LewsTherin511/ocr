import cv2.cv2 as cv2
import pytesseract
import numpy  as np

def main():
	# frame = cv2.imread('data/box_score_05.png')
	frame = cv2.imread('data/screenshot_01.png')
	# frame = cv2.GaussianBlur(frame,(5,5),cv2.BORDER_DEFAULT)
	height, width, channels = frame.shape
	frame_crop = frame[-int(height/3):, 0:int(width/3)]

	# Adding custom options
	# custom_config = r'--oem 3 --psm 6'
	custom_config = r'--oem 3 --psm 4'


	frame_gray = cv2.cvtColor(frame_crop, cv2.COLOR_RGB2GRAY)
	ret, frame_thresh = cv2.threshold(frame_gray,128,255,cv2.THRESH_BINARY)
	contours_list, hierarchy = cv2.findContours(frame_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	frame_thresh_orig = frame_thresh.copy()

	for cnt in contours_list:
		x, y, w, h = cv2.boundingRect(cnt)
		approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
		if len(approx) == 4 and (w*h>50):
			frame = cv2.drawContours(frame, cnt, -1, (0,255,0), 1)
			# frame = cv2.circle(frame, (x,y), 2, (0,255,0),3)
			# frame = cv2.circle(frame, (x+w,y+h), 2, (0,255,0),3)
			frame_thresh[y:y+h, x:x+w] = cv2.bitwise_not(frame_thresh[y:y+h, x:x+w])
			frame_thresh = cv2.bitwise_not(frame_thresh)

	kernel = np.ones((2, 2), np.uint8)
	frame_thresh = cv2.erode(frame_thresh, kernel, iterations=1)
	frame_thresh = cv2.dilate(frame_thresh, kernel, iterations=1)
	result = pytesseract.image_to_string(frame_thresh, config=custom_config)
	# result = pytesseract.image_to_string(frame_thresh)
	print(result)
	cv2.imshow('thresh_smooth', frame_thresh)
	cv2.imshow('thresh_orig', frame_thresh_orig)
	cv2.imshow('orig', frame)
	cv2.waitKey(0)

	cv2.destroyAllWindows()


	# img_gray = get_grayscale(img)
	# img_canny = canny(img)
	# img_thresh = thresholding(img_gray)
	# img_denoise = remove_noise(img)
	# img_open = opening(img)
	# print(pytesseract.image_to_string(img_denoise, config=custom_config))
	# print(pytesseract.image_to_string(img_canny))





# get grayscale image
def get_grayscale(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
	return cv2.medianBlur(image,3)

#thresholding
def thresholding(image):
	return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
	kernel = np.ones((5,5),np.uint8)
	return cv2.dilate(image, kernel, iterations = 1)

#erosion
def erode(image):
	kernel = np.ones((5,5),np.uint8)
	return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
	kernel = np.ones((1,1),np.uint8)
	return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
	return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
	coords = np.column_stack(np.where(image > 0))
	angle = cv2.minAreaRect(coords)[-1]
	if angle < -45:
		angle = -(90 + angle)
	else:
		angle = -angle
	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	return rotated


if __name__ == '__main__':
	main()