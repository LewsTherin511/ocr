import cv2.cv2 as cv2
import pytesseract
import numpy  as np

def main():
	img = cv2.imread('data/box_score_03.png')

	# Adding custom options
	custom_config = r'--oem 3 --psm 6'

	img_gray = get_grayscale(img)
	img_canny = canny(img)
	img_thresh = thresholding(img_gray)
	img_denoise = remove_noise(img)
	img_open = opening(img)
	print(pytesseract.image_to_string(img_denoise, config=custom_config))
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