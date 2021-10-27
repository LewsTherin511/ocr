import cv2.cv2 as cv2
import pytesseract
import numpy  as np
import gluoncv as gcv
import mxnet as mx
import re
from matplotlib import pyplot as plt

def main():

	cap = cv2.VideoCapture('data/top-100-shots-rallies-2018-atp-season.mp4')
	# try:
	# 	ctx = [mx.gpu(1)]
	# except:
	# 	ctx = [mx.cpu()]
	ctx = mx.cpu()
	## object detection stuff
	custom_classes = ['box_score']
	net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=custom_classes, pretrained_base=False, ctx=ctx)
	net.load_parameters("data/object_detection/box_score_ssd_512_mobilenet1.0_coco_run_00/ep_035.params", ctx=ctx)	### GREAT!!!
	net.hybridize()


	frame_counter = 0
	while True:
		ret, frame_np_orig = cap.read()
		key = cv2.waitKey(1)
		if (not ret) or key == ord('q'):
			break
		else:
			height, width, channels = frame_np_orig.shape

			if frame_counter % 15 == 0:

				# Image pre-processing
				frame_nd_orig = mx.nd.array(cv2.cvtColor(frame_np_orig, cv2.COLOR_BGR2RGB)).astype('uint8')
				frame_nd_new, frame_np_new = gcv.data.transforms.presets.ssd.transform_test(frame_nd_orig, short=512, max_size=700)
				## detection
				frame_nd_new = frame_nd_new.as_in_context(ctx)
				class_IDs, scores, bboxes = net(frame_nd_new)

				## locate area around box score for cropping, in relative coords
				box_score_x0_rel = bboxes[0][0][0].asnumpy()/frame_np_new.shape[1]
				box_score_y0_rel = bboxes[0][0][1].asnumpy()/frame_np_new.shape[0]
				box_score_x1_rel = bboxes[0][0][2].asnumpy()/frame_np_new.shape[1]
				box_score_y1_rel = bboxes[0][0][3].asnumpy()/frame_np_new.shape[0]
				width_rel = box_score_x1_rel-box_score_x0_rel
				height_rel = box_score_y1_rel-box_score_y0_rel
				## cropping only if detection prob > 60%
				if scores[0][0] > 0.6:
					frame_crop = frame_np_orig[int((box_score_y0_rel-height_rel/10)*frame_np_orig.shape[0]):int((box_score_y1_rel+height_rel/10)*frame_np_orig.shape[0]),
								 int((box_score_x0_rel-width_rel/10)*frame_np_orig.shape[1]):int((box_score_x1_rel+width_rel/10)*frame_np_orig.shape[1])
								]
				else:
					frame_crop = None

				if (frame_crop is not None) and frame_crop.shape[0]>0 and frame_crop.shape[1]>0:
					# try:
					cv2.imshow('step_00', frame_crop)
					# except:
					# 	pass

					frame_gray = cv2.cvtColor(frame_crop, cv2.COLOR_RGB2GRAY)
					ret, frame_thresh = cv2.threshold(frame_gray,128,255,cv2.THRESH_BINARY)
					# cv2.imshow('step_01_crop&thresh', frame_thresh)
					## looking for areas of different color within box score, making everything more homogeneous
					contours_list, hierarchy = cv2.findContours(frame_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
					for cnt in contours_list:
						x, y, w, h = cv2.boundingRect(cnt)
						approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
						## locating quadrangolar areas and applying bitwise-not mask on them
						if len(approx) == 4 and (w*h>50):
							# frame_thresh = cv2.drawContours(frame_thresh, [cnt], 0, (0,0,255), 2)
							frame_thresh[y:y+h, x:x+w] = cv2.bitwise_not(frame_thresh[y:y+h, x:x+w])
							cv2.imshow('step_02_fix_inside_boxes', frame_thresh)

							## if cropped thresholded image is mostly white on black, apply bitwise_not
							if np.average(np.average(frame_thresh, axis=0), axis=0) < 127:
								frame_thresh = cv2.bitwise_not(frame_thresh)

							## black text on white background
							# frame_thresh = cv2.bitwise_not(frame_thresh)
							cv2.imshow('step_03_black_on_white', frame_thresh)
					## cleaning noisy areas
					kernel = np.ones((2, 2), np.uint8)
					frame_thresh = cv2.erode(frame_thresh, kernel, iterations=1)
					frame_thresh = cv2.dilate(frame_thresh, kernel, iterations=1)

					## if resulting frame here is mostly black, switch it to improve text detection
					# cv2.imwrite(f'data/thresholded/frame_thresh_{frame_counter}.png', frame_thresh)
					# hist = cv2.calcHist([frame_thresh], [0], None, [2], [0, 1])
					# hist /= hist.sum()
					# if hist[0]>hist[1]:
					# 	frame_thresh = cv2.bitwise_not(frame_thresh)
					try:
						cv2.imshow('step_04_final', frame_thresh)
					except:
						pass

					## text detection
					## custom_config = r'--oem 3 --psm 6'
					custom_config = r'--oem 3 --psm 4'
					result = pytesseract.image_to_string(frame_thresh, config=custom_config)
					polish_result(result)
					print()


		frame_counter += 1

	cv2.destroyAllWindows()
	cap.release()

	# img_gray = get_grayscale(img)
	# img_canny = canny(img)
	# img_thresh = thresholding(img_gray)
	# img_denoise = remove_noise(img)
	# img_open = opening(img)
	# print(pytesseract.image_to_string(img_denoise, config=custom_config))
	# print(pytesseract.image_to_string(img_canny))



def polish_result(result):
	lines = result.strip().split('\n')
	for line in lines:
		if len(line) > 0 and not line.isspace():
			# for each line, identify the selection mark, the name, and the mess at the end
			# assuming names can't have numbers in them
			match = re.match(r'^(\W+)?([^\d]+?)\s*([^a-zA-Z]+)$', line.strip())
			if match:
				selected_raw, name, numbers_raw = match.groups()
				# now parse the unprocessed bits
				selected = selected_raw is not None
				numbers = re.findall(r'\d+', numbers_raw)
				print(selected, name, numbers)




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