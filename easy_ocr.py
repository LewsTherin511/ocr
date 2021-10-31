from easyocr import Reader
import argparse
import cv2.cv2 as cv2

def main():

	frame = cv2.imread('data/box_score_00.png')
	# frame = cv2.imread('data/pre_processing_05.png')



	reader = Reader(lang_list=['en'], gpu=False)
	results = reader.readtext(frame)

	for (bbox, text, prob) in results:
		# display the OCR'd text and associated probability
		print(text)


if __name__ == '__main__':
	main()