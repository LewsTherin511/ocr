import cv2


def main():

	cap = cv2.VideoCapture('data/top-100-shots-rallies-2018-atp-season.mp4')

	while True:
		ret, frame = cap.read()
		frame = cv2.resize(frame, (640,480))
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		ret, frame_thresh = cv2.threshold(frame_gray, 127, 255, cv2.THRESH_BINARY_INV)
		key = cv2.waitKey(1)
		if key == ord('q'):
			break
		cv2.imshow('output', frame_thresh)


if __name__ == '__main__':
	main()