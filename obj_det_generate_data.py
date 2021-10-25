import cv2.cv2 as cv2
import pandas as pd

def main():
	# df_labels = pd.read_csv('parsed_data.csv', index_col='frame', header=0)
	df_labels = pd.read_csv('data/parsed_data.csv')

	## checking for NaNs
	# print(df_labels.isnull().sum().sum())
	# df_labels.dropna(inplace=True)

	cap = cv2.VideoCapture('data/top-100-shots-rallies-2018-atp-season.mp4')

	class_index = 0
	train_size = 30000
	with open("data/object_detection/box_score_train.lst", "w+") as file_out_train, open("data/object_detection/box_score_test.lst", "w") as file_out_test:

		frame_counter = 0
		id_entry = 0
		while True:
			out_file = file_out_train if frame_counter<train_size else file_out_test
			ret, frame = cap.read()
			key = cv2.waitKey(1)
			if (not ret) or key == ord('q'):
				break
			else:
				height, width, channels = frame.shape
				## check csv entry corresponding to current frame
				x_0 = df_labels[df_labels['frame'] == frame_counter]['x_0']
				y_0 = df_labels[df_labels['frame'] == frame_counter]['y_0']
				x_1 = df_labels[df_labels['frame'] == frame_counter]['x_1']
				y_1 = df_labels[df_labels['frame'] == frame_counter]['y_1']

				if frame_counter%40 == 0:
					## check if current frame actually had bbox annotations
					if len(x_0)>0 and len(y_0)>0 and len(x_1)>0 and len(y_1)>0:
						cv2.imwrite(f'data/object_detection/images/frame_{str(frame_counter).zfill(6)}.png', frame)
						out_file.write(f"{id_entry}\t{4}\t{5}\t{frame.shape[1]:}\t{frame.shape[0]}")
						x_0 = (x_0.iloc[0]/width)
						y_0 = (y_0.iloc[0]/height)
						x_1 = (x_1.iloc[0]/width)
						y_1 = (y_1.iloc[0]/height)
						out_file.write(f"\t{class_index}\t{x_0:0.4f}\t{y_0:0.4f}\t{x_1:0.4f}\t{y_1:0.4f}\tframe_{str(frame_counter).zfill(6)}.png\n")
						# print(f'frame: {frame_counter} - x_0: {x_0}, y_0: {y_0}')
						# frame = cv2.rectangle(frame, (x_0,y_0), (x_1,y_1), (0,0,255), 4)
						id_entry += 1


				cv2.imshow('output', frame)
				frame_counter += 1

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()