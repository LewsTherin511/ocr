import cv2
import pandas as pd

def main():
	# df_labels = pd.read_csv('parsed_data.csv', index_col='frame', header=0)
	df_labels = pd.read_csv('parsed_data.csv')

	print(df_labels.isnull().sum().sum())

	# # print(len(df_labels))
	# df_labels.dropna(inplace=True)
	# # print(len(df_labels))

	cap = cv2.VideoCapture('data/top-100-shots-rallies-2018-atp-season.mp4')


	frame_counter = 0
	while True:
		ret, frame = cap.read()
		key = cv2.waitKey(1)
		if (not ret) or key == ord('q'):
			break
		x_0 = df_labels[df_labels['frame'] == frame_counter]['x_0']
		y_0 = df_labels[df_labels['frame'] == frame_counter]['y_0']
		x_1 = df_labels[df_labels['frame'] == frame_counter]['x_1']
		y_1 = df_labels[df_labels['frame'] == frame_counter]['y_1']

		if len(x_0)>0 and len(y_0)>0 and len(x_1)>0 and len(y_1)>0:
			x_0 = int(x_0.iloc[0])
			y_0 = int(y_0.iloc[0])
			x_1 = int(x_1.iloc[0])
			y_1 = int(y_1.iloc[0])
			# print(f'frame: {frame_counter} - x_0: {x_0}, y_0: {y_0}')
			frame = cv2.rectangle(frame, (x_0,y_0), (x_1,y_1), (0,0,255), 4)


		cv2.imshow('output', frame)
		frame_counter += 1



if __name__ == '__main__':
	main()