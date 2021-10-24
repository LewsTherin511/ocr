import json
import pandas as pd
import csv

def main():
	with open('data/top-100-shots-rallies-2018-atp-season-scoreboard-annotations.json') as infile:
		data = json.load(infile)

	rows_list = []
	for frame, content in data.items():
		row_dict = {}
		row_dict['frame'] = frame.zfill(5)
		row_dict['x_0'] = content['bbox'][0]
		row_dict['y_0'] = content['bbox'][1]
		row_dict['x_1'] = content['bbox'][2]
		row_dict['y_1'] = content['bbox'][3]
		row_dict['serving'] = content['serving_player']
		row_dict['name_1'] = content['name_1']
		row_dict['name_2'] = content['name_2']
		row_dict['score_1'] = content['score_1']
		row_dict['score_2'] = content['score_2']
		rows_list.append(row_dict)


	df_out = pd.DataFrame(rows_list).sort_values('frame')

	df_out.to_csv('parsed_data.csv', index=None)
	# keys = rows_list[0].keys()
	# with open('parsed_data.csv', 'w+') as outfile:
	# 	dict_writer = csv.DictWriter(outfile, keys)
	# 	dict_writer.writeheader()
	# 	dict_writer.writerows(rows_list)




if __name__ == '__main__':
	main()