import pandas as pd
import os

def main():
	pd.options.display.max_columns = None
	pd.options.display.width=None

	df_labels = pd.read_csv('data/parsed_data.csv')
	df_pred = pd.read_csv('predictions_00.csv')

	df= pd.merge(df_labels,df_pred,how='left',on = ['frame'])

	print(df)

	# df['match_name_1'] = 1 if


if __name__ == '__main__':
	main()