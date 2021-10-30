import pandas as pd
import os
import numpy as np

def main():
	pd.options.display.max_columns = None
	pd.options.display.width=None

	df_labels = pd.read_csv('data/parsed_labels.csv')
	df_pred = pd.read_csv('results/00_predictions.csv')
	df_match= pd.merge(df_labels,df_pred,how='left',on = ['frame'])

	# print(df_match)

	# df_match = pd.read_csv('results/01_match.csv')

	df_match['match_serving'] = np.where(df_match["serving_label"] == df_match["serving_pred"], 1, 0)
	df_match['match_name_1'] = np.where(df_match["name_1_label"] == df_match["name_1_pred"], 1, 0)
	df_match['match_name_2'] = np.where(df_match["name_2_label"] == df_match["name_2_pred"], 1, 0)
	df_match['match_score_1'] = np.where(df_match["score_1_label"] == df_match["score_1_pred"], 1, 0)
	df_match['match_score_2'] = np.where(df_match["score_2_label"] == df_match["score_2_pred"], 1, 0)

	accuracy_serving = df_match['match_serving'].sum()/df_match.shape[0]
	accuracy_name_1 = df_match['match_name_1'].sum()/df_match.shape[0]
	accuracy_name_2 = df_match['match_name_2'].sum()/df_match.shape[0]
	accuracy_score_1 = df_match['match_score_1'].sum()/df_match.shape[0]
	accuracy_score_2 = df_match['match_score_2'].sum()/df_match.shape[0]

	print(f'Serving: {accuracy_serving:.2f}')
	print(f'Name 01: {accuracy_name_1:.2f}')
	print(f'Name 02: {accuracy_name_2:.2f}')
	print(f'Score 01: {accuracy_score_1:.2f}')
	print(f'Score 02: {accuracy_score_2:.2f}')



if __name__ == '__main__':
	main()