import pandas as pd
import difflib

def main():
	# df_names = pd.read_csv('names/wrong.csv')
	# possible_names = ['mark', 'anthony', 'jack']
	# df_names["correct_name"] = df_names["name"].apply(lambda x: difflib.get_close_matches(x, possible_names)[0])


	df_labels = pd.read_csv('results/01_predictions.csv')
	df_names = pd.read_csv('data/names_labels.csv')
	possible_names = df_names['Name'].tolist()
	df_labels["name_1_pred_polished"] = [next(iter(difflib.get_close_matches(str(name).lower(), possible_names)), name) for name in df_labels["name_1_pred"]]
	df_labels["name_2_pred_polished"] = [next(iter(difflib.get_close_matches(str(name).lower(), possible_names)), name) for name in df_labels["name_2_pred"]]
	df_labels.to_csv('results/01_predictions_polished.csv')




if __name__ == '__main__':
	main()