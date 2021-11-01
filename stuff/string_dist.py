import difflib

def main():
	name = '_* A. Zverev'
	targets = ['A. Zverev', 'R. Federer', 'R. Nadal']


	for x in targets:
		ratio = difflib.SequenceMatcher(None, name, x).ratio()
		print(f'Similarity between {name} and {x}: {ratio}')

	correct_name = difflib.get_close_matches(name, targets)
	print(correct_name)


if __name__ == '__main__':
	main()