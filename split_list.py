

def main():
	list_all = [['ba', '4', 'cdf'],
				['a4a', '56', 'scdf'],
				['baaa', 'd8g', '128']]

	for lst in list_all:
		lst[:] = list(''.join(lst))
		lst[:] = [x.replace('8', '5') for x in lst]


	print('After')
	for x in list_all:
		print(x)


if __name__ == '__main__':
	main()