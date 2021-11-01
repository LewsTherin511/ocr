import os
import numpy as np

def main():
	# score_1 = ['3', '0', '3', '0']
	# score_2 = ['6', '1', '0']

	# score_1 = ['6', '3', '1', '1', '5']
	# score_2 = ['3', '6', '0', '3', '0']

	# score_1 = ['1', '0']
	# score_2 = ['1', '1', '5']

	score_1 = ['4', '1', '0']
	score_2 = ['6', '2', '4', '0']

	print('Pre')
	print(f'\tscore_1: {score_1}')
	print(f'\tscore_2: {score_2}')


	if len(score_1) >= 2 and len(score_2)>= 2 and abs(len(score_1)-len(score_2))==1:
		long = score_1 if len(score_1)>len(score_2) else score_2
		short = score_1 if len(score_1)<len(score_2) else score_2
		# print(f'The long list is: {long}')
		# print(f'The short list is: {short}')
		if short[-1] == '0':
			tmp = ''.join(long[-2:])
			del long[-2:]
			long.append(tmp)


	if len(score_1) == len(score_2) and len(score_1) >= 3:
		collapse_last_two = False
		## if last digit is 0 or 5, check previous ones to determine whether it is score in curernt game, or last two digits should be collapsed
		if score_1[-1] == '0':
			# print('check_01')
			if score_1[-2] == '3' or score_1[-2] == '4':
				# print('check_02')
				if int(score_1[-4]) >= 6 or int(score_2[-4])>= 6:
					# print('check_03')
					collapse_last_two = True
		elif score_1[-1] == '5':
			# print('check_01')
			if score_1[-2] == '1' or score_1[-2] == '4':
				# print('check_02')
				if int(score_1[-4]) >= 6 or int(score_2[-4])>= 6:
					# print('check_03')
					collapse_last_two = True


		if collapse_last_two:
			# print('collapsing scores!!')
			for score in score_1, score_2:
				tmp = ''.join(score[-2:])
				del score[-2:]
				score.append(tmp)
				# print(f'\t\tcollapsed score {score}')


	print('Post')
	print(f'\tscore_1: {score_1}')
	print(f'\tscore_2: {score_2}')



if __name__ == '__main__':
	main()