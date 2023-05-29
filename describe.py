import sys
import pandas as pd
import math
import numpy as np

def	data_third_quartile(data) -> float:
	sorted_data = np.sort(data)
	return sorted_data[int(data_count(data) / 4) * 3]

def	data_median(data) -> float:
	sorted_data = np.sort(data)
	return sorted_data[int(data_count(data) / 2)]

def	data_first_quartile(data) -> float:
	sorted_data = np.sort(data)
	return sorted_data[int(data_count(data) / 4)]

def data_max(data) -> float:
	max = sys.float_info.min
	for val in data:
		if (val > max and not math.isnan(val)):
			max = val
	return max

def data_min(data) -> float:
	min = sys.float_info.max
	for val in data:
		if (val < min and not math.isnan(val)):
			min = val
	return min

def data_standard_deviation(data) -> float:
	mean = data_mean(data)
	size = data_count(data)
	sum = 0
	for val in data:
		if math.isnan(val):
			val = 0
		sum += (val - mean)**2
	return math.sqrt(sum / (size - 1))

def data_mean(data) -> float:
	sum = 0
	size = data_count(data)
	for val in data:
		if math.isnan(val):
			continue
		else:
			sum += val
	return sum / data_count(data)

def data_count(data) -> int:
	i = 0
	for _ in data:
		i += 1
	return i

if __name__ == "__main__":
	data = pd.read_csv(sys.argv[1])
	f1 = data['Arithmancy']
	print(data_third_quartile(f1))
