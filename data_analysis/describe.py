import sys
from pandas import read_csv, DataFrame
import math
import numpy as np

def	data_coefficient_varation(data) -> float:
	return data_variance(data) / data_mean(data) * 100

def	data_kurtosis(data) -> float:
	mean = data_mean(data)
	variance = data_variance(data)
	return (data_sum(data - mean)) ** 4 / data_count(data) * (variance ** 4)

def	data_variance(data) -> float:
	mean = data_mean(data)
	size = data_count(data)
	sum = 0
	for val in data:
		if math.isnan(val):
			val = 0
		sum += (val - mean)**2
	return sum / (size - 1)

def	data_sum(data) -> float:
	sum = 0
	for val in data:
		sum += val
	return sum

def	data_skewness(data) -> float:
	return 3 * (data_mean(data) - data_median(data)) / data_standard_deviation(data)

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
	return math.sqrt(data_variance(data))

def data_mean(data) -> float:
	sum = 0
	size = data_count(data)
	for val in data:
		if math.isnan(val):
			continue
		else:
			sum += val
	return sum / size

def data_count(data) -> int:
	i = 0
	for _ in data:
		i += 1
	return i

if __name__ == "__main__":
	data = read_csv(sys.argv[1]).fillna(0)
	features = data.columns[6:]
	data_functions = [data_count, data_mean, data_standard_deviation, data_min, data_first_quartile, data_median, data_third_quartile, data_max, data_skewness, data_sum, data_variance, data_kurtosis, data_coefficient_varation]
	d = {}
	for f in features:
		d[f] = list(map(lambda func: func(data[f]), data_functions))
	df = DataFrame(data=d, index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'Skewness', 'Sum', 'σ', 'Kurtosis', 'CV'])
	print(df)
