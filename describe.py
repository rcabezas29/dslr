import sys
import pandas as pd
import math

def	data_third_quartile(data) -> float:
	return data[(data_count(data) / 4) * 3]

def	data_second_quartile(data) -> float:
	return data[data_count(data) / 2]

def	data_first_quartile(data) -> float:
	return data[data_count(data) / 4]

def data_max(data) -> float:
    max = sys.float_info.min
    for val in data:
        if (val > max):
            max = val
    return max

def data_min(data) -> float:
    min = sys.float_info.max
    for val in data:
        if (val < min):
            min = val
    return min

def data_standard_deviation(data) -> float:
    mean = data_mean(data)
    sum = 0
    for val in data:
        sum += (val - mean)**2
    return math.sqrt(sum / (data_count - 1))

def data_mean(data) -> float:
    sum = 0
    for val in data:
        sum += val
    return sum / data_count(data)

def data_count(data) -> int:
    i = 0
    for _ in data:
        i += 1
    return i

if __name__ == "__main__":
    try:
        data = pd.read_csv(sys.argv[1])
        f1 = data['Arithmancy']
        f2 = data['Astronomy']
        f3 = data['Herbology']
        f4 = data['Defense Against the Dark Arts']
        f5 = data['Divination']
        f6 = data['Muggle Studies,Ancient Runes']
        f7 = data['History of Magic']
        f8 = data['Transfiguration']
        f9 = data['Potions']
        f10 = data['Care of Magical Creatures']
        f11 = data['Charms']
        f12 = data['Flying']
        
    except:
        print('Unexpected argument')
        exit(1)