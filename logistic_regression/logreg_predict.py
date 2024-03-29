import sys
import pandas as pd
from LogisticRegression import LogisticRegression as logReg
import csv
from utils import *

def	write_prediction_file(house_prediction):
	with open('houses.csv', mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(['Index', 'Hogwarts House'])
		for index, value in enumerate(house_prediction):
			writer.writerow([index, value])

def	execute_prediction(x, houses_thetas) -> list:
	results = pd.DataFrame()
	for h in houses:
		lr = logReg(houses_thetas[h].to_numpy())
		x = lr.normalize_data(x)
		results[h] = [i * 100 for i in lr.predict(x)]

	house_prediction = []
	for _, row in results.iterrows():
		house_prediction.append(row.idxmax())

	return house_prediction

def	main():
	try:
		df = pd.read_csv(sys.argv[1], index_col=0)
	except:
		usage()

	x = clean_data(df)
	houses_thetas = pd.read_csv(".weights.csv")
	house_prediction = execute_prediction(x, houses_thetas)

	write_prediction_file(house_prediction)

if __name__ == "__main__":
    main()
