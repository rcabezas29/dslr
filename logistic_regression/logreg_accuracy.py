import pandas as pd
import sys
from LogisticRegression import LogisticRegression as logReg
from utils import *

def	main():
	try:
		df = pd.read_csv(sys.argv[1], index_col=0)
	except:
		usage()

	x = clean_data(df)
	houses_thetas = pd.read_csv(".weights.csv")

	results = pd.DataFrame()

	for h in houses:
		lr = logReg(houses_thetas[h].to_numpy())
		x = lr.normalize_data(x)
		results[h] = [i * 100 for i in lr.predict(x)]

	house_prediction = []
	for _, row in results.iterrows():
		house_prediction.append(row.idxmax())

	mask = df['Hogwarts House'] == house_prediction

	print(f"{100 - (mask.value_counts()[1] / mask.value_counts()[0]) * 100}%")

if __name__ == "__main__":
    main()
