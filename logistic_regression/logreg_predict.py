import sys
import pandas as pd
from LogisticRegression import LogisticRegression as logReg
import numpy as np

houses = [
    'Slytherin',
    'Ravenclaw',
    'Hufflepuff',
    'Gryffindor'
]

def	usage():
	print("  Usage:\npython3 logreg_predict.py [dataset]")

def	clean_data(df: pd.DataFrame) -> pd.DataFrame:
	return df[['Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','Transfiguration','Charms','Flying']]

def	main():
	try:
		df = pd.read_csv(sys.argv[1], index_col=0)
	except:
		usage()
		exit(1)

	features = df.columns[5:]
	x = clean_data(df[features].fillna(0)).to_numpy()
	houses_thetas = pd.read_csv(".weights.csv")

	results = pd.DataFrame()

	for h in houses:
		lr = logReg(houses_thetas[h].to_numpy())
		x = lr.normalize_data(x)
		results[h] = [i * 100 for i in lr.predict_(x)]
		# results.append(lr.predict_(x) * 100)

	print(results)

	# for i, x in enumerate(results):
	# 	print()

if __name__ == "__main__":
    main()