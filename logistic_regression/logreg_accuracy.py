import pandas as pd
import sys
import numpy as np
from tqdm import tqdm
from LogisticRegression import LogisticRegression as logReg

houses = [
    'Slytherin',
    'Ravenclaw',
    'Hufflepuff',
    'Gryffindor'
]

def	usage():
	print("  Usage:\npython3 logreg_predict.py [dataset].csv")

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
		results[h] = [i * 100 for i in lr.predict(x)]

	print(results)

	house_prediction = []
	for _, row in results.iterrows():
		house_prediction.append(row.idxmax())

	mask = df['Hogwarts House'] == house_prediction

	print(f"{100 - (mask.value_counts()[1] / mask.value_counts()[0]) * 100}%")

if __name__ == "__main__":
    main()
