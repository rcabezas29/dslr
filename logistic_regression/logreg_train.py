import pandas as pd
import sys
from LogisticRegression import LogisticRegression as logReg
import numpy as np
from tqdm import tqdm

houses = [
    'Slytherin',
    'Ravenclaw',
    'Hufflepuff',
    'Gryffindor'
]

def	hogwarts_house_format(house: str, df: pd.DataFrame) -> np.array:
	mask = df['Hogwarts House'] == house
	return mask.astype(int).values.reshape(-1, 1)

def	usage():
	print("  Usage:\npython3 logreg_train.py [dataset]")
	exit(1)

def	clean_data(df: pd.DataFrame) -> np.ndarray:
	features = df.columns[5:]
	df = df[features].fillna(0)
	return df[['Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','Transfiguration','Charms','Flying']].to_numpy()

if __name__ == "__main__":
	try:
		df = pd.read_csv(sys.argv[1], index_col=0)
	except:
		usage()

	x = clean_data(df)

	res = pd.DataFrame()
	for house in tqdm(houses):
		y = hogwarts_house_format(house, df)
		thetas = np.zeros((x.shape[1] + 1, 1))
		lr = logReg(thetas)
		lr.fit(x, y)
		res[house] = [i[0] for i in lr.theta]

	res.to_csv('.weights.csv')
