import pandas as pd
import sys
from LogisticRegression import LogisticRegression as logReg
import numpy as np

houses = [
    'Slytherin',
    'Ravenclaw',
    'Hufflepuff',
    'Gryffindor'
]

def	howarts_house_format(house: str, df: pd.DataFrame) -> np.array:
	mask = df['Hogwarts House'] == house
	return mask.astype(int).values.reshape(-1, 1)

def	usage():
	print("  Usage:\npython3 logreg_train.py [dataset]")

def	clean_data(df: pd.DataFrame) -> pd.DataFrame:
	return df[['Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','Transfiguration','Charms','Flying']]

if __name__ == "__main__":
	try:
		df = pd.read_csv(sys.argv[1], index_col=0)
	except:
		usage()
		exit(1)

	features = df.columns[5:]
	x = clean_data(df[features].fillna(0)).to_numpy()

	res = pd.DataFrame()
	for house in houses:
		y = howarts_house_format(house, df)
		thetas = np.zeros((x.shape[1] + 1, 1))
		lr = logReg(thetas, max_iter=1000)
		lr.fit_(x, y)
		res[house] = [i[0] for i in lr.theta]
	print(res)
	res.to_csv('.weights.csv')
