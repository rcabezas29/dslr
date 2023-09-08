import sys
import pandas as pd
from LogisticRegression import LogisticRegression as logReg

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

	for h in houses:
		lr = logReg(houses_thetas[h].to_numpy())
		print(lr.predict_(x))

if __name__ == "__main__":
    main()