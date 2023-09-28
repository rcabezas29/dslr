import pandas as pd
import numpy as np
import sys

houses = [
	'Slytherin',
	'Ravenclaw',
	'Hufflepuff',
	'Gryffindor'
]

def	usage():
	print(f"  Usage:\npython3 {sys.argv[0]} [dataset].csv")
	exit(1)

def	clean_data(df: pd.DataFrame) -> np.ndarray:
	features = df.columns[5:]
	df = df[features].fillna(0)
	return df[['Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','Transfiguration','Charms','Flying']].to_numpy()
