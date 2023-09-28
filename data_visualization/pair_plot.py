import sys
from pandas import read_csv
import seaborn as sns
import matplotlib.pyplot as plt

houses_colors = {
    'Slytherin': '#366447',
    'Ravenclaw': '#3c4e91',
    'Hufflepuff': '#efbc2f',
    'Gryffindor': '#a6332e',
}

data = read_csv(sys.argv[1])

features = data.columns[6:]

features = ['Hogwarts House'] + features.tolist()

sns.pairplot(data[features], hue='Hogwarts House', diag_kind="hist", palette=houses_colors)
plt.show()
