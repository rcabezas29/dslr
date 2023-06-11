import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

houses_colors = {
    'Slytherin': '#366447',
    'Ravenclaw': '#3c4e91',
    'Hufflepuff': '#efbc2f',
    'Gryffindor': '#a6332e',
}

if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1])

    sns.pairplot(data[6:], hue='Hogwarts House', diag_kind="hist", palette=houses_colors)
    plt.show()
