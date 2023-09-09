import sys
from pandas import read_csv
import matplotlib.pyplot as plt
from utils import usage

houses = {
    'Slytherin': 0,
    'Ravenclaw': 1,
    'Hufflepuff': 2,
    'Gryffindor': 3,
}

houses_colors = {
    'Slytherin': '#366447',
    'Ravenclaw': '#3c4e91',
    'Hufflepuff': '#efbc2f',
    'Gryffindor': '#a6332e',
}

def main():
    try:
        data = read_csv(sys.argv[1])
        features = data.columns[6:]
    except:
        usage()

    fig, axs = plt.subplots(4, 4, tight_layout=True)
    for i, f in enumerate(features):
        j = i % 4
        i = int(i / 4)

        for house in houses:
            house_data = data.loc[data['Hogwarts House'] == house]
            axs[i][j].hist(house_data[f], alpha=0.5, bins=15, color=houses_colors[house], label=house)

        axs[i][j].set_title(f)

    plt.show()

if __name__ == "__main__":
    main()
