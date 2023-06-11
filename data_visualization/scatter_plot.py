import sys
import pandas as pd
import matplotlib.pyplot as plt

houses = [
    'Slytherin',
    'Ravenclaw',
    'Hufflepuff',
    'Gryffindor',
]

houses_colors = {
    'Slytherin': '#366447',
    'Ravenclaw': '#3c4e91',
    'Hufflepuff': '#efbc2f',
    'Gryffindor': '#a6332e',
}

if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1])

    for house in houses:
        house_data = data.loc[data['Hogwarts House'] == house]
        plt.scatter(house_data['Astronomy'], house_data['Defense Against the Dark Arts'], c=houses_colors[house])
    plt.show()
