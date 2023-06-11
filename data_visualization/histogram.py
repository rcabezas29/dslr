import sys
import pandas as pd
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1])
    features = data.columns[6:]

    for f in features:
        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

        for house in houses:
            house_data = data.loc[data['Hogwarts House'] == house]
            axs.hist(house_data[f], alpha=0.5, bins=15, color=houses_colors[house], label=house)
        
        plt.legend(loc='upper right')
        plt.show()
