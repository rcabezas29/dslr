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
    'Slytherin': 'green',
    'Ravenclaw': 'blue',
    'Hufflepuff': 'yellow',
    'Gryffindor': 'red',
}

if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1]).fillna(0)
    features = data.columns[6:]

    for f in features:
        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

        for house in houses:
            house_data = data.loc[data['Hogwarts House'] == house]
            axs.hist(house_data[f], alpha=0.25, bins=15, color=houses_colors[house])
        
        plt.show()