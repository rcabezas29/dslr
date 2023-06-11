import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1])

    sns.pairplot(data[6:], hue='Hogwarts House')
    plt.show()
