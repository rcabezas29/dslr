import sys
import pandas as pd

if __name__ == "__main__":
    try:
        data = pd.read_csv(sys.argv[1])
    except:
        print('Unexpected argument')
        exit(1)
    data = data.sort_values(by=['Birthday'])