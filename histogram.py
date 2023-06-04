import sys
import pandas as pd
from datetime import datetime

houses_per_year = {
    'Slytherin': 0,
    'Ravenclaw': 0,
    'Hufflepuff': 0,
    'Gryffindor': 0,
}

def divide_data_by_date(data):
    divided_data = {}
    for student in data:
        print(student)
        birthday = datetime.fromisoformat(student['Birthday'])
        divided_data[birthday.year] = houses_per_year

    for student in data:
        birthday = datetime.fromisoformat(student['Birthday'])
        divided_data[birthday.year][houses_per_year][student['Hogwarts House']] += 1
        

if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1]).fillna(0)
    #data = data.sort_values(by=['Birthday'])
    divide_data_by_date(data)
    #print(data)