import sklearn
import math as m
import numpy as np
import pandas
import csv


data=[]
try:
    with open('train.csv', 'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            data = data.append(row)
            print(data)
except Exception as e:
    print (e)

data2 = np.array(data)

for row in data:
    print(row)