import pandas as pd
import numpy as np

path = "../All_in_one/"
test = pd.read_csv("../TestResults.csv")
print(test.head())
test.loc[test.ymin]