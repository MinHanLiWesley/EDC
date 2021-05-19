import numpy as np
import pandas as pd
import glob
Final_name = "test.csv"
# NAME = "training_dataV1_3m_6p_6t_0.0.csv"
# df = pd.read_csv(NAME,header=0)
# print(df[:540])
temp = []
interesting_files = sorted(glob.glob("tr*.csv"))
print(interesting_files)
for mass_index in range(1,19):
    mass_index *= 90
    for f in interesting_files:
        df =pd.read_csv(f,header = 0)
        temp.append(df[mass_index-90:mass_index])

df2 = pd.concat(f for f in temp)
df2.to_csv(Final_name,index=False)


    