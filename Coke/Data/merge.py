import pandas as pd
import  math
df = pd.read_csv("training_data_coking_V3_27_36.csv")
print(df)
# df2 = pd.read_csv("RawDataInput.csv")
df2 = pd.read_csv("RawData27_36.csv")
# df2.drop(['A'],axis=1,inplace=True)
# df3 = pd.read_csv("RawData/raw_data_coke_m=23_p=11.4.csv")
# print(len(df))
# print(len(df2))
# df2.eval("A=CCl4%500",inplace=True)
# print(df2[50:80])
# print(len(df2.query("A==@k")))
# print(len(df2.query("A!=0.0")))

# df5 = df2.drop(df2.query("CCl4%500 !=0").index)
# print(len(df5))

# df2.to_csv("RawData2.csv",index=False)
a = []
for i in range(len(df2)):
    if i % 18 == 0:
        a.append(0)
    else:
        a.append(df2.iloc[i-1,-1])

df['prev_coke']=a
df['coke']=df2.iloc[:,-1]
print(df)
df.to_csv("trainingdata_coking_V3_27_36_500ccl4.csv",index=False)
# df5 = pd.read_csv("trainingdata_coking_V2.csv")
# print(df5[:20])
# df.rename(columns={'prev_X':'X_prev'},inplace=True)
# df.loc[:,'X_prev'] = a
# df.to_csv("Data/training_data_cracking_FPC_V6_randtemp.csv",index=False)
# print(df)
# print(df2.query("Ti ==340 & Te ==368.35").index)
# df2.iloc[df2.query("Ti ==340 & Te ==368.35").index,3] = 368.4
# df2.iloc[df2.query("Ti ==368.35 & Te ==393.65").index,2] = 368.4
# df2.iloc[df2.query("Ti ==368.4 & Te ==393.65").index,3] = 393.7
# df2.iloc[df2.query("Ti ==393.65 & Te ==416.5").index,2] = 393.7
# df2.to_csv("RawDataInput_total.csv",index=False)
# print(df2.query("Ti ==340 & Te ==368.35").index)








