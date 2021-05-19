import pandas as pd
import pandas.io.common
import glob

from pandas.core.construction import is_empty_data
Final_name = "../training_data_FPC_V8_addprev_area.csv"
# Final_name = "RawDataInput.csv"

from os.path import dirname
from os import listdir
print(__file__)
from natsort import natsorted


# interesting_files = natsorted(glob.glob(f"{dirname(__file__)}/FPC_cracking_V6_prevX_normal/*.csv"))
# interesting_files2 = natsorted(glob.glob(f"{dirname(__file__)}/FPC_cracking_V6_prevX_random/*.csv"))


# for f in interesting_files:
#     try:
#         print(f)
#     except(pd.errors.EmptyDataError):
#         print(f)

# df = pd.concat((pd.read_csv(f, header = 0) for f in interesting_files))
# df2 = pd.concat((pd.read_csv(f, header = 0) for f in interesting_files2))
# df3 = pd.concat([df,df2],ignore_index=True)
# print(df3)

# df4 = pd.read_csv("../training_data_FPC_V7_addprev_Temprand.csv")
# df5 = pd.read_csv("../training_data_FPC_V6_addprev_Temprand.csv")
# df_diff = pd.concat([df5,df4]).drop_duplicates(['Ti','Te','X'],keep=False)
# print(df_diff)
df4 = pd.read_csv(Final_name)
print(df4.iloc[37476:37476+432])
df4.drop(df4.index[37476:37476+432],inplace=True)
df4.reset_index(drop=True,inplace=True)
print(df4)
# df3['prev_X'] = df4['X_prev']
# df3.rename(columns={'prev_X':'X_prev'},inplace=True)
# df3['X']=df4['X']
# print(df3)
# print(df4)
df4.to_csv(Final_name,index=False)
# print(df4.columns[df3.isna().any()].tolist())
# print(df)

# df = pd.read_csv(Final_name)
# print(len(df))
# print(df.query("Ti>368 & Ti<369"))
# print(len(df.query("Te==368.35")))
# df.loc[df.query("Te==368.35").index,'Te'] = 368.4
# print(len(df.query("Te==368.4")))
# df.to_csv(Final_name,index=False)
# df2 = pd.read_csv("RawDataInput.csv")
# print(df2['X'])
# print(df['X'])
# df['X'] = df2['X']/100
# print(df['CCl4_X_0'])
# df.to_csv(Final_name,index=False)
# plt.scatter(range(674976),df['CCl4_X_0'].to_list())
# plt.savefig('test.png')

# df = pd.read_csv("raw_data_cracking.csv")
# df2 = pd.read_csv("RawDataInput.csv")
# df = df.append(df2,ignore_index=True,verify_integrity=True)
# print(df)
# print(df2)
# df.sort_values(by=['mass_flow_rate','CCl4_X_0','pressure_0'],kind='mergesort',ignore_index=True)
# df.to_csv("raw2.csv",index=False)