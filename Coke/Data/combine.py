import pandas as pd
import pandas.io.common
import glob
from pathlib import Path
from pandas.core.construction import is_empty_data
Final_name = "training_data_coking_V3_27_36.csv"
# Final_name = "RawDataInput_total.csv"
# Final_name = "RawDataInput_V2.csv"

from os.path import abspath,dirname
from os import listdir
print(__file__)
from natsort import natsorted

# interesting_files = glob.glob(f"{abspath(dirname(__file__))}/cokingV1/*.csv")
# interesting_files = glob.glob(f"cokingV1/*.csv")
interesting_files = natsorted(glob.glob(f"{abspath(dirname(__file__))}/dataV3/*.csv"))
# print(len(interesting_files))
# # interesting_files = natsorted(listdir(f"{dirname(__file__)}/FPC_cracking_V4_3m_irrev"))
# mass = ['23.0','27.0','36.0']
# pin = ['10.4','11.4','12.4','13.4','14.4']
# ccl4 = ['0.0','500.0','1000.0','1500.0','2000.0','2500.0']
# chcl3 = ['0.0','100.0','200.0','300.0','400.0','500.0']
# tri = chcl3
# cp = chcl3
# tin=['300.0','310.0','320.0','330.0','340.0','350.0']

# for f in interesting_files:
#     a = Path(f).stem.split("_")
#     if (a[0] not in mass or
#         a[1] not in pin or
#         a[2] not in ccl4 or
#         a[3] not in chcl3 or
#         a[4] not in tri or
#         a[5] not in cp or
#         a[6] not in tin):
#         print(f)
    
# for f in interesting_files:
#     try:
#         pd.read_csv(f, header = 0)
#     except:
#         print(f)
print(len(interesting_files))
# df = pd.concat((pd.read_csv(f, header = 0) for f in interesting_files))
 
# df.to_csv(Final_name,index=False)
# print(df)
raw_data_lst = natsorted(glob.glob(f"{abspath(dirname(__file__))}/RawData/*.csv"))
df = pd.read_csv('RawData/raw_data_coke_m=36_p=11.4.csv')
df2 = pd.concat((pd.read_csv(f).query('CCl4 % 500 == 0') for f in raw_data_lst[4:]))
print(len(df2)/18)
df2.to_csv("RawData27_36.csv",index=False)
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

# for name in interesting_files:
#     df = pd.read_csv(name)
#     print(df.isnull().values.any())
