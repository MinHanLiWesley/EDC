# import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# import PyQt5
# print(matplotlib.get_backend())
# matplotlib.use('Qt5Agg')
# for T_in in np.linspace(300,350,6):
#     t_5 = 453 + np.random.normal(loc=0.,scale=2)
#     t_9 = 478 + np.random.normal(loc=0.,scale=1)
#     t_18 = 486.6

#     # t_0 = Tin t_5 = 453 t_9 = 478  t_18 = 486
#     T_list = [T_in , t_5 ,t_9 ,t_18]
#     # t 1 ~ 4
#     T_1to4 = np.linspace(T_in, t_5, 5,endpoint=False) + np.random.normal(loc=0,scale=1)
#     # print(T_1to4)
#     T_6to8 = np.linspace(t_5,t_9,4,endpoint= False) + np.random.normal(loc=0,scale=1)
#     T_10to18 = np.linspace(t_9,t_18,9,endpoint=False) + np.random.normal(loc=0,scale=1)

#     def append(pieces):
#         global T_list
#         for i in range(1,len(pieces)):
#             T_list.append(pieces[i])
    
#     append(T_1to4)
#     append(T_6to8)
#     append(T_10to18)
#     # print(len(T_list))
#     # print(sorted(T_list))
    
#     Index = range(1,19)
#     plt.figure(figsize=(15,10),dpi=100,linewidth = 2)
#     plt.plot(Index,T_list[1:],'s-',color = 'r', label="Temp")
#     plt.show()


import pandas as pd

# df = pd.read_csv("training_data_FPC_V6_addprev_Temprand.csv")
# df_index = df.query("X > 1").index
# df.iloc[34020:,[35,36]] = df.iloc[34020:,[35,36]].values/100
# print(df.iloc[34020:,[35,36]])
# # df2  = pd.read_csv("training_data_FPC_V1_3m_5p_6t_21Clppm.csv")
# df.to_csv("training_data_FPC_V6_addprev_Temprand.csv",index=False)

df = pd.read_csv("Data/training_data_cracking_FPC_V6_randtemp.csv")
a = []
for i in range(len(df)):
    if i % 18 == 0:
        a.append(0)
    else:
        a.append(df.iloc[i-1,-1])

df.rename(columns={'prev_X':'X_prev'},inplace=True)
df.loc[:,'X_prev'] = a
df.to_csv("Data/training_data_cracking_FPC_V6_randtemp.csv",index=False)
print(df)

# from scipy import interpolate

# def f(x):
#     tubes = [0,1,5,9,13,17,18]
#     temp = [308,347.58,430.50,460.79,480.61,486.10,486]
#     tck = interpolate.splrep(tubes,temp)
#     return interpolate.splev(x,tck)
# temp_list = []
# for i in range(19):
#     temp_list.append(f(i))

# plt.figure(figsize=(12,8),dpi=200,linewidth = 2)
# plt.title('Real factory operating temperature',fontsize=20)
# plt.xlabel('Tubes',fontsize=20)
# plt.ylabel('Temperature(°C)',fontsize=20)
# plt.grid(axis='y')
# plt.xticks(range(19))
# for i in range(19):
#     print((f"{temp_list[i]:.2f}"+","),end='\0')
# plt.plot(range(19),temp_list,'s-',color='r',label='temp')
# for a,b in zip(range(19),temp_list):
#     plt.text(a,b+0.001,'%.2f' % b,ha='center',va='bottom',fontsize=9)
# plt.savefig("temp.png")



# df = pd.read_csv("Data/training_data_cracking_FPC_V6_randtemp.csv")
# df1 = pd.read_csv("training_data_FPC_V5_addprev.csv")
# df.rename(columns={"prev_X":"X_prev"},inplace=True)
# print(df)
# print(df1)
# df2 = pd.concat([df1,df],ignore_index=True)
# print(df2)
# df2.to_cssv("training_data_FPC_V5_addprev.csv",index=False)
# df1.to_csv("training_data_FPC_V6_addprev.csv",index=False)
# print(df1.loc[:34019])
# df1.loc[:34019].to_csv("training_data_FPC_V5_addprev.csv",index=False)