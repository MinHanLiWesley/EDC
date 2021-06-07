
import random
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d,CubicSpline

# big_arr = []
# mass_flow_rate,ccl4,pin,tin = 28.0,300.0,13.1,330.0
# raw_raw_T_list=[tin]
# delta = 467.-tin
# Final_name="Tprofile7.csv"
# big_arr.append(np.hstack([mass_flow_rate,ccl4,pin,np.round(np.linspace(raw_raw_T_list[0],(raw_raw_T_list[0]+delta),19,endpoint=True),2)]))
# for i in range(4):
#     # mass_flow_rate = random.randint(22,36)
#     # ccl4 = random.randrange(0,1000,50)
#     # pin = random.randint(124,144)/10
#     # mass_flow_rate = 36
#     # ccl4 = 0
#     # pin = 12.4
#     # raw_T_list=[random.randrange(300,350,10)]
    
#     # delta = random.randint(110,140)
#     raw_T_list = [raw_raw_T_list[0]]
#     boo=True
#     while(boo):
#         try:
#             a, b  = random.randint(10,delta-20.),random.randint(10,delta-70)
#             c = random.randint(10,delta-a-b)
#             d = random.randint(5,delta-a-b-c)
#             params={'t00': a,
#                     't04': b,
#                     't07': c,
#                     't10': d,
#                     't13': delta-a-b-c-d,
#                     }
#         except:
#             continue
#         boo = False
        
#     x = [0,4,7,10,13,18]
#     for _, value in params.items():
#         raw_T_list.append(np.round((raw_T_list[-1]+value), 2))
#     cs = interp1d(x,raw_T_list)
#     T_list = [np.round(cs(i),2) for i in range(19)]

#     arr = np.hstack([mass_flow_rate,ccl4,pin,T_list]).reshape(-1,)
#     big_arr.append(arr)

# col_list = ['mass','ccl4','pin','t0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18']
# df = pd.DataFrame(big_arr,columns=col_list)
# print(df)
# df.to_csv(Final_name,index=False)
# print(T_list)



from natsort import natsorted
import glob
interesting_files = natsorted(glob.glob("T*.csv"))
df = pd.concat((pd.read_csv(f, header = 0) for f in interesting_files))
df.to_csv("t_request.csv",index=False)