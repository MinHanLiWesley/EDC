嘗試hybrid model (DNN + KM)
x1 = [入管溫度, 出管溫度]
~y = [x1, x2, x3,..., x29, 初始操作壓力, CCl4起始重量百分濃度, 總滯留時間(含下一管), 下一管滯留時間] #從KM計算出經過的重量百分濃度 
y = 108rxns的裂解率(X)

訓練資料格式
Ti, Te, x1, x2, x3,..., x29, Pi, CCl4, t, t_r, X

建立資料方式
mass_flow_rate = 72~55 # 30TPY的mass flow rate/area對照於20TPY的area約為55 T/H進料速度
#入口溫度及出口溫度分別固定為322及486，在這之間再generate 17個溫度點
初始壓力為Pi = 11.4~13
CCl4起始重量百分濃度 = 0~0.0025

預計先建立 = 18(爐管/裂解反應) * 10(mass flow rate取10個點) * 10(溫度profile) * 10(Pi取10個點) * 10(CCl4取10個點)
跑10000次模擬！

