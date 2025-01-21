import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from itertools import product
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

#完整的預測程式
#%%
#名稱變數設定
#2011-06-18->2021-06-18 資料期間

data_name = '2603.csv'             #設定 data 名稱
start_date = '2011-07-01'          #data 開始的日期
end_date = '2021-06-18'            #data 結束的日期
Stock = 0
start_date_plot = '2011/07/01'     #設立圖片開始的時間
end_date_plot = '2021/06/18'       #設定圖片尾端的時間
interval = 300                     #設定圖像間隔(天數)
image_name = '0620.png'            #儲存圖像名稱
#%%
#圖片設定 Section 0 Setting Data

my_font = fm.FontProperties(fname=r'C:\Windows\Fonts\msjh.ttc', size=20) #fm.FontProperties --> 需要先導入matplotlib.font_manager
#找到中文字型在電腦中的位置，匯入中文字型庫
Data = pd.read_csv(data_name, encoding = 'utf-8')                                            #讀取 cvs 檔(Stock_index.csv)
Data = Data[(Data['Date'] >= start_date) & (Data['Date'] <= end_date)]   #從檔案中找尋 'Data'(字典) 介於起始日期跟結束日期
Data['Date'] = pd.to_datetime(Data['Date']).dt.strftime('%Y/%m/%d')      #可以將字典形式時間轉換為可讀時間，時間格式設定
Data.sort_index(ascending=False,inplace=True)  # 時間從舊到新排序

Stock_list = Data.Stock.unique().tolist()                                #取出(unique) Data 有哪些項目 (7個) 並設成 list(tolist)
#Stock Data Open High Low Close Volume
stock = Stock_list[Stock]                                                #從串列中找 title 為 stock 的數據
#%%
#歷史報酬率(日報酬)

data = Data[Data['Stock'] == stock].copy()

data['DayReturn'] = np.log(data['Close']).diff().shift(0)
#值為把 data 的 close 取對數，計算與前一行的差異(diff)，對數據進行移動(shift)

data['Lag_1'] = data['DayReturn'].shift(0).diff()           #建立新的一列 lag_1，值為對數據往後進行移動(shift)，計算與前一行的差異
data['Lag_2'] = data['DayReturn'].shift(1).diff()           #建立新的一列 lag_2，值為對數據往後進行一次移動(shift)，計算與前一行的差異
data['Lag_3'] = data['DayReturn'].shift(2).diff()           #建立新的一列 lag_3，值為對數據往後進行兩次移動(shift)，計算與前一行的差異
data['Lag_4'] = data['DayReturn'].shift(3).diff()           #建立新的一列 lag_4，值為對數據往後進行三次移動(shift)，計算與前一行的差異
data['Lag_5'] = data['DayReturn'].shift(4).diff()           #建立新的一列 lag_5，值為對數據往後進行四次移動(shift)，計算與前一行的差異
cols = ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5']        #設立串列 cols
data = data.dropna()                                        #去掉含有缺失值的樣本(行)
data['Direction'] = np.sign(data['DayReturn']).astype(int)    #返回數字符號的元素指示(sign)，轉換數據類型(ast)，由浮點數轉成整數(int)
#%%
def model_fit(data, data_signal, signal_name, sign_list):
    models = {'Logistiic': LogisticRegression(max_iter=1000),  #每次執行最大疊代數(max_iter) = 1000，如果執行發散，最大只對計算到 1000 次，type = int
              'Linear' : LinearRegression(),                   #線性回歸，將複雜的資料數據，擬和至一條直線上，就能方便預測未來的資料
              'Svm' : SVC()}                                   #可以做非線性的分類

    def fitting(data_fit):                                               #設立估計函式
        for model in models.values():                                    #讓數據執行上面三個函式
            model.fit(data_fit[cols+sign_list], data_fit['Direction'])   #建立模型，輸入數據(data_fit[cols+sign_list])，標籤為 Direction
            #*科普* fit:函數用以訓練模型，模型有多個輸入，類型為 list，list 的元素對應於各個輸入的 numpy array。
            #          模型的每個輸入都有名字，則可傳入一個字典，將輸入名與其輸入數據對應起來。
            #          模型有多個輸出，可以傳入一個 numpy array 的 list。
            #          模型的輸出擁有名字，則可以傳入一個字典，將輸出名與其標籤對應起來。

    def derive_position(data_next):
        data_pos = pd.DataFrame()                                                                  #建立一個二維表格
        for model in models.keys():                                                                #讀取數據寫進二維表格
            data_pos['Pos_'+model] = np.where(models[model].predict(data_next[cols+sign_list])>0,  #二維表格的 keys 前面 + Pos_
                                              1, -1)                                               #滿足 預測(predict) Lag_? + sign_list > 0 輸出 1，反之 -1
        return data_pos                                                                            #回傳 data_pos 給函數

    for start, end in zip(range(2011, 2017), range(2015, 2021)):                                   #可疊代的數據作為參數，將數據中對應的元素打包成多個元組
        data_filter = data_signal[(data_signal['Date'] >= (str(start)+'/01/01'))&                  #時間間隔 5 年，每一次都抓 5 年的資料去做之後的執行動作，共 16 筆
                                  (data_signal['Date'] <= (str(end)+'/01/01'))].copy()             #淺拷貝，新資料庫 data_filter 為 Data 時間大於開始日期，小於結束日期
        fitting(data_filter)                                                                       #代入到設立估計函式 fitting

        if start == 2011:                                                                               #如果起始值為 2000-01-01
            data_pos = derive_position(data_filter)                                                     #代入到二維表格函式 derive_position.
            data_pos[signal_name] = (1*(data_pos[['Pos_'+model for model in models.keys()]].sum(1)>=2)
                         -1*(data_pos[['Pos_'+model for model in models.keys()]].sum(1)<=-2))
            Signal = data_pos[signal_name]

        data_next = data_signal[(data_signal['Date'] >= (str(end)+'/01/01'))&                           #介於結束值當年-1-1至隔年-1-1的數據
                                (data_signal['Date'] <= (str(end+1)+'/01/01'))]
        data_pos = derive_position(data_next)
        data_pos[signal_name] = (1*(data_pos[['Pos_'+model for model in models.keys()]].sum(1)>=2)
                                 -1*(data_pos[['Pos_'+model for model in models.keys()]].sum(1)<=-2))
        Signal = pd.concat([Signal, data_pos[signal_name]], axis=0)                                     #合併資料庫(concat)，axis=0 為橫向合併

    return pd.concat([data.reset_index(drop=True), Signal.reset_index(drop=True)], axis=1)              #回傳合併(concat)，axis=1為直向合併
#%%
#1.MA

sma = range(5, 61, 10)       #短期
lma = range(20, 281, 20)     #長期

data_ma = pd.DataFrame()     #建立二維表格
ma_list = ['Signal_MA'+'_%d_%d'%(SMA, LMA) for SMA, LMA in product(sma, lma) if SMA<LMA]   #設立一個串列，​裡面存有列表，當短期天數小於長期天數
for SMA, LMA in product(sma, lma):                                       #移動平均計算
    if SMA < LMA:                                                        #條件:短期天數 < 長期天數
        data_ma['SMA'+'%d'%(SMA)] = data['Close'].rolling(SMA).mean()    #快線計算 = 取 Close 的數據以 SMA 設定的值作為標準，進行平均計算(前 4 個 index 會是 NaN )
        data_ma['LMA'+'%d'%(LMA)] = data['Close'].rolling(LMA).mean()    #慢線計算 = 取 Close 的數據以 LMA 設定的值作為標準，進行平均計算(前 19 個 index 會是 NaN )
        #Trading Strareg                                                 #交易策略
        data_ma['Signal_MA'+'_%d_%d'%(SMA, LMA)] = (1*((data_ma['SMA'+'%d'%(SMA)].shift(1)<data_ma['LMA'+'%d'%(LMA)].shift(1))&  #數據往後進行移動一次(shift)
                                                       (data_ma['SMA'+'%d'%(SMA)]>data_ma['LMA'+'%d'%(LMA)])&                    #當隔天快線 < 隔天慢線 & 快線 > 慢線 & 快線 > 隔天快線
                                                       (data_ma['SMA'+'%d'%(SMA)]>data_ma['SMA'+'%d'%(SMA)].shift(1)))           #買入訊號(1)
                                                    -1*((data_ma['SMA'+'%d'%(SMA)].shift(1)>data_ma['LMA'+'%d'%(LMA)].shift(1))& #<<三個條件都滿足>>
                                                        (data_ma['SMA'+'%d'%(SMA)]<data_ma['LMA'+'%d'%(LMA)])&                   #當隔天快線 > 隔天慢線 & 快線 < 慢線 & 快線 < 隔天快線
                                                        (data_ma['SMA'+'%d'%(SMA)]<data_ma['SMA'+'%d'%(SMA)].shift(1))))         #賣出訊號(-1)
data_ma = pd.concat([data['Date'], data[cols], data_ma[ma_list], data['Direction']], axis=1).dropna()                            #回傳合併(concat)，axis=1 為直向合併，去掉含有缺失值的樣本(行)
#%%
#2.MACD
short_ema, long_ema = 12, 26
data_macd = pd.DataFrame()
data_macd['DI'] = (data['High'] + data['Low'] + 2*data['Close'])/4
data_macd['SEMA'] = data_macd['DI'].ewm(span=short_ema, adjust=False).mean()
data_macd['LEMA'] = data_macd['DI'].ewm(span=long_ema, adjust=False).mean()
data_macd['DIF'] = data_macd['SEMA'] - data_macd['LEMA']
for SPAN in range(5, 21):
    data_macd['MACD'+'%d'%(SPAN)] = data_macd['DIF'].ewm(span=SPAN, adjust=False).mean()
    data_macd['OSC'+'%d'%(SPAN)] = data_macd['DIF'] - data_macd['MACD'+'%d'%(SPAN)]
    #Trading Straregy
    data_macd['Signal_MACD'+'%d'%(SPAN)] = (1*((data_macd['OSC'+'%d'%(SPAN)].shift(1)<0)&(data_macd['OSC'+'%d'%(SPAN)]>0))    #買入
                                       -1*((data_macd['OSC'+'%d'%(SPAN)].shift(1)>0)&(data_macd['OSC'+'%d'%(SPAN)]<0)))    #賣出
macd_list = ['Signal_MACD'+'%d'%(SPAN) for SPAN in range(5, 21)]
data_macd = pd.concat([data['Date'], data[cols], data_macd[macd_list], data['Direction']], axis=1).dropna()
#%%
#3.KD

N, K_P, D_P = 9, 3, 3
data_kd = pd.DataFrame()
data_kd['RSV'] = 100*((data['Close']-data['Low'].rolling(N).min())/
                      (data['High'].rolling(N).max()-data['Low'].rolling(N).min()))
K, D = [50 for x in range(N-1)], [50 for x in range(N-1)]
for i in range(8, len(data_kd['RSV'])):
    K.append((K_P-1)/K_P*K[i-1] + 1/K_P*data_kd['RSV'].iloc[i])
    D.append((D_P-1)/D_P*D[i-1] + 1/D_P*K[i])
data_kd = data_kd.assign(K = pd.Series(K).values, D = pd.Series(D).values)

down = range(5, 31)
up = range(70, 96)
for DOWN, UP in product(down, up):
    #Trading Straregy
    data_kd['Signal_KD'+'_%d_%d'%(DOWN, UP)] = (1*((data_kd['K'].shift(1)<data_kd['D'].shift(1))&(data_kd['K']>data_kd['D'])&
                                                (data_kd['K']<DOWN)&(data_kd['D']<DOWN))    #買入
                                             -1*((data_kd['K'].shift(1)>data_kd['D'].shift(1))&(data_kd['K']<data_kd['D'])&
                                                 (data_kd['K']>UP)&(data_kd['D']>UP)))    #賣出
kd_list = ['Signal_KD'+'_%d_%d'%(DOWN, UP) for DOWN, UP in product(down, up)]
data_kd2 = data_kd
data_kd = pd.concat([data['Date'], data[cols], data_kd[kd_list], data['Direction']], axis=1).dropna()
#%%
#4.布林通道

data_bband = data[['Date','Close']]

time_period = 20        #SMA的計算週期(20)
std_factor = 2          #上下標準差
history = []            #每個計算週期所需的價格數據
sma_values = []         #初始化SMA值
upper_band = []         #初始化阻力線價格
lower_band = []         #初始化支撐線價格

for close_price in data_bband['Close']:
    history.append(close_price)
    if len(history) > time_period:    #計算移動平均時週期不>20
        del (history[0])

    sma = np.mean(history)            #將計算的SMA直存入列表
    sma_values.append(sma)

    std = np.sqrt(np.sum((history - sma) ** 2) / len(history))    #計算標準差
    upper_band.append(sma + std_factor * std)
    lower_band.append(sma - std_factor * std)

# 這個寫法直接把list裝進去 pd會跳 warning
#data_bband['sma_values'] = sma_values
#data_bband['upperband'] = upper_band
# 下面這個寫法就不會了
data_bband = data_bband.assign(sma_values = pd.Series(sma_values).values, \
                               upperband = pd.Series(upper_band).values)

down = range(5, 31)
up = range(70, 96)
for DOWN, UP in product(down, up):
    data_bband['Signal_BBAND'+'_%d_%d'%(DOWN, UP)] = (1*((data_kd2['K'] < 25) & (data_kd2['K'] > data_kd2['D']) &
                                                      (data_kd2['K'].shift() < data_kd2['D'].shift()) &
                                                      (data_bband['Close'] <= data_bband['sma_values']))                  #買入
                                                      -1*((data_bband['Close'] < data_bband['upperband']) &
                                                       (data_bband['Close'].shift() > data_bband['upperband'].shift())))  #賣出

bband_list = ['Signal_BBAND'+'_%d_%d'%(DOWN, UP) for DOWN, UP in product(down, up)]
data_bband = pd.concat([data['Date'], data[cols], data_bband[bband_list], data['Direction']], axis=1).dropna()

#%%
#5.ALL

all_list = ma_list+macd_list+kd_list+bband_list
data_all = pd.concat([data['Date'], data[cols], data_ma.iloc[:, 6:-1].copy(),
                      data_macd.iloc[:, 6:-1].copy(), data_kd.iloc[:, 6:-1].copy(),
                      data_bband.iloc[:, 6:-1].copy(),data['Direction']], axis=1).dropna()
#%%
data = model_fit(data, data_ma, 'Signal_MA', ma_list)
data = model_fit(data, data_macd, 'Signal_MACD', macd_list)
data = model_fit(data, data_kd, 'Signal_KD', kd_list)
data = model_fit(data, data_bband, 'Signal_BBAND', bband_list)
data = model_fit(data, data_all, 'Signal_ALL', all_list)
data['CumReturn_MA'] = np.cumsum(data['DayReturn'][1:]*data['Signal_MA'][:-1].values)
data['CumReturn_MACD'] = np.cumsum(data['DayReturn'][1:]*data['Signal_MACD'][:-1].values)
data['CumReturn_KD'] = np.cumsum(data['DayReturn'][1:]*data['Signal_KD'][:-1].values)
data['CumReturn_BBAND'] = np.cumsum(data['DayReturn'][1:]*data['Signal_BBAND'][:-1].values)
data['CumReturn_ALL'] = np.cumsum(data['DayReturn'][1:]*data['Signal_ALL'][:-1].values)
data['BuyHold'] = np.cumsum(data['DayReturn'][1:])
data = data[(data['Date'] >= start_date_plot) & (data['Date'] <= end_date_plot)]
data = data.set_index('Date')
#%%
#畫圖
fig = plt.figure(figsize=(30, 15))
ax4 = plt.subplot(111)
ax3 = ax4.twinx()
ax3.plot(data['Close'], 'k', linewidth = 2, alpha = 0.5, label = '收盤價')
ax3.legend(prop = my_font, loc = 'lower left')
ax4.plot(data['CumReturn_MA'], label = 'MA累積報酬')
ax4.plot(data['CumReturn_MACD'], label = 'MACD累積報酬')
ax4.plot(data['CumReturn_KD'], label = 'KD累積報酬')
ax4.plot(data['CumReturn_BBAND'], label = 'Bollinger Band累積報酬')
ax4.plot(data['CumReturn_ALL'], label = 'ALL累積報酬')
ax4.plot(data['BuyHold'], label = 'Buy&Hold')
ax4.axvline(x='2005/01/03', color='black')
ax4.set_xticks(range(0, len(data.index), interval))
ax4.set_xticklabels(data.index[::interval], fontsize=18)
plt.yticks(fontsize=18)
ax4.set_title('Stock:'+str(stock)+'累積報酬', fontproperties = my_font, fontsize=30)
ax4.legend(prop = my_font, loc = 'upper left')
ax4.text(data.index[-1], data['CumReturn_MA'][-1], '%.2f%%'%(data['CumReturn_MA'][-1]*100), fontsize=20)
ax4.text(data.index[-1], data['CumReturn_MACD'][-1], '%.2f%%'%(data['CumReturn_MACD'][-1]*100), fontsize=20)
ax4.text(data.index[-1], data['CumReturn_KD'][-1], '%.2f%%'%(data['CumReturn_KD'][-1]*100), fontsize=20)
ax4.text(data.index[-1], data['CumReturn_BBAND'][-1], '%.2f%%'%(data['CumReturn_BBAND'][-1]*100), fontsize=20)
ax4.text(data.index[-1], data['CumReturn_ALL'][-1], '%.2f%%'%(data['CumReturn_ALL'][-1]*100), fontsize=20)
ax4.text(data.index[-1], data['BuyHold'][-1], '%.2f%%'%(data['BuyHold'][-1]*100), fontsize=20)
fig.savefig(image_name)

# 把結果設成一個變數，比較好擷取
performance = data.iloc[-1][['CumReturn_MA','CumReturn_MACD','CumReturn_KD',\
                             'CumReturn_BBAND','CumReturn_ALL','BuyHold']]*100
print(performance)