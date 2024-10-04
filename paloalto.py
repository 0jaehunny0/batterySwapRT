import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('ChargePoint Data CY20Q4.csv', engine='python')

data["time"] = pd.to_datetime(data["Start Date"])

data["chargingTime"] = pd.to_datetime(data["Charging Time (hh:mm:ss)"])

siteLi = data["Station Name"].unique()
for i in range(len(siteLi)): print(len(data[data["Station Name"] == siteLi[i]]))

k=6

bin = np.arange(-10, 1440, 20)
tempModel = data[(data["Station Name"] == siteLi[k])]["Model Number"].unique()

import matplotlib

font = {'family' : 'normal',
        'size'   : 23}
matplotlib.rc('font', **font)

from matplotlib import font_manager

font_path = 'C:/Users/jaehunny/Downloads/font/LinBiolinum_Rah.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()

font_dirs = ['C:/Users/jaehunny/Downloads/font']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

csfont = {'fontname':"Linux Biolinum"}


k = 6
plt.figure(figsize= (14.4, 3.6))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
bin = np.arange(-10, 1440, 20)
x = data[(data["Station Name"] == siteLi[k]) & (data["Model Number"] ==tempModel[1])]
x = x.sort_values(by = "time")
timediff  = x["time"].diff()
xxx = []
for i in timediff:
    xxx.append(i.seconds/60)
weight = np.ones_like(xxx[1:]) / len(xxx[1:]) * 100

a = ax1.hist(xxx[1:], bins = bin, weights = weight,  label="Model: "+tempModel[1])
ax1.legend()
x = data[(data["Station Name"] == siteLi[k]) & (data["Model Number"] ==tempModel[2])]
x = x.sort_values(by = "time")
timediff  = x["time"].diff()
xxx = []
for i in timediff:
    xxx.append(i.seconds/60)
weight = np.ones_like(xxx[1:]) / len(xxx[1:]) * 100

a = ax2.hist(xxx[1:], bins = bin, weights = weight,  label="Model: "+tempModel[2], color="C1")
ax2.legend()
ax1.set_xlabel("inter-arrival time (m)")
ax2.set_xlabel("inter-arrival time (m)")
ax1.set_ylabel("probability (%)")
plt.tight_layout(pad = 0.2)
plt.show()


import pickle as pk

x = data[(data["Station Name"] == siteLi[k]) & (data["Model Number"] ==tempModel[1])]
x = x.sort_values(by = "time")
timediff  = x["time"].diff()
xxx = []
for i in timediff:
    xxx.append(i.seconds/60)

np.median(np.array(xxx[1:]))

with open(file='model1.pickle', mode='wb') as f:
    pk.dump(xxx[1:], f)


x = data[(data["Station Name"] == siteLi[k]) & (data["Model Number"] ==tempModel[2])]
x = x.sort_values(by = "time")
timediff  = x["time"].diff()
xxx = []
for i in timediff:
    xxx.append(i.seconds/60)
with open(file='model2.pickle', mode='wb') as f:
    pk.dump(xxx[1:], f)


with open(file='model1.pickle', mode='rb') as f:
    model1=pk.load(f)

with open(file='model2.pickle', mode='rb') as f:
    model2=pk.load(f)
