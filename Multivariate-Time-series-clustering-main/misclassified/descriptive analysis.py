#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 10:27:48 2021

@author: nicolaronzoni
"""

import pandas as pd 
import numpy as np

df=pd.read_csv("/Users/nicolaronzoni/Downloads/I35W_NB 30min 2013/Densitydetectors.csv")
df

#december 
#s60=df.loc[-1488:,'S60']
s60=df['S60'][-1488:]
s60=np.array(s60)
len(s60)

s124=df['S124'][-1488:]
s124=np.array(s124)
len(s124)

s1816=df['S1816'][-1488:]
s1816=np.array(s1816)
len(s1816)

#november 
s60=df['S60'][-2928:-1488]
s60=np.array(s60)
len(s60)

s124=df['S124'][-2928:-1488]
s124=np.array(s124)
len(s124)

s1816=df['S1816'][-2928:-1488]
s1816=np.array(s1816)
len(s1816)

#June
s60=df['S60'][7248:8688]
s60=np.array(s60)
len(s60)

s124=df['S124'][7248:8688]
s124=np.array(s124)
len(s124)

s1816=df['S1816'][7248:8688]
s1816=np.array(s1816)
len(s1816)

#from 03/06/2013 to 10/06/2013
s60=df['S60'][7391:7727]
s60=np.array(s60)
len(s60)

#from 05/03/2013 to 18/03/2013
s60=df['S60f'][3647:3982]
s60=np.array(s60)
len(s60)

s60=df['S60f'][2999:3311]
s60=np.array(s60)
len(s60)



s1816=df['S1816'][1967:2302]
s1816=np.array(s1816)
len(s1816)



# datetime index
#December
index=pd.date_range("2013-12-01",periods=1488,freq="30min")
index
#November 
index=pd.date_range("2013-11-01",periods=1440,freq="30min")
index
#June 
index=pd.date_range("2013-06-01",periods=1440,freq="30min")
index

#03-06-2013 10-06-2013 
index=pd.date_range("2013-06-04",periods=336,freq="30min")
index

#from 04/03/2013 to 10/03/2013
index=pd.date_range("2013-03-04",periods=312,freq="30min")
index

#from 04/03/2013 to 10/03/2013
index=pd.date_range("2013-02-11",periods=335,freq="30min")
index


#s60 detector 
ts_s60=pd.Series(data=s60,index=index)
ts_s60

#s124 detector 
ts_s124=pd.Series(data=s124,index=index)
ts_s124

#s1816 detector 
ts_s1816=pd.Series(data=s1816,index=index)
ts_s1816





import matplotlib.pyplot as plt

#s60
ts_s60.plot(title='flow of S60 detector')
plt.show

#s124
ts_s124.plot(title='density of S124 detector')
plt.show

#s1816 
ts_s1816.plot(title='density of S1816 detector')
plt.show




