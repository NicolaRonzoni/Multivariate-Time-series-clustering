#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:35:16 2021

@author: nicolaronzoni
"""

import pandas as pd 
import scipy 
import sklearn
import tslearn 
import numpy as np
import random

#########TRAIN DATA##########
#import  train data
df= pd.read_csv("/Users/nronzoni/Downloads/Multivariate-Time-series-clustering-main/6min data&code/S1816 6min.csv") 
df
#treatment of flow variables 
flow=df.loc[:,'Flow']
#normalization/standardization of train data 
flow=np.array(flow)
flow= flow.reshape((len(flow), 1))
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler(feature_range=(0, 1))
#scaler=StandardScaler()
#fit train data 
scaler_flow_train = scaler.fit(flow)
print('Min: %f, Max: %f' % (scaler_flow_train.data_min_, scaler_flow_train.data_max_))
#scale train data 
normalized_flow=scaler_flow_train.transform(flow)
normalized_flow
#from array to list 
normalized_flow=normalized_flow.tolist()
len(normalized_flow)
from toolz.itertoolz import sliding_window, partition
#for every day of the train set store the flow observations 
day_flow=list(partition(240,normalized_flow))
day_flow
len(day_flow)
#from list to multidimensional array 
day_flow=np.asarray(day_flow)
day_flow
from tslearn.utils import to_time_series, to_time_series_dataset
#create univariate series for normalized flow_observation 
first_time_series = to_time_series(day_flow)
print(first_time_series.shape)

#treatment of density variable 
density =df.loc[:,'Density']
#normalization/standardization of train data 
density=np.array(density)
density= density.reshape((len(density), 1))
#fit train data
scaler_density_train = scaler.fit(density)
print('Min: %f, Max: %f' % (scaler_density_train.data_min_, scaler_density_train.data_max_))
#scale train data 
normalized_density = scaler_density_train.transform(density)
normalized_density
#from array to list 
normalized_density=normalized_density.tolist()
len(normalized_density)
#for every day of the train set store the speed observations 
day_density=list(partition(240,normalized_density))
day_density
len(day_density)
#from list to multidimensionalarray
day_density=np.asarray(day_density)
day_density
#create univariate series for normalized speed observation 
second_time_series = to_time_series(day_density)
print(second_time_series.shape)

#create the multivariate time series TRAIN  
multivariate=np.dstack((first_time_series,second_time_series))
multivariate_time_series_train = to_time_series(multivariate)
print(multivariate_time_series_train.shape)

#######TEST DATA ########
#import  test data
df_test= pd.read_csv("/Users/nronzoni/Downloads/Multivariate-Time-series-clustering-main/I35W_NB 6min 2013/S1816 6min.csv") 
df_test
#treatment of flow variables 
flow_test=df_test.loc[:,'Flow']
#normalization/standardization of train data 
flow_test=np.array(flow_test)
flow_test= flow_test.reshape((len(flow_test), 1))

#fit test data 
scaler_flow_test = scaler.fit(flow_test)
print('Min: %f, Max: %f' % (scaler_flow_test.data_min_, scaler_flow_test.data_max_))
#scale test data 
normalized_flow_test=scaler_flow_test.transform(flow_test)
normalized_flow_test
#from array to list 
normalized_flow_test=normalized_flow_test.tolist()
len(normalized_flow_test)

#for every day of the test set store the flow observations 
day_flow_test=list(partition(240,normalized_flow_test))
day_flow_test
len(day_flow_test)
#from list to multidimensional array 
day_flow_test=np.asarray(day_flow_test)
day_flow_test
#create univariate series for normalized flow_observations of the test set 
first_time_series_test = to_time_series(day_flow_test)
print(first_time_series_test.shape)

#treatment of density variable 
density_test =df_test.loc[:,'Density']
#normalization/standardization of test data 
density_test=np.array(density_test)
density_test= density_test.reshape((len(density_test), 1))
#fit test data
scaler_density_test = scaler.fit(density_test)
print('Min: %f, Max: %f' % (scaler_density_test.data_min_, scaler_density_test.data_max_))
#scale train data 
normalized_density_test = scaler_density_test.transform(density_test)
normalized_density_test
#from array to list 
normalized_density_test=normalized_density_test.tolist()
len(normalized_density_test)
#for every day of the test set store the speed observations 
day_density_test=list(partition(240,normalized_density_test))
day_density_test
len(day_density_test)
#from list to multidimensionalarray
day_density_test=np.asarray(day_density_test)
day_density_test
#create univariate series for normalized speed observations of the test set  
second_time_series_test = to_time_series(day_density_test)
print(second_time_series_test.shape)

#create the multivariate time series TEST 
multivariate_test=np.dstack((first_time_series_test,second_time_series_test))
multivariate_time_series_test = to_time_series(multivariate_test)
print(multivariate_time_series_test.shape)


#clustering 
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, silhouette_score
#fit the algorithm on train data 
#tune the hyperparameters possible metrics: euclidean, dtw, softdtw
km_dba = TimeSeriesKMeans(n_clusters=4, metric="softdtw", max_iter=5,max_iter_barycenter=5, random_state=0).fit(multivariate_time_series_train)
km_dba.cluster_centers_.shape
#prediction on train data 
prediction_train= km_dba.fit_predict(multivariate_time_series_train,y=None)
len(prediction_train)
#prediction on test data 
prediction_test= km_dba.predict(multivariate_time_series_test)
len(prediction_test)
prediction_test

#accuracy of the clustering on the train data 
silhouette_score(multivariate_time_series_train, prediction_train, metric="softdtw")
#accuracy of the clustering on the test data
silhouette_score(multivariate_time_series_test, prediction_test, metric="softdtw")

############################################ k=2 #########################################
#select randomly time series from first cluster 

cluster1=multivariate_time_series_train[prediction_train==0]

random.shuffle(cluster1)

sample1=cluster1[50:65]

sample1.shape

sample1flow=sample1[:,:,0]
sample1flow=scaler_flow_train.inverse_transform(sample1flow)
sample1flow

sample1density=sample1[:,:,1]
sample1density=scaler_density_train.inverse_transform(sample1density)
sample1density

#select randomly time series from second  cluster 

cluster2=multivariate_time_series_train[prediction_train==1]

random.shuffle(cluster2)



sample2=cluster2[50:65]

sample2.shape

sample2flow=sample2[:,:,0]
sample2flow=scaler_flow_train.inverse_transform(sample2flow)
sample2flow

sample2density=sample2[:,:,1]
sample2density=scaler_density_train.inverse_transform(sample2density)


#plot centroids k=2 
import matplotlib.pyplot as plt
x= np.arange(0,24,0.1)
centroids=km_dba.cluster_centers_
centroids.shape
flow_1=centroids[0][:,0]
flow_1= flow_1.reshape((len(flow_1), 1))
flow_2=centroids[1][:,0]
flow_2= flow_2.reshape((len(flow_2), 1))
density_1=centroids[0][:,1]
density_1= density_1.reshape((len(density_1), 1))
density_2=centroids[1][:,1]
density_2= density_2.reshape((len(density_2), 1))

plt.figure(figsize=(15,15))
plt.title('centroids path')
plt.subplot(2,2,1)
for i in range(0,15):
    plt.plot(x,sample1flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_1),'r-', label = 'flow')
plt.xlabel(' hours of the day')
plt.ylabel('flow')
plt.title(' k=0')
plt.legend()
plt.subplot(2,2,2)
for i in range(0,15):
    plt.plot(x,sample1density[i],'k-', alpha=.2)
plt.plot(x,scaler_density_train.inverse_transform(density_1),'r-', label = 'density')
plt.xlabel(' hours of the day')
plt.ylabel('density')
plt.title('k=0')
plt.legend()
plt.subplot(2,2,3)
for i in range(0,15):
    plt.plot(x,sample2flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_2),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title(' k=1')
plt.legend()
plt.subplot(2,2,4)
for i in range(0,15):
    plt.plot(x,sample2density[i],'k-', alpha=.2)
plt.plot(x,scaler_density_train.inverse_transform(density_2),'r-', label = 'density')
plt.xlabel('hours of the day')
plt.ylabel('density')
plt.title('k=1')
plt.legend()
plt.show()

########################################### k=3 ################################
#select randomly time series from first cluster 

cluster1=multivariate_time_series_train[prediction_train==0]

random.shuffle(cluster1)

sample1=cluster1[50:65]

sample1.shape

sample1flow=sample1[:,:,0]
sample1flow=scaler_flow_train.inverse_transform(sample1flow)
sample1flow

sample1density=sample1[:,:,1]
sample1density=scaler_density_train.inverse_transform(sample1density)
sample1density

#select randomly time series from second  cluster 

cluster2=multivariate_time_series_train[prediction_train==1]


random.shuffle(cluster2)

sample2=cluster2[50:65]

sample2.shape

sample2flow=sample2[:,:,0]
sample2flow=scaler_flow_train.inverse_transform(sample2flow)
sample2flow

sample2density=sample2[:,:,1]
sample2density=scaler_density_train.inverse_transform(sample2density)

#select randomly time series from third cluster 

cluster3=multivariate_time_series_train[prediction_train==2]

random.shuffle(cluster3)

sample3=cluster3[50:65]

sample3.shape

sample3flow=sample3[:,:,0]
sample3flow=scaler_flow_train.inverse_transform(sample3flow)
sample3flow

sample3density=sample3[:,:,1]
sample3density=scaler_density_train.inverse_transform(sample3density)


# plot the centroids k=3 
centroids=km_dba.cluster_centers_
centroids.shape
flow_1=centroids[0][:,0]
flow_1= flow_1.reshape((len(flow_1), 1))
flow_2=centroids[1][:,0]
flow_2= flow_2.reshape((len(flow_2), 1))
flow_3=centroids[2][:,0]
flow_3= flow_3.reshape((len(flow_3), 1))
density_1=centroids[0][:,1]
density_1= density_1.reshape((len(density_1), 1))
density_2=centroids[1][:,1]
density_2= density_2.reshape((len(density_2), 1))
density_3=centroids[2][:,1]
density_3= density_3.reshape((len(density_3), 1))
plt.figure(figsize=(15,15))
plt.subplot(3,2,1)
for i in range(0,15):
    plt.plot(x,sample1flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_1),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=0')
plt.legend()
plt.subplot(3,2,2)
for i in range(0,15):
    plt.plot(x,sample1density[i],'k-', alpha=.2)
plt.plot(x,scaler_density_train.inverse_transform(density_1),'r-', label = 'density')
plt.xlabel('hours of the day')
plt.ylabel('density')
plt.title(' k=0')
plt.legend()
plt.subplot(3,2,3)
for i in range(0,15):
    plt.plot(x,sample2flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_2),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=1')
plt.legend()
plt.subplot(3,2,4)
for i in range(0,15):
    plt.plot(x,sample2density[i],'k-', alpha=.2)
plt.plot(x,scaler_density_train.inverse_transform(density_2),'r-', label = 'density')
plt.xlabel('hours of the day')
plt.ylabel('density')
plt.title('k=1')
plt.legend()
plt.subplot(3,2,5)
for i in range(0,15):
    plt.plot(x,sample3flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_3),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=2')
plt.legend()
plt.subplot(3,2,6)
for i in range(0,15):
    plt.plot(x,sample3density[i],'k-', alpha=.2)
plt.plot(x,scaler_density_train.inverse_transform(density_3),'r-', label = 'density')
plt.xlabel('hours of the day')
plt.ylabel('density')
plt.title('k=2')
plt.legend()
plt.show()

###################################### k=4 ######################@@@@@###########
#select randomly time series from first cluster 

cluster1=multivariate_time_series_train[prediction_train==0]

random.shuffle(cluster1)

sample1=cluster1[0:7]

sample1.shape

sample1flow=sample1[:,:,0]
sample1flow=scaler_flow_train.inverse_transform(sample1flow)
sample1flow

sample1density=sample1[:,:,1]
sample1density=scaler_density_train.inverse_transform(sample1density)
sample1density

#select randomly time series from second  cluster 

cluster2=multivariate_time_series_train[prediction_train==1]


random.shuffle(cluster2)

sample2=cluster2[0:7]

sample2.shape

sample2flow=sample2[:,:,0]
sample2flow=scaler_flow_train.inverse_transform(sample2flow)
sample2flow

sample2density=sample2[:,:,1]
sample2density=scaler_density_train.inverse_transform(sample2density)

#select randomly time series from third cluster 

cluster3=multivariate_time_series_train[prediction_train==2]

random.shuffle(cluster3)

sample3=cluster3[0:7]

sample3.shape

sample3flow=sample3[:,:,0]
sample3flow=scaler_flow_train.inverse_transform(sample3flow)
sample3flow

sample3density=sample3[:,:,1]
sample3density=scaler_density_train.inverse_transform(sample3density)

#select randomly time series from fourth cluster 

cluster4=multivariate_time_series_train[prediction_train==3]

random.shuffle(cluster4)

sample4=cluster4[0:7]

sample4.shape

sample4flow=sample4[:,:,0]
sample4flow=scaler_flow_train.inverse_transform(sample4flow)
sample4flow

sample4density=sample4[:,:,1]
sample4density=scaler_density_train.inverse_transform(sample4density)



# plot the centroids K=4 
centroids=km_dba.cluster_centers_
centroids.shape
flow_1=centroids[0][:,0]
flow_1= flow_1.reshape((len(flow_1), 1))
flow_2=centroids[1][:,0]
flow_2= flow_2.reshape((len(flow_2), 1))
flow_3=centroids[2][:,0]
flow_3= flow_3.reshape((len(flow_3), 1))
flow_4=centroids[3][:,0]
flow_4= flow_4.reshape((len(flow_4), 1))
density_1=centroids[0][:,1]
density_1= density_1.reshape((len(density_1), 1))
density_2=centroids[1][:,1]
density_2= density_2.reshape((len(density_2), 1))
density_3=centroids[2][:,1]
density_3= density_3.reshape((len(density_3), 1))
density_4=centroids[3][:,1]
density_4= density_4.reshape((len(density_4), 1))
plt.figure(figsize=(20,30))
plt.subplot(4,2,1)
for i in range(0,7):
    plt.plot(x,sample1flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_1),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=0')
plt.legend()
plt.subplot(4,2,2)
for i in range(0,7):
    plt.plot(x,sample1density[i],'k-', alpha=.2)
plt.plot(x,scaler_density_train.inverse_transform(density_1),'r-', label = 'density')
plt.xlabel('hours of the day')
plt.ylabel('density')
plt.title('k=0')
plt.legend()
plt.subplot(4,2,3)
for i in range(0,7):
    plt.plot(x,sample2flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_2),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=1')
plt.legend()
plt.subplot(4,2,4)
for i in range(0,7):
    plt.plot(x,sample2density[i],'k-', alpha=.2)
plt.plot(x,scaler_density_train.inverse_transform(density_2),'r-', label = 'density')
plt.xlabel('hours of the day')
plt.ylabel('density')
plt.title('k=1')
plt.legend()
plt.subplot(4,2,5)
for i in range(0,7):
    plt.plot(x,sample3flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_3),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=2')
plt.legend()
plt.subplot(4,2,6)
for i in range(0,7):
    plt.plot(x,sample3density[i],'k-', alpha=.2)
plt.plot(x,scaler_density_train.inverse_transform(density_3),'r-', label = 'density')
plt.xlabel('hours of the day')
plt.ylabel('density')
plt.title('k=2')
plt.legend()
plt.subplot(4,2,7)
for i in range(0,7):
    plt.plot(x,sample4flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_4),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=3')
plt.legend()
plt.subplot(4,2,8)
for i in range(0,7):
    plt.plot(x,sample4density[i],'k-', alpha=.2)
plt.plot(x,scaler_density_train.inverse_transform(density_4),'r-', label = 'density')
plt.xlabel('hours of the day')
plt.ylabel('density')
plt.title('k=3')
plt.legend()
plt.show()


###################################### k=5 ######################@@@@@###########
#select randomly time series from first cluster 

cluster1=multivariate_time_series_train[prediction_train==0]

random.shuffle(cluster1)

sample1=cluster1[5:20]

sample1.shape

sample1flow=sample1[:,:,0]
sample1flow=scaler_flow_train.inverse_transform(sample1flow)
sample1flow

sample1density=sample1[:,:,1]
sample1density=scaler_density_train.inverse_transform(sample1density)
sample1density

#select randomly time series from second  cluster 

cluster2=multivariate_time_series_train[prediction_train==1]


random.shuffle(cluster2)

sample2=cluster2[0:15]

sample2.shape

sample2flow=sample2[:,:,0]
sample2flow=scaler_flow_train.inverse_transform(sample2flow)
sample2flow.shape

sample2density=sample2[:,:,1]
sample2density=scaler_density_train.inverse_transform(sample2density)

#select randomly time series from third cluster 

cluster3=multivariate_time_series_train[prediction_train==2]

random.shuffle(cluster3)

sample3=cluster3[0:15]

sample3.shape

sample3flow=sample3[:,:,0]
sample3flow=scaler_flow_train.inverse_transform(sample3flow)
sample3flow.shape

sample3density=sample3[:,:,1]
sample3density=scaler_density_train.inverse_transform(sample3density)

#select randomly time series from fourth cluster 

cluster4=multivariate_time_series_train[prediction_train==3]

random.shuffle(cluster4)

sample4=cluster4[5:25]

sample4.shape

sample4flow=sample4[:,:,0]
sample4flow=scaler_flow_train.inverse_transform(sample4flow)
sample4flow

sample4density=sample4[:,:,1]
sample4density=scaler_density_train.inverse_transform(sample4density)

#select randomly time series from fifth cluster 

cluster5=multivariate_time_series_train[prediction_train==4]

random.shuffle(cluster5)

sample5=cluster5[5:25]

sample5.shape

sample5flow=sample5[:,:,0]
sample5flow=scaler_flow_train.inverse_transform(sample5flow)
sample5flow

sample5density=sample5[:,:,1]
sample5density=scaler_density_train.inverse_transform(sample5density)



# plot the centroids K=5 
centroids=km_dba.cluster_centers_
centroids.shape
flow_1=centroids[0][:,0]
flow_1= flow_1.reshape((len(flow_1), 1))
flow_2=centroids[1][:,0]
flow_2= flow_2.reshape((len(flow_2), 1))
flow_3=centroids[2][:,0]
flow_3= flow_3.reshape((len(flow_3), 1))
flow_4=centroids[3][:,0]
flow_4= flow_4.reshape((len(flow_4), 1))
flow_5=centroids[4][:,0]
flow_5= flow_5.reshape((len(flow_5), 1))

density_1=centroids[0][:,1]
density_1= density_1.reshape((len(density_1), 1))
density_2=centroids[1][:,1]
density_2= density_2.reshape((len(density_2), 1))
density_3=centroids[2][:,1]
density_3= density_3.reshape((len(density_3), 1))
density_4=centroids[3][:,1]
density_4= density_4.reshape((len(density_4), 1))
density_5=centroids[4][:,1]
density_5= density_5.reshape((len(density_5), 1))
plt.figure(figsize=(20,30))
plt.subplot(5,2,1)
for i in range(0,15):
    plt.plot(x,sample1flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_1),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=0')
plt.legend()
plt.subplot(5,2,2)
for i in range(0,15):
    plt.plot(x,sample1density[i],'k-', alpha=.2)
plt.plot(x,scaler_density_train.inverse_transform(density_1),'r-', label = 'density')
plt.xlabel('hours of the day')
plt.ylabel('density')
plt.title('k=0')
plt.legend()
plt.subplot(5,2,3)
for i in range(0,15):
    plt.plot(x,sample2flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_2),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=1')
plt.legend()
plt.subplot(5,2,4)
for i in range(0,15):
    plt.plot(x,sample2density[i],'k-', alpha=.2)
plt.plot(x,scaler_density_train.inverse_transform(density_2),'r-', label = 'density')
plt.xlabel('hours of the day')
plt.ylabel('density')
plt.title('k=1')
plt.legend()
plt.subplot(5,2,5)
for i in range(0,15):
    plt.plot(x,sample3flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_3),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=2')
plt.legend()
plt.subplot(5,2,6)
for i in range(0,5):
    plt.plot(x,sample3density[i],'k-', alpha=.2)
plt.plot(x,scaler_density_train.inverse_transform(density_3),'r-', label = 'density')
plt.xlabel('hours of the day')
plt.ylabel('density')
plt.title('k=2')
plt.legend()
plt.subplot(5,2,7)
for i in range(0,15):
    plt.plot(x,sample4flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_4),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=3')
plt.legend()
plt.subplot(5,2,8)
for i in range(0,15):
    plt.plot(x,sample4density[i],'k-', alpha=.2)
plt.plot(x,scaler_density_train.inverse_transform(density_4),'r-', label = 'density')
plt.xlabel('hours of the day')
plt.ylabel('density')
plt.title('k=3')
plt.subplot(5,2,9)
for i in range(0,15):
    plt.plot(x,sample5flow[i],'k-', alpha=.2)
plt.plot(x,scaler_flow_train.inverse_transform(flow_5),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=4')
plt.legend()
plt.subplot(5,2,10)
for i in range(0,15):
    plt.plot(x,sample5density[i],'k-', alpha=.2)
plt.plot(x,scaler_density_train.inverse_transform(density_5),'r-', label = 'density')
plt.xlabel('hours of the day')
plt.ylabel('density')
plt.title('k=4')
plt.legend()
plt.show()



#similarity between centroids of the clusters 
from tslearn.metrics import soft_dtw, cdist_soft_dtw
similarity=[]
matrix=cdist_soft_dtw(centroids, gamma=1.)
matrix
sim=matrix.max()
similarity.append(sim)

similarity=np.array(similarity)
diss=list(-similarity)

cluster=np.arange(2,8)
plt.title('soft-DTW similarity measure S60')
plt.plot(cluster,diss)
plt.xlabel('n° of cluster')
plt.ylabel('similarity between closest clusters')
plt.show()

#visualization 
import calplot
#all days of 2013 
all_days = pd.date_range('1/1/2013', periods=365, freq='D')
#assign at every day the cluster 
events_train = pd.Series(prediction_train, index=all_days)
#plot the result 
calplot.calplot(events_train,yearlabel_kws={'color': 'black'}, cmap='cool', suptitle='Clustering of the days train set S60', linewidth=2.3 )  

#test days 2013
events_test = pd.Series(prediction_test, index=all_days)
calplot.calplot(events_test,yearlabel_kws={'color': 'black'}, cmap='cool', suptitle='Clustering of the days test set S1816', linewidth=2.3)  
prediction_test




#test days 2014
before= np.full(shape=90,fill_value=4,dtype=np.int)
before
after= np.full(shape=153,fill_value=4,dtype=np.int)
after
#concatenate arrays 
test=np.concatenate((before, prediction_test,after))
len(test)
test_days=pd.date_range('1/1/2014', periods=365, freq='D')
events_test = pd.Series(test, index=test_days)
#plot the result 
calplot.calplot(events_test) 
prediction_test
prediction_train


#dissimilarity 
[-405.6441181518861,
 -402.4074137016698,
 -401.6816786141743,
 -401.40418224504447,
 -398.8178130168854,
 -396.4210957099466,
 -394.54693366714804,
 -378.0648034474099,
 -386.8063417099886]
