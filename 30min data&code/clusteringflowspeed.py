#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 18:45:12 2021

@author: nicolaronzoni
"""
import pandas as pd 
import scipy 
import sklearn
import tslearn 
import numpy as np

#########TRAIN DATA##########
#import  train data
df= pd.read_csv("/Users/nronzoni/Downloads/Multivariate-Time-series-clustering-main/I35W_NB 30min 2013/S1816 30min.csv") 
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
#print('Min: %f, Max: %f' % (scaler_flow_train.data_min_, scaler_flow_train.data_max_))
#scale train data 
normalized_flow=scaler_flow_train.transform(flow)
normalized_flow
#from array to list 
normalized_flow=normalized_flow.tolist()
len(normalized_flow)
from toolz.itertoolz import sliding_window, partition
#for every day of the train set store the flow observations 
day_flow=list(partition(48,normalized_flow))
day_flow
len(day_flow)
#from list to multidimensional array 
day_flow=np.asarray(day_flow)
day_flow
from tslearn.utils import to_time_series, to_time_series_dataset
#create univariate series for normalized flow_observation 
first_time_series = to_time_series(day_flow)
print(first_time_series.shape)

#treatment of speed variable 
speed =df.loc[:,'Speed']
#normalization/standardization of train data 
speed=np.array(speed)
speed= speed.reshape((len(speed), 1))
#fit train data
scaler_speed_train = scaler.fit(speed)
print('Min: %f, Max: %f' % (scaler_speed_train.data_min_, scaler_speed_train.data_max_))
#scale train data 
normalized_speed = scaler_speed_train.transform(speed)
normalized_speed
#from array to list 
normalized_speed=normalized_speed.tolist()
len(normalized_speed)
#for every day of the train set store the speed observations 
day_speed=list(partition(48,normalized_speed))
day_speed
len(day_speed)
#from list to multidimensionalarray
day_speed=np.asarray(day_speed)
day_speed
#create univariate series for normalized speed observation 
second_time_series = to_time_series(day_speed)
print(second_time_series.shape)

#create the multivariate time series TRAIN  
multivariate=np.dstack((first_time_series,second_time_series))
multivariate_time_series_train = to_time_series(multivariate)
print(multivariate_time_series_train.shape)

#######TEST DATA ########
#import  test data
df_test= pd.read_csv("/Users/nronzoni/Downloads/Multivariate-Time-series-clustering-main/I35W_NB 30min 2013/S60.csv") 
df_test
#treatment of flow variables 
flow_test=df_test.loc[:,'Flow']
#normalization/standardization of train data 
flow_test=np.array(flow_test)
flow_test= flow_test.reshape((len(flow_test), 1))

#fit test data 
scaler_flow_test = scaler.fit(flow_test)
#print('Min: %f, Max: %f' % (scaler_flow_test.data_min_, scaler_flow_test.data_max_))
#scale test data 
normalized_flow_test=scaler_flow_test.transform(flow_test)
normalized_flow_test
#from array to list 
normalized_flow_test=normalized_flow_test.tolist()
len(normalized_flow_test)

#for every day of the test set store the flow observations 
day_flow_test=list(partition(48,normalized_flow_test))
day_flow_test
len(day_flow_test)
#from list to multidimensional array 
day_flow_test=np.asarray(day_flow_test)
day_flow_test
#create univariate series for normalized flow_observations of the test set 
first_time_series_test = to_time_series(day_flow_test)
print(first_time_series_test.shape)

#treatment of speed variable 
speed_test =df_test.loc[:,'Speed']
#normalization/standardization of test data 
speed_test=np.array(speed_test)
speed_test= speed_test.reshape((len(speed_test), 1))
#fit test data
scaler_speed_test = scaler.fit(speed_test)
print('Min: %f, Max: %f' % (scaler_speed_test.data_min_, scaler_speed_test.data_max_))
#scale train data 
normalized_speed_test = scaler_speed_test.transform(speed_test)
normalized_speed_test
#from array to list 
normalized_speed_test=normalized_speed_test.tolist()
len(normalized_speed_test)
#for every day of the test set store the speed observations 
day_speed_test=list(partition(48,normalized_speed_test))
day_speed_test
len(day_speed_test)
#from list to multidimensionalarray
day_speed_test=np.asarray(day_speed_test)
day_speed_test
#create univariate series for normalized speed observations of the test set  
second_time_series_test = to_time_series(day_speed_test)
print(second_time_series_test.shape)

#create the multivariate time series TEST 
multivariate_test=np.dstack((first_time_series_test,second_time_series_test))
multivariate_time_series_test = to_time_series(multivariate_test)
print(multivariate_time_series_test.shape)


#clustering 
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, silhouette_score
#fit the algorithm on train data 
#tune the hyperparameters possible metrics: euclidean, dtw, softdtw
km_dba = TimeSeriesKMeans(n_clusters=2, metric="softdtw", max_iter=5,max_iter_barycenter=5, random_state=0).fit(multivariate_time_series_train)
km_dba.cluster_centers_.shape
#prediction on train data 
prediction_train= km_dba.fit_predict(multivariate_time_series_train,y=None)
len(prediction_train)
#prediction on test data 
prediction_test= km_dba.fit_predict(multivariate_time_series_test,y=None)
len(prediction_test)
prediction_test

#accuracy of the clustering on the train data 
silhouette_score(multivariate_time_series_train, prediction_train, metric="softdtw")
#accuracy of the clustering on the test data
silhouette_score(multivariate_time_series_test, prediction_test, metric="softdtw")

#plot centroids k=2 
import matplotlib.pyplot as plt
x= np.arange(0,24,0.5)
centroids=km_dba.cluster_centers_
centroids.shape
flow_1=centroids[0][:,0]
flow_1= flow_1.reshape((len(flow_1), 1))
flow_2=centroids[1][:,0]
flow_2= flow_2.reshape((len(flow_2), 1))
speed_1=centroids[0][:,1]
speed_1= speed_1.reshape((len(speed_1), 1))
speed_2=centroids[1][:,1]
speed_2= speed_2.reshape((len(speed_2), 1))
plt.figure(figsize=(15,15))
plt.title('centroids path')
plt.subplot(2,2,1)
plt.plot(x,scaler_flow_train.inverse_transform(flow_1),'r-', label = 'flow')
plt.xlabel(' hours of the day')
plt.ylabel('flow')
plt.title(' k=0')
plt.legend()
plt.subplot(2,2,2)
plt.plot(x,scaler_speed_train.inverse_transform(speed_1),'r-', label = 'speed')
plt.xlabel(' hours of the day')
plt.ylabel('speed')
plt.title('k=0')
plt.legend()
plt.subplot(2,2,3)
plt.plot(x,scaler_flow_train.inverse_transform(flow_2),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title(' k=1')
plt.legend()
plt.subplot(2,2,4)
plt.plot(x,scaler_speed_train.inverse_transform(speed_2),'r-', label = 'speed')
plt.xlabel('hours of the day')
plt.ylabel('speed')
plt.title('k=1')
plt.legend()
plt.show()

# plot the centroids k=3 
centroids=km_dba.cluster_centers_
centroids.shape
flow_1=centroids[0][:,0]
flow_1= flow_1.reshape((len(flow_1), 1))
flow_2=centroids[1][:,0]
flow_2= flow_2.reshape((len(flow_2), 1))
flow_3=centroids[2][:,0]
flow_3= flow_3.reshape((len(flow_3), 1))
speed_1=centroids[0][:,1]
speed_1= speed_1.reshape((len(speed_1), 1))
speed_2=centroids[1][:,1]
speed_2= speed_2.reshape((len(speed_2), 1))
speed_3=centroids[2][:,1]
speed_3= speed_3.reshape((len(speed_3), 1))
plt.figure(figsize=(15,15))
plt.subplot(3,2,1)
plt.plot(x,scaler_flow_train.inverse_transform(flow_1),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=0')
plt.legend()
plt.subplot(3,2,2)
plt.plot(x,scaler_speed_train.inverse_transform(speed_1),'r-', label = 'speed')
plt.xlabel('hours of the day')
plt.ylabel('speed')
plt.title(' k=0')
plt.legend()
plt.subplot(3,2,3)
plt.plot(x,scaler_flow_train.inverse_transform(flow_2),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=1')
plt.legend()
plt.subplot(3,2,4)
plt.plot(x,scaler_speed_train.inverse_transform(speed_2),'r-', label = 'speed')
plt.xlabel('hours of the day')
plt.ylabel('speed')
plt.title('k=1')
plt.legend()
plt.subplot(3,2,5)
plt.plot(x,scaler_flow_train.inverse_transform(flow_3),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=2')
plt.legend()
plt.subplot(3,2,6)
plt.plot(x,scaler_speed_train.inverse_transform(speed_3),'r-', label = 'speed')
plt.xlabel('hours of the day')
plt.ylabel('speed')
plt.title('k=2')
plt.legend()
plt.show()

centroids
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
speed_1=centroids[0][:,1]
speed_1= speed_1.reshape((len(speed_1), 1))
speed_2=centroids[1][:,1]
speed_2= speed_2.reshape((len(speed_2), 1))
speed_3=centroids[2][:,1]
speed_3= speed_3.reshape((len(speed_3), 1))
speed_4=centroids[3][:,1]
speed_4= speed_4.reshape((len(speed_4), 1))
plt.figure(figsize=(20,20))
plt.subplot(4,2,1)
plt.plot(x,scaler_flow_train.inverse_transform(flow_1),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=0')
plt.legend()
plt.subplot(4,2,2)
plt.plot(x,scaler_speed_train.inverse_transform(speed_1),'r-', label = 'speed')
plt.xlabel('hours of the day')
plt.ylabel('speed')
plt.title('k=0')
plt.legend()
plt.subplot(4,2,3)
plt.plot(x,scaler_flow_train.inverse_transform(flow_2),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=1')
plt.legend()
plt.subplot(4,2,4)
plt.plot(x,scaler_speed_train.inverse_transform(speed_2),'r-', label = 'speed')
plt.xlabel('hours of the day')
plt.ylabel('speed')
plt.title('k=1')
plt.legend()
plt.subplot(4,2,5)
plt.plot(x,scaler_flow_train.inverse_transform(flow_3),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=2')
plt.legend()
plt.subplot(4,2,6)
plt.plot(x,scaler_speed_train.inverse_transform(speed_3),'r-', label = 'speed')
plt.xlabel('hours of the day')
plt.ylabel('speed')
plt.title('k=2')
plt.legend()
plt.subplot(4,2,7)
plt.plot(x,scaler_flow_train.inverse_transform(flow_4),'r-', label = 'flow')
plt.xlabel('hours of the day')
plt.ylabel('flow')
plt.title('k=3')
plt.legend()
plt.subplot(4,2,8)
plt.plot(x,scaler_speed_train.inverse_transform(speed_4),'r-', label = 'speed')
plt.xlabel('hours of the day')
plt.ylabel('speed')
plt.title('k=3')
plt.legend()
plt.show()

prediction_test

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
plt.title('soft-DTW similarity measure')
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
calplot.calplot(events_train,yearlabel_kws={'color': 'black'}, cmap='cool', suptitle='Clustering of the days train set S1816', linewidth=2.3 )  

#test days 2013
events_test = pd.Series(prediction_test, index=all_days)
calplot.calplot(events_test,yearlabel_kws={'color': 'black'}, cmap='cool', suptitle='Clustering of the days test set S60', linewidth=2.3)  
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
