#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:26:59 2021

@author: nronzoni
"""

import pandas as pd 
import scipy 
import sklearn
import tslearn 
import numpy as np

#########TRAIN DATA##########
#import  train data
df= pd.read_csv("/Users/nronzoni/Downloads/Multivariate-Time-series-clustering-main/I35W_NB 30min 2013/S61.csv") 
df
#treatment of the first variable 
first_train=df.loc[:,'Flow']
first_train=np.array(first_train)
first_train= first_train.reshape((len(first_train), 1))
#from array to list 
first_train=first_train.tolist()
len(first_train)
from toolz.itertoolz import sliding_window, partition
#for every day of the train set store the flow observations 
days_first=list(partition(48,first_train))
days_first
len(days_first)
#from list to multidimensional array 
days_first=np.asarray(days_first)
days_first
from tslearn.utils import to_time_series, to_time_series_dataset
#create univariate series for normalized flow_observation 
first_time_series = to_time_series(days_first)
print(first_time_series.shape)
#normalize time series
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
first_time_series = TimeSeriesScalerMinMax(value_range=(0.0, 1.0)).fit_transform(first_time_series)
#first_time_series = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(first_time_series)
print(first_time_series.shape)


#treatment of the second variable 
second_train=df.loc[:,'Density']
second_train=np.array(second_train)
second_train= second_train.reshape((len(second_train), 1))
#from array to list 
second_train=second_train.tolist()
len(second_train)
#for every day of the train set store the flow observations 
days_second=list(partition(48,second_train))
days_second
len(days_second)
#from list to multidimensional array 
days_second=np.asarray(days_second)
days_second
#create univariate series for normalized flow_observation 
second_time_series = to_time_series(days_second)
print(second_time_series.shape)
#normalize time series
second_time_series = TimeSeriesScalerMinMax(value_range=(0.0, 1.0)).fit_transform(second_time_series)
#second_time_series = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(second_time_series)
print(second_time_series.shape)

#create the multivariate time series TRAIN  
multivariate=np.dstack((first_time_series,second_time_series))
multivariate_time_series_train = to_time_series(multivariate)
print(multivariate_time_series_train.shape)


#######TEST DATA ########
#import  test data
df_test= pd.read_csv("/Users/nronzoni/Downloads/Multivariate-Time-series-clustering-main/I35W_NB 30min 2013/S124.csv") 
df_test
#treatment of the first variable 
first_test=df_test.loc[:,'Flow']
first_test=np.array(first_test)
first_test= first_test.reshape((len(first_test), 1))
#from array to list 
first_test=first_test.tolist()
len(first_test)
#for every day of the train set store the flow observations 
days_first_test=list(partition(48,first_test))
days_first_test
len(days_first_test)
#from list to multidimensional array 
days_first_test=np.asarray(days_first_test)
days_first_test
#create univariate series for normalized flow_observation 
first_time_series_test = to_time_series(days_first_test)
print(first_time_series_test.shape)
#normalize time series
first_time_series_test = TimeSeriesScalerMinMax(value_range=(0.0, 1.0)).fit_transform(first_time_series_test)
#first_time_series_test = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(first_time_series_test)
print(first_time_series_test.shape)


#treatment of the second variable 
second_test=df_test.loc[:,'Density']
second_test=np.array(second_test)
second_test= second_test.reshape((len(second_test), 1))
#from array to list 
second_test=second_test.tolist()
len(second_test)
#for every day of the train set store the flow observations 
days_second_test=list(partition(48,second_test))
days_second_test
len(days_second_test)
#from list to multidimensional array 
days_second_test=np.asarray(days_second_test)
days_second_test
#create univariate series for normalized flow_observation 
second_time_series_test = to_time_series(days_second_test)
print(second_time_series_test.shape)
#normalize time series
second_time_series_test = TimeSeriesScalerMinMax(value_range=(0.0, 1.0)).fit_transform(second_time_series_test)
#second_time_series_test = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(second_time_series_test)
print(second_time_series_test.shape)

#create the multivariate time series TEST  
multivariate_test=np.dstack((first_time_series_test,second_time_series_test))
multivariate_time_series_test = to_time_series(multivariate_test)
print(multivariate_time_series_test.shape)


#clustering 
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, silhouette_score
#fit the algorithm on train data 
#tune the hyperparameters possible metrics: euclidean, dtw, softdtw
km_dba = TimeSeriesKMeans(n_clusters=3, metric="softdtw", max_iter=5,max_iter_barycenter=5, random_state=0).fit(multivariate_time_series_train)
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
plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
plt.plot(x,centroids[0][:,0],'r-', label = 'flow')
plt.xlabel('day')
plt.ylabel('flow')
plt.title('centroid flow k=1')
plt.legend()
plt.subplot(2,2,2)
plt.plot(x,centroids[0][:,1],'r-', label = 'density')
plt.xlabel('day')
plt.ylabel('density')
plt.title('centroid density k=1')
plt.legend()
plt.subplot(2,2,3)
plt.plot(x,centroids[1][:,0],'r-', label = 'flow')
plt.xlabel('day')
plt.ylabel('flow')
plt.title('centroid flow k=0')
plt.legend()
plt.subplot(2,2,4)
plt.plot(x,centroids[1][:,1],'r-', label = 'density')
plt.xlabel('day')
plt.ylabel('density')
plt.title('centroid density  k=0')
plt.legend()
plt.show()

# plot the centroids k=3 
centroids=km_dba.cluster_centers_
centroids.shape
plt.figure(figsize=(15,15))
plt.subplot(3,2,1)
plt.plot(x,centroids[0][:,0],'r-', label = 'flow')
plt.xlim(0, 24)
plt.xlabel('hour of the day')
plt.ylabel('flow')
plt.title('centroid flow k=2')
plt.legend()
plt.subplot(3,2,2)
plt.plot(x,centroids[0][:,1],'r-', label = 'density')
plt.xlabel('hour of the day')
plt.ylabel('density')
plt.title('centroid density k=2')
plt.legend()
plt.subplot(3,2,3)
plt.plot(x,centroids[1][:,0],'r-', label = 'flow')
plt.xlabel('hour of the day')
plt.ylabel('flow')
plt.title('centroid flow k=1')
plt.legend()
plt.subplot(3,2,4)
plt.plot(x,centroids[1][:,1],'r-', label = 'density')
plt.xlabel('hour of the day')
plt.ylabel('density')
plt.title('centroid density k=1')
plt.legend()
plt.subplot(3,2,5)
plt.plot(x,centroids[2][:,0],'r-', label = 'flow')
plt.xlabel('hour of the day')
plt.ylabel('flow')
plt.title('centroid flow k=0')
plt.legend()
plt.subplot(3,2,6)
plt.plot(x,centroids[2][:,1],'r-', label = 'density')
plt.xlabel('day')
plt.ylabel('density')
plt.title('centroid density k=0')
plt.legend()
plt.show()

centroids
# plot the centroids K=4 

centroids=km_dba.cluster_centers_
centroids.shape
plt.figure(figsize=(20,20))
plt.subplot(4,2,1)
plt.plot(x,centroids[0][:,0],'r-', label = 'flow')
plt.xlabel('day')
plt.ylabel('flow')
plt.title('centroid flow k=2')
plt.legend()
plt.subplot(4,2,2)
plt.plot(x,centroids[0][:,1],'r-', label = 'density')
plt.xlabel('day')
plt.ylabel('density')
plt.title('centroid speed k=2')
plt.legend()
plt.subplot(4,2,3)
plt.plot(x,centroids[1][:,0],'r-', label = 'flow')
plt.xlabel('day')
plt.ylabel('flow')
plt.title('centroid flow k=1')
plt.legend()
plt.subplot(4,2,4)
plt.plot(x,centroids[1][:,1],'r-', label = 'density')
plt.xlabel('day')
plt.ylabel('density')
plt.title('centroid density k=1')
plt.legend()
plt.subplot(4,2,5)
plt.plot(x,centroids[2][:,0],'r-', label = 'flow')
plt.xlabel('day')
plt.ylabel('flow')
plt.title('centroid flow k=0')
plt.legend()
plt.subplot(4,2,6)
plt.plot(x,centroids[2][:,1],'r-', label = 'density')
plt.xlabel('day')
plt.ylabel('density')
plt.title('centroid density k=0')
plt.legend()
plt.subplot(4,2,7)
plt.plot(x,centroids[3][:,0],'r-', label = 'flow')
plt.xlabel('day')
plt.ylabel('flow')
plt.title('centroid flow k=3')
plt.legend()
plt.subplot(4,2,8)
plt.plot(x,centroids[3][:,1],'r-', label = 'density')
plt.xlabel('day')
plt.ylabel('denisty')
plt.title('centroid density k=3')
plt.legend()
plt.show()



#visualization 
import calplot
#all days of 2013 
all_days = pd.date_range('1/1/2013', periods=365, freq='D')
#assign at every day the cluster 
events_train = pd.Series(prediction_train, index=all_days)
#plot the result 
calplot.calplot(events_train,yearlabel_kws={'color': 'black'}, cmap='cool', suptitle='Clustering of the days')  

#test days 2013
events_test = pd.Series(prediction_test, index=all_days)
calplot.calplot(events_test,yearlabel_kws={'color': 'black'}, cmap='cool', suptitle='Clustering of the days')  
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
