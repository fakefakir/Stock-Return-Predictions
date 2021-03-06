# Recurrent Neural Network and a Hybrid Model for Prediction of Stock Returns
This piece of work is based on Dr.Rather's paper of **_Recurrent Neural Network and a Hybrid Model for Prediction of Stock Returns (2015)_**.  [Availabe Here](https://www.sciencedirect.com/science/article/pii/S0957417414007684)

## Overview

This project consists of three models. A linear Autoregression (AR) model, a Autoregression Moving Referece Recurrent Neural Network (AR-MRNN) model, and a Hybird Prediction Model (HPM) as in Rather's paper.

Source codes for above models in three separate files: **AR.py**, **RNN.py** and **HPM.py**.

Dataset is in the folder of **Data**, which include of daily historical data of six stocks from **02/01/2007** to **26/03/2010**. 

Data are downloaded from National Stock Exchange of India (https://www.nseindia.com)

## Autoregressive Model (AR)
This is implemented using the standard statsmodels package.  
Statsmodels Documentation: https://www.statsmodels.org/stable/index.html

## Recurrent Neural Network Model (AR-MRNN)
This modle has two parameters:  
p: The autoregressive order  
k: The refrence point  

## Hybird Prediction Model (HPM)
Implemented the _Algorithm 1_ in Rather's paper

## My Utilities Toolkit (myUtilities)
One function and one class included  
  
  
 _get_return_ function, calculate the daily (fre=1) or weekly returns (fre=7) from daily prices data  
 _EarlyStoppingByMSE_ calss, set the threshold of training losses for the neural network model to stop training
