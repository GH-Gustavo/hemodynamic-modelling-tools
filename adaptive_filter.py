# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:47:35 2022

@author: Gustavo Henrique de Almeida
"""
## Filtro adaptativo para batimentos ectópicos

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import random as rd


def adap_filter(X):
    n = len(X)

    t = np.zeros(n)
    Ua = np.zeros(n)
    lampda = np.zeros(n)
    sigma = np.zeros(n)

    for N in range(1):
        t[0] = (X[0]+6*X[0]+15*X[0]+20*X[0]+15*X[0+1]+6*X[0+2]+X[0+3])/64
        t[1] = (X[1]+6*X[1]+15*X[1-1]+20*X[1]+15*X[1+1]+6*X[1+2]+X[1+3])/64
        t[2] = (X[2]+6*X[2-1]+15*X[2-2]+20*X[2]+15*X[2+1]+6*X[2+2]+X[2+3])/64
    
        for i in range (3,n-3):
            t[i] = (X[i-3]+6*X[i-2]+15*X[i-1]+20*X[i]+15*X[i+1]+6*X[i+2]+X[i+3])/64
        
        
        t[n-3] = (X[n-3-3]+6*X[n-3-2]+15*X[n-3-1]+20*X[n-3]+15*X[n-3+1]+6*X[n-3+2]+X[n-3])/64
        t[n-2] = (X[n-2-3]+6*X[n-2-2]+15*X[n-2-1]+20*X[n-2]+15*X[n-2+1]+6*X[n-2]+X[n-2])/64  # n seria n-1+2
        t[n-1] = (X[n-3-1] + 6*X[n-1-2]+15*X[n-1-1]+20*X[n-1]+15*X[n-1-1]+6*X[n-1-1]+X[n-1-1])/64
    
    # Calculate < a(n), in this code, use Ua(n) to replace < a(n), lampda for ;, sigma for C, cause matlab can't recognize them. 

        Ua[0] =np.mean(X)
        lampda[0] = Ua[0]**2
        c=0.05   # controlling coefficient
   
        for i in range(1,n):
            Ua[i] = Ua[i-1]-c*(Ua[i-1]-t[i-1])
            lampda[i] = lampda[i-1] -c*(lampda[i-1]-t[i-1]**2)
        for i in range(n):
            sigma[i] = np.sqrt(abs(Ua[i]**2-lampda[i]))
                
    # Classify normal and abnormal RR-intervals 

        Cf=3
        p=10
        sigma_mean= np.mean(sigma);
        last = Ua[0]
   
        for i in range(1,n):
            if(abs(X[i]-X[i-1])> ((p*X[i-1]/100)+Cf*sigma_mean) and (abs(X[i]-last)>((p*last/100)+Cf*sigma_mean))):
                X[i] = rd.random()*sigma[i]+(Ua[i]-0.5*sigma[i])  #produce a random number between Ua(i)-0.5*sigma(i) and Ua(i)+0.5*sigma(i)
            else:
                last = X[i]
          
#After the above loop, the X(n) is changed into X(n)%, but in the program X(n)% is still denoted by X(n)
# and use X(n)% to recalculate t(n), Ua, lampda and sigma, so I use 2 times loop. 'for N=1:2'

 
# The adaptive controlling procedure %

    Cf1=3
    sigmaB=0.02  # sigmaB is 200ms 
    
    f_hrv=X

    for i in range(n):
        if(abs(X[i]-Ua[i])>(Cf1*sigma[i]+sigmaB)):
            f_hrv[i] = t[i]

    return(f_hrv)

