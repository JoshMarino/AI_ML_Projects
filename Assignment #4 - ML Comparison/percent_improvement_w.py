import numpy as np
import math
import matplotlib.pyplot as plt
import random
import scipy.optimize as opt

from math import pi,sqrt,fabs,sin,cos,atan2,exp
from numpy import transpose
from numpy.linalg import inv



#1.00*w --> -46.91%   6.281m
#1.01*w --> -40.46%   6.006m
#1.025*w --> -30.98%   5.600m
#1.05*w --> -16.28%   4.972m
#1.075*w --> -3.8%   4.438m
#1.10*w --> % +5.09%   4.058m
#1.125*w --> % +10.5%   3.827m
#1.15*w --> % +13.28%   3.708m
#1.175*w --> % +14.03%   3.676m
#1.20*w --> % +13.04%   3.718m
#1.225*w --> % +10.69%   3.819m
#1.275*w --> % +4.17%   4.098m
#1.40*w --> % -13.36%   4.847m

#1.15*w --> % +13.28%   3.708m
#1.16*w --> % +13.8%   3.686m
#1.17*w --> % +14.02%   3.676m
#1.174*w --> % +14.03%   3.676m
#1.175*w --> % +14.03%   3.676m
#1.176*w --> % +14.02%   3.676m
#1.18*w --> % +13.96%   3.679m


percent_w=np.zeros((19,1))
percent_improvement=np.zeros((19,1))


percent_w[0]=1
percent_improvement[0]=-46.91

percent_w[1]=1.01
percent_improvement[1]=-40.46

percent_w[2]=1.025
percent_improvement[2]=-30.98

percent_w[3]=1.05
percent_improvement[3]=-16.28

percent_w[4]=1.075
percent_improvement[4]=-3.80

percent_w[5]=1.10
percent_improvement[5]=+5.09

percent_w[6]=1.125
percent_improvement[6]=+10.50

percent_w[7]=1.15
percent_improvement[7]=+13.28

percent_w[8]=1.175
percent_improvement[8]=+14.03

percent_w[9]=1.20
percent_improvement[9]=+13.04

percent_w[10]=1.225
percent_improvement[10]=+10.69

percent_w[11]=1.275
percent_improvement[11]=+4.17

percent_w[12]=1.40
percent_improvement[12]=-13.36

percent_w[13]=1.15
percent_improvement[13]=+13.28

percent_w[14]=1.16
percent_improvement[14]=+13.8

percent_w[15]=1.17
percent_improvement[15]=+14.02

percent_w[16]=1.174
percent_improvement[16]=+14.03

percent_w[17]=1.176
percent_improvement[17]=+14.02

percent_w[18]=1.18
percent_improvement[18]=+13.96



plt.scatter(percent_w,percent_improvement)
plt.scatter(percent_w[8],percent_improvement[8],color='red',s=75)
plt.title('Analysis of Adjusting Angular Velocity on Percent Improvement')
plt.xlabel('Percent Angular Velocity [%]')
plt.ylabel('Improvement from LWLR Angular Velocity [%]')
plt.grid()
plt.show()
