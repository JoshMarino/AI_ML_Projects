import numpy as np
import math
import matplotlib.pyplot as plt
import random

from math import pi,sqrt,fabs,sin,cos,atan2,exp
from numpy import transpose
from numpy.linalg import inv


# Reading data from .dat files into separate arrays
barcodes = np.loadtxt('ds1_Barcodes.dat', dtype='int') # Subject#, Barcode#
odometry = np.loadtxt('ds1_Odometry.dat') # Time, v, w
groundtruth = np.loadtxt('ds1_Groundtruth.dat') # Time, x, y, theta
measurement = np.loadtxt('ds1_Measurement.dat') # Time, Subject#, range, bearing
landmark_groundtruth = np.loadtxt('ds1_Landmark_Groundtruth.dat') # Time, x, y, x-std-dev, y-std-dev







#Step #1 - Build a Suitable Training Dataset

#Learning Aim - Minimizing Dead Reackoning Error
#
#Training Dataset - Apply learning algorithm to n-sections of dataset, randomly
#		  - Since robot subject to noise, the actual velocities differ from commanded/measured ones
#		  - Model this difference by a zero-centered random variable with finite variance
#		  - These variables are robot-specific motion parameters, alpha1 -> alpha6
#		  - Replace velocities by adding to velocities by sampling from normal distribution with zero mean and variance b
#		  - Will then apply learning algorithm to each training dataset to learn alpha1 -> alpha6 for each dataset
#		  - Last, will average each robot-specific motion parameter over all training datasets
#
#	   	  - Task: mobile robot navigation
#		  - Inputs: (x,y,theta)_t-1, (v,w)_t
#		  - Outputs: (x,y,theta)_t
#	   	  - Performance Measure: error from groundtruth position
#	   	  - Training Experience: n-sections of ds1
#		  - Target Function Representation:
#		  	- v_error = v + sample[alpha1*abs(v) + alpha2*abs(w)]
#		  	- w_error = w + sample[alpha3*abs(v) + alpha4*abs(w)]
#		  	- alpha_error = sample[alpha5*abs(v) + alpha6*abs(w)]




#Positional Error from Motion Model/Simulated Controller on Entire Robot Dataset before applying Machine Learning
robot_position = np.zeros((odometry.size,3))

i = 0
while i<len(odometry):
	#Handling first time step (t=0)
	if i==0:
		robot_position[i,0] = groundtruth[0,1] #x-position
		robot_position[i,1] = groundtruth[0,2] #y-position
		robot_position[i,2] = groundtruth[0,3] #Heading

	#Handling last time step (different delta_t only)
	elif i==len(odometry)-1:
		heading = robot_position[i-1,2]
		v = odometry[i,1] #translation speed
		w = odometry[i,2] #rotational speed
		delta_t = odometry[i,0]-odometry[i-1,0] #time step

		#No rotational velocity, forward movement only
		if w==0:
			robot_position[i,0] = robot_position[i-1,0] + (v*delta_t)*cos(heading) #x-position
			robot_position[i,1] = robot_position[i-1,1] + (v*delta_t)*sin(heading) #y-position
			robot_position[i,2] = robot_position[i-1,2] #heading

		#Circular trajectory with radius r=v/w
		else:
			robot_position[i,0] = robot_position[i-1,0] - (v/w)*sin(heading) + (v/w)*sin(heading+w*delta_t) #x-position
			robot_position[i,1] = robot_position[i-1,1] + (v/w)*cos(heading) - (v/w)*cos(heading+w*delta_t) #y-position
			robot_position[i,2] = robot_position[i-1,2] + (w*delta_t) #heading

	#Handling every other time step
	else:
		heading = robot_position[i-1,2] #previous time step robot heading
		v = odometry[i,1] #translation speed
		w = odometry[i,2] #rotational speed
		delta_t = odometry[i+1,0]-odometry[i,0] #time step

		#No rotational velocity, forward movement only
		if w==0:
			robot_position[i,0] = robot_position[i-1,0] + (v*delta_t)*cos(heading) #x-position
			robot_position[i,1] = robot_position[i-1,1] + (v*delta_t)*sin(heading) #y-position
			robot_position[i,2] = robot_position[i-1,2] #heading

		#Circular trajectory with radius r=v/w
		else:
			robot_position[i,0] = robot_position[i-1,0] - (v/w)*sin(heading) + (v/w)*sin(heading+w*delta_t) #x-position
			robot_position[i,1] = robot_position[i-1,1] + (v/w)*cos(heading) - (v/w)*cos(heading+w*delta_t) #y-position
			robot_position[i,2] = robot_position[i-1,2] + (w*delta_t) #heading

	i = i+1



x = robot_position[:,0]
y = robot_position[:,1]
x1 = groundtruth[:,1]
y1 = groundtruth[:,2]

plt.figure(1)
plt.plot(x, y, label='Robot Position')
plt.plot(x1, y1, label='Robot Groundtruth')
plt.title('Positional Error from Motion Controller')
plt.xlabel('X-position [m]')
plt.ylabel('Y-position [m]')
plt.legend(loc=0)
plt.grid()
#plt.show()





#2 - Code a learning algorithm. Demonstrate th algorithm's functioning on a simple dataset (e.g. noisy data generated by a sine function).
#Learning Algorithm - Locally Weighted Linear Regression
q = np.zeros((1,1))
#W = np.zeros((100,100))
#X = np.zeros((100,1))
#Y = np.zeros((100,1))
sine_wave = np.zeros((100,2))
sine_obs = np.zeros((100,2))


def Distance_Eucl_Unweighted(xi,q):	
	return sqrt(transpose(xi-q)*(xi-q))


def Kernel_Gaussian(xi,q):
	return exp(-Distance_Eucl_Unweighted(xi,q))


def Predicted_Target_Function_Output(q,W,X,Y):
	q_transpose = transpose(q)
	WX = np.dot(W,X)
	WXtranWX = np.dot(transpose(WX),WX)
	Inv = (WXtranWX)
	WY = np.dot(W,Y)
	WXtranWY = np.dot(transpose(WX),WY)
	
	return np.dot(np.dot(q_transpose,Inv),WXtranWY)








j = 0
while j<len(sine_wave):
	sine_wave[j,0] = j*2*pi/len(sine_wave)
	sine_wave[j,1] = sin(sine_wave[j,0]) + random.uniform(-0.15,0.15)

	j = j+1


k = 0
while k<len(sine_wave):	
	sine_obs[k,0] = sine_wave[k,0]

	Xk = sine_obs[k,0]
	Yk = sine_obs[k,1]
	q[0,0] = sine_obs[k,0]
	Wkk = Kernel_Gaussian(Xk,q[0,0])

	if k == 0:
		X = [[Xk]]
		Y = [[Yk]]
		W = [[Wkk]]
	else:
		X = np.vstack((X,Xk))
		Y = np.vstack((Y,Yk))
		Wrowtemp = np.zeros((1,k))
		W = np.vstack((W,Wrowtemp))
		Wcolumntemp = np.zeros((k+1,1))
		Wcolumntemp[k,0] = Wkk
		W = np.hstack((W,Wcolumntemp))

	sine_obs[k,1] = Predicted_Target_Function_Output(q,W,X,Y)

	k = k+1

print W
#print Predicted_Target_Function_Output(q,W,X,Y)



time = sine_wave[:,0]
sine = sine_wave[:,1]

timeobs = sine_obs[:,0]
sineobs = sine_obs[:,1]

plt.figure(2)
plt.plot(time, sine, label='Noisy Sine Wave')
plt.plot(timeobs, sineobs, label='LWLR Sine Wave')
plt.title('Performance of LWLR on Noisy Data Generated by Sine Function')
plt.xlabel('X [rad]')
plt.ylabel('Sin(X) [m]')
plt.legend(loc=0)
plt.xlim([-0,2*pi])
plt.ylim([-1.2,1.2])
plt.grid()
plt.show()
