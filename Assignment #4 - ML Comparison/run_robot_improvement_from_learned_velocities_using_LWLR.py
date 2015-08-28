import numpy as np
import math
import matplotlib.pyplot as plt
import random
import scipy.optimize as opt

from math import pi,sqrt,fabs,sin,cos,atan2,exp
from numpy import transpose
from numpy.linalg import inv


# Reading data from .dat files into separate arrays
odometry = np.loadtxt('ds0_Odometry.dat') # Time, v, w
groundtruth = np.loadtxt('ds0_Groundtruth.dat') # Time, x, y, theta



#Intepolation between groundtruth file to find robot position at time stamps on odometry files
groundtruth_closest = np.loadtxt('groundtruth_closest.txt') # Time, x, y, theta






#Step #1 - Build a Suitable Training Dataset

#Learning Aim - Minimizing Dead Reackoning Error
#
#Training Dataset - Apply learning algorithm to ds1
#		  - Since robot subject to noise, the actual velocities differ from commanded/measured ones
#		  - Model this difference by applying LWLR to query point velocites contained in odometry file
#		  - First determine the extracted velocities from groundtruth files -> target function
#		  - Will then apply learning algorithm to training dataset to learn improved velocities -> target function approximation
#		  - Last, will send improved velocities to motion control to determine improved robot position
#
#	   	  - Task: mobile robot navigation
#		  - Inputs: (x,y,theta)_t-1, (x,y,theta)_t
#		  - Outputs: (v,w) improved
#	   	  - Performance Measure: error from groundtruth position
#	   	  - Training Experience: ds1
#		  - Target Function Representation:
#		  	- x'= x - (v/w)*sin(theta) + (v/w)*sin(theta')
#		  	- y' = y + (v/w)*cos(theta) - (v/w)*cos(theta')
#		  	   --> Solve non-linear system of equations for v,w









#Positional Error from Motion Model/Simulated Controller on Entire Robot Dataset before applying Machine Learning
robot_position = np.zeros((len(odometry),3))

i = 0
distance_odo=0
total_distance_odo=0
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


	distance_odo = sqrt((groundtruth_closest[i,1]-robot_position[i,0])**2 + (groundtruth_closest[i,2]-robot_position[i,1])**2)
	total_distance_odo = total_distance_odo + distance_odo

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












#3 - Apply your learning algorithm to your dataset developed in Step #1. Evaluate its performance using at least two measures (your choice).
#Learning Algorithm - Locally Weighted Linear Regression


#Y array containing extracted velocities from groundtruth file
Y = np.loadtxt('Y_array.txt') # time, w

X = np.zeros((len(odometry),1)) # time, w
q = np.zeros((1,1)) #query point
Y_array = np.zeros((len(odometry),1)) #output values corresponding to input vectors, Y_i = ith row of y_i
W = np.zeros((len(odometry),1)) #weight matrix with W_ii = w_i


#Defining the Unweighted Euclidean Distance function 
def Distance_Eucl_Unweighted(Xi,q):
	xi = transpose(Xi)

	return sqrt(np.dot(transpose(xi-q),(xi-q)))


#Defining the Quadratic Kernel function
def Kernel_Quadratic(Xi,q):
	distance = Distance_Eucl_Unweighted(Xi,q)
	#print distance
	percent = 0.05
	if fabs(distance)<percent:
		kernel = (percent**2-distance**2)/(percent**2)
	else:
		kernel = 1e-10
	#print kernel
	return kernel


#Defining the Gaussian Kernel function
def Kernel_Gaussian(Xi,q):
	return exp(-Distance_Eucl_Unweighted(Xi,q))


#Defining the Predicted Target Function Output
def Predicted_Target_Function_Output(q,W,X,Y):
	WX = np.zeros((len(odometry),1),dtype=np.float64)
	WY = np.zeros((len(odometry),1),dtype=np.float64)

	i=0
	while i<len(W):
		WX[i] = W[i]*X[i]
		WY[i] = W[i]*Y[i]

		i = i+1

	q_transpose = transpose(q)
	#WX = np.dot(W,X)
	WXtranWX = np.dot(transpose(WX),WX)
	Inv = inv(WXtranWX)
	#WY = np.dot(W,Y)
	WXtranWY = np.dot(transpose(WX),WY)


	return np.dot(np.dot(q_transpose,Inv),WXtranWY) #y(q) = (q^T)*((WX)^T*(WX))^-1*(WX)^T*(WY)






#Determining all possible combinations of (v,w) in odometry file, to be stored as query points
query = np.zeros((1,2))

i = 0
while i<len(X):
	#Handling first time step (t=0)
	if i==0:
		query[0,0] = i #iteration in odometry
		query[0,1] = odometry[i,2] #w


	#Handling every other time step
	elif odometry[i,2] not in query[:,1]:
		q1 = np.zeros((1,2))

		q1[0,0] = i #iteration in odometry
		q1[0,1] = odometry[i,2] #w

		query = np.vstack((query,q1))

	i = i+1


np.savetxt('query_velocities.txt', query, delimiter='\t')
print "Saved query velocties from odometry file. Contains",len(query),"entries."




#While loop calculating the predicted target function output
print "Different sets of predicted target function velocities (v,w).",len(query),"query points found from odometry file."
vels_pred = np.zeros((len(query),2))


counter = 0
while counter<len(query):

	#Query point corresponding to each velocity (Xi) -- any value would work though
	q[0,0] = query[counter,1] #w


	#Stacked while loop to calculate X,Y once and W matrix each iteration of query points
	counter2 = 0
	while counter2<len(odometry)-1:

		#Calculates X and Y matrices fully once
		if counter==0:

			#Calculates X matrix once - w
			X[counter2] = odometry[counter2,2]


			#Updates previously calculated Y matrix in order to account for inaccuracies in fsolve()
			w = Y[counter2] #rotational speed

			if counter2==len(odometry)-1:
				delta_t = odometry[counter2,0]-odometry[counter2-1,0]
			else:
				delta_t = odometry[counter2+1,0]-odometry[counter2,0] #time step


			if fabs(w)>1:
				if groundtruth_closest[counter2+1,3] == groundtruth_closest[counter2,3]:
					entry1 = groundtruth_closest[counter2,0]
					entry2 = groundtruth_closest[counter2+1,0]

					new_counter2 = counter2+2
					while entry1 == entry2:
						entry2 = groundtruth_closest[new_counter2,0]
						new_counter2 = new_counter2+1
				else:
					new_counter2=counter2+1

				w = (groundtruth_closest[new_counter2,3]-groundtruth_closest[counter2,3]) / delta_t

			if fabs(w)>5:
				w = 0


			Y_array[counter2] = w




		Xrow = np.array([X[counter2]]) #creates Xrow in the correct format

		W[counter2] = Kernel_Quadratic(Xrow,q) #W matrix gets updated each time there is a new query point


		counter2 = counter2+1



	#Stores X and Y arrays to .txt file
	#if counter==1:
	#	np.savetxt('Y_array_11.txt', Y, delimiter='\t')
	#	print "Done saving file."


	#Predicted target function output calculating velocities for each query point
	returned_v = Predicted_Target_Function_Output(q,W,X,Y_array)

	vels_pred[counter,0] = q[0,0] #odometry w
	vels_pred[counter,1] = returned_v[0,0] #predicted target function w


	print counter,"\t","W_odo:",round(q[0,0],5),"\t","W_pred","\t",round(returned_v[0,0],5),"\t","W_grnd","\t",Y[query[counter,0]],query[counter,0]

	counter = counter+1






#Saving Predicted Velocities to .txt files
np.savetxt('Predicted_Velocities_0.txt', vels_pred, delimiter='\t', fmt='%1.4f')




#Figure 2 - Relationship Between Extracted Groundtruth Velocities and Odometry Velocities
w_odo = X[:,0]

w_grnd = Y_array[:,0]


plt.figure(2)
plt.scatter(w_odo, w_grnd, label='Angular Velocity')
plt.title('Relationship Between Extracted Groundtruth Angular Velocities and Odometry')
plt.xlabel('Odometry Angular Velocities [rad/s]')
plt.ylabel('Groundtruth Angular Velocities [rad/s]')
plt.legend(loc=0)
plt.grid()
#plt.show()



#Figure 3 - Relationship Between Extracted Groundtruth Velocities and Odometry Velocities
w_odo1 = vels_pred[:,0]

w_grnd1 = vels_pred[:,1]


plt.figure(3)
plt.scatter(w_odo1, w_grnd1, label='Angular Velocity')
plt.title('Relationship Between Extracted Groundtruth Angular Velocities and Odometry')
plt.xlabel('Odometry Angular Velocities [rad/s]')
plt.ylabel('Groundtruth Angular Velocities [rad/s]')
plt.legend(loc=0)
plt.grid()
#plt.show()





#While loop to calculate (x,y,theta) for each iteration based on predicted velocities (vels_pred)

robot_position_pred = np.zeros((odometry.size,3))


i = 0
while i<len(odometry):
	#Handling first time step (t=0)
	if i==0:
		robot_position_pred[i,0] = groundtruth[0,1] #x-position
		robot_position_pred[i,1] = groundtruth[0,2] #y-position
		robot_position_pred[i,2] = groundtruth[0,3] #heading

		v=0
		w=0

	#Handling last time step (different delta_t only)
	elif i==len(odometry)-1:
		heading = robot_position_pred[i-1,2] #previous time step robot heading
		delta_t = odometry[i,0]-odometry[i-1,0] #time step
		v_odo = odometry[i,1] #translation speed
		w_odo = odometry[i,2] #rotational speed


		j=0
		while j<len(vels_pred):

			if fabs(w_odo - vels_pred[j,0])<0.001:
				v = v_odo
				w = vels_pred[j,1]

			j = j+1


		#No rotational velocity, forward movement only
		if w==0:
			robot_position_pred[i,0] = robot_position_pred[i-1,0] + (v*delta_t)*cos(heading) #x-position
			robot_position_pred[i,1] = robot_position_pred[i-1,1] + (v*delta_t)*sin(heading) #y-position
			robot_position_pred[i,2] = robot_position_pred[i-1,2] #heading

		#Circular trajectory with radius r=v/w
		else:
			robot_position_pred[i,0] = robot_position_pred[i-1,0] - (v/w)*sin(heading) + (v/w)*sin(heading+w*delta_t) #x-position
			robot_position_pred[i,1] = robot_position_pred[i-1,1] + (v/w)*cos(heading) - (v/w)*cos(heading+w*delta_t) #y-position
			robot_position_pred[i,2] = robot_position_pred[i-1,2] + (w*delta_t) #headingg


	#Handling every other time step
	else:
		heading = robot_position_pred[i-1,2] #previous time step robot heading
		delta_t = odometry[i+1,0]-odometry[i,0] #time step
		v_odo = odometry[i,1] #translation speed
		w_odo = odometry[i,2] #rotational speed


		j=0
		while j<len(vels_pred):

			if fabs(w_odo - vels_pred[j,0])<0.001:
				v = v_odo
				w = vels_pred[j,1]

			j = j+1


		#No rotational velocity, forward movement only
		if w==0:
			robot_position_pred[i,0] = robot_position_pred[i-1,0] + (v*delta_t)*cos(heading) #x-position
			robot_position_pred[i,1] = robot_position_pred[i-1,1] + (v*delta_t)*sin(heading) #y-position
			robot_position_pred[i,2] = robot_position_pred[i-1,2] #heading

		#Circular trajectory with radius r=v/w
		else:
			robot_position_pred[i,0] = robot_position_pred[i-1,0] - (v/w)*sin(heading) + (v/w)*sin(heading+w*delta_t) #x-position
			robot_position_pred[i,1] = robot_position_pred[i-1,1] + (v/w)*cos(heading) - (v/w)*cos(heading+w*delta_t) #y-position
			robot_position_pred[i,2] = robot_position_pred[i-1,2] + (w*delta_t) #heading


	i = i+1



#Saving Predicted Robot Positions to .txt files
np.savetxt('Robot_Position_Predicted_0.txt', robot_position_pred, delimiter='\t', fmt='%1.4f')



#Predicted robot position for each query point, based on LWLR on velocities
x_obs = robot_position_pred[:,0]
y_obs = robot_position_pred[:,1]


#Figure 3 - Performance of LWLR on Robot Dataset
plt.figure(3)
plt.plot(x1, y1, label='Robot Groundtruth')
plt.plot(x, y, label='Robot Position')
plt.plot(x_obs, y_obs, label='Improved Robot Position')
plt.title('Robot Position Improvement from Learned Velocities using LWLR')
plt.xlabel('X-position [m]')
plt.ylabel('Y-position [m]')
plt.legend(loc=0)
plt.grid()
plt.show()
