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
Y = np.loadtxt('Y_array_0.txt') # v, w

q = np.zeros((2,1)) #query point
X = np.zeros((len(odometry),2)) #input vectors with X_i = ith row of x_i
Y_array = np.zeros((len(odometry),2)) #output values corresponding to input vectors, Y_i = ith row of y_i
W = np.zeros((len(odometry),len(odometry))) #weight matrix with W_ii = w_i


#Defining the Unweighted Euclidean Distance function 
def Distance_Eucl_Unweighted(Xi,q):
	xi = transpose(Xi)

	return sqrt(np.dot(transpose(xi-q),(xi-q)))


#Defining the Gaussian Kernel function
def Kernel_Gaussian(Xi,q):
	return exp(-Distance_Eucl_Unweighted(Xi,q))


#Defining the Predicted Target Function Output
def Predicted_Target_Function_Output(q,W,X,Y):
	q_transpose = transpose(q)
	WX = np.dot(W,X)
	WXtranWX = np.dot(transpose(WX),WX)
	Inv = inv(WXtranWX)
	WY = np.dot(W,Y)
	WXtranWY = np.dot(transpose(WX),WY)
	
	return np.dot(np.dot(q_transpose,Inv),WXtranWY) #y(q) = (q^T)*((WX)^T*(WX))^-1*(WX)^T*(WY)


#Defining Kinematic Model Function to Solve for Actual Velocities from Groundtruth File - no w
def kinematic_model_no_w(variables,counter2):

	(v,w) = variables

	if (odometry[counter2,0]-groundtruth_closest[counter2,0]) < 0: #round down
		percent = (odometry[counter2,0]-groundtruth_closest[counter2-1,0]) / (groundtruth_closest[counter2,0]-groundtruth_closest[counter2-1,0])
		percent1 = (odometry[counter2+1,0]-groundtruth_closest[counter2,0]) / (groundtruth_closest[counter2+1,0]-groundtruth_closest[counter2,0])

		x = groundtruth_closest[counter2-1,1] + percent*(groundtruth_closest[counter2,1]-groundtruth_closest[counter2-1,1])
		y = groundtruth_closest[counter2-1,2] + percent*(groundtruth_closest[counter2,2]-groundtruth_closest[counter2-1,2])
		theta = groundtruth_closest[counter2-1,3] + percent*(groundtruth_closest[counter2,3]-groundtruth_closest[counter2-1,3])

		x_prime = groundtruth_closest[counter2,1] + percent1*(groundtruth_closest[counter2+1,1]-groundtruth_closest[counter2,1])
		y_prime = groundtruth_closest[counter2,2] + percent1*(groundtruth_closest[counter2+1,2]-groundtruth_closest[counter2,2])
		theta_prime = groundtruth_closest[counter2,3] + percent1*(groundtruth_closest[counter2+1,3]-groundtruth_closest[counter2,3])

	else: #round up
		percent = (odometry[counter2,0]-groundtruth_closest[counter2,0]) / (groundtruth_closest[counter2+1,0]-groundtruth_closest[counter2,0])
		percent1 = (odometry[counter2+1,0]-groundtruth_closest[counter2+1,0]) / (groundtruth_closest[counter2+2,0]-groundtruth_closest[counter2+1,0])

		x = groundtruth_closest[counter2,1] + percent*(groundtruth_closest[counter2+1,1]-groundtruth_closest[counter2,1])
		y = groundtruth_closest[counter2,2] + percent*(groundtruth_closest[counter2+1,2]-groundtruth_closest[counter2,2])
		theta = groundtruth_closest[counter2,3] + percent*(groundtruth_closest[counter2+1,3]-groundtruth_closest[counter2,3])

		x_prime = groundtruth_closest[counter2+1,1] + percent1*(groundtruth_closest[counter2+2,1]-groundtruth_closest[counter2+1,1])
		y_prime = groundtruth_closest[counter2+1,2] + percent1*(groundtruth_closest[counter2+2,2]-groundtruth_closest[counter2+1,2])
		theta_prime = groundtruth_closest[counter2+1,3] + percent1*(groundtruth_closest[counter2+2,3]-groundtruth_closest[counter2+1,3])


	delta_t = odometry[counter2+1,0]-odometry[counter2,0]

	#Groundtruth determines no rotation
	first_eq = x + (v*delta_t)*cos(theta) - x_prime
	second_eq = w


	return [first_eq,second_eq]


#Defining Kinematic Model Function to Solve for Actual Velocities from Groundtruth File - v and w
def kinematic_model(variables,counter2):

	(v,w) = variables

	if (odometry[counter2,0]-groundtruth_closest[counter2,0]) < 0: #round down
		percent = (odometry[counter2,0]-groundtruth_closest[counter2-1,0]) / (groundtruth_closest[counter2,0]-groundtruth_closest[counter2-1,0])
		percent1 = (odometry[counter2+1,0]-groundtruth_closest[counter2,0]) / (groundtruth_closest[counter2+1,0]-groundtruth_closest[counter2,0])

		x = groundtruth_closest[counter2-1,1] + percent*(groundtruth_closest[counter2,1]-groundtruth_closest[counter2-1,1])
		y = groundtruth_closest[counter2-1,2] + percent*(groundtruth_closest[counter2,2]-groundtruth_closest[counter2-1,2])
		theta = groundtruth_closest[counter2-1,3] + percent*(groundtruth_closest[counter2,3]-groundtruth_closest[counter2-1,3])

		x_prime = groundtruth_closest[counter2,1] + percent1*(groundtruth_closest[counter2+1,1]-groundtruth_closest[counter2,1])
		y_prime = groundtruth_closest[counter2,2] + percent1*(groundtruth_closest[counter2+1,2]-groundtruth_closest[counter2,2])
		theta_prime = groundtruth_closest[counter2,3] + percent1*(groundtruth_closest[counter2+1,3]-groundtruth_closest[counter2,3])

	else: #round up
		percent = (odometry[counter2,0]-groundtruth_closest[counter2,0]) / (groundtruth_closest[counter2+1,0]-groundtruth_closest[counter2,0])
		percent1 = (odometry[counter2+1,0]-groundtruth_closest[counter2+1,0]) / (groundtruth_closest[counter2+2,0]-groundtruth_closest[counter2+1,0])

		x = groundtruth_closest[counter2,1] + percent*(groundtruth_closest[counter2+1,1]-groundtruth_closest[counter2,1])
		y = groundtruth_closest[counter2,2] + percent*(groundtruth_closest[counter2+1,2]-groundtruth_closest[counter2,2])
		theta = groundtruth_closest[counter2,3] + percent*(groundtruth_closest[counter2+1,3]-groundtruth_closest[counter2,3])

		x_prime = groundtruth_closest[counter2+1,1] + percent1*(groundtruth_closest[counter2+2,1]-groundtruth_closest[counter2+1,1])
		y_prime = groundtruth_closest[counter2+1,2] + percent1*(groundtruth_closest[counter2+2,2]-groundtruth_closest[counter2+1,2])
		theta_prime = groundtruth_closest[counter2+1,3] + percent1*(groundtruth_closest[counter2+2,3]-groundtruth_closest[counter2+1,3])


	delta_t = odometry[counter2+1,0]-odometry[counter2,0]


	#Solve for v and w
	first_eq = x - (v/w)*sin(theta) + (v/w)*sin(theta_prime) - x_prime
	second_eq = y + (v/w)*cos(theta) - (v/w)*cos(theta_prime) - y_prime

	return [first_eq,second_eq]





#Determining all possible combinations of (v,w) in odometry file, to be stored as query points
query = np.zeros((1,3))

i = 0
while i<len(odometry):
	#Handling first time step (t=0)
	if i==0:
		query[0,0] = i #iteration in odometry
		query[0,1] = odometry[i,1] #v
		query[0,2] = odometry[i,2] #w


	#Handling every other time step
	elif odometry[i,1] not in query[:, 1] or odometry[i,2] not in query[:, 2]:
		q1 = np.zeros((1,3))
		q1[0,0] = i #iteration in odometry
		q1[0,1] = odometry[i,1] #v
		q1[0,2] = odometry[i,2] #w

		query = np.vstack((query,q1))


	i = i+1



#While loop calculating the predicted target function output
print "Different sets of velocities (v,w) found in odometry file"
vels_pred = np.zeros((len(query),4))


counter = 0
while counter<len(query):

	#Query point corresponding to each velocity (Xi) -- any value would work though
	q[0,0] = query[counter,1] #v
	q[1,0] = query[counter,2] #w

	#Stacked while loop to calculate X,Y once and W matrix each iteration of query points
	counter2 = 0
	while counter2<len(odometry)-1:

		#Calculates X and Y matrix fully one time
		if counter==0:

			#Calculates X matrix once - v,w
			X[counter2,0] = odometry[counter2,1]
			X[counter2,1] = odometry[counter2,2]




			#Uncomment the following section if you want to see the extraction of velocities from groundtruth file
			#Handling first entry
			#if counter2==0:
			#	Y[counter2,0] = 1 #calculates Y matrix once - v,w
			#	Y[counter2,1] = 1

			#Handling last entry
			#elif counter2==len(odometry)-2 or counter2==len(odometry)-1:
			#	counter3 = counter2 - 2

			#	delta_t = odometry[counter3+1,0]-odometry[counter3,0]
			#	pred_v = sqrt((groundtruth_closest[counter3+1,1]-groundtruth_closest[counter3,1])**2 + (groundtruth_closest[counter3+1,2]-groundtruth_closest[counter3,2])**2) / delta_t
			#	pred_w = (groundtruth_closest[counter3+1,3]-groundtruth_closest[counter3,3]) / delta_t


			#	if fabs(groundtruth_closest[counter3,3]-groundtruth_closest[counter3+1,3]) < 0.01:
			#		actual_velocities = opt.fsolve(kinematic_model_no_w,(pred_v,0),args=counter3,epsfcn=1e-3)#system of non-linear eqns -> actual velocities
			#	else:
			#		actual_velocities = opt.fsolve(kinematic_model,(pred_v,pred_w),args=counter3,epsfcn=1e-3)#system of non-linear eqns -> actual velocities


			#	Y[counter2,0] = actual_velocities[0] #calculates Y matrix once - v,w
			#	Y[counter2,1] = actual_velocities[1]

			#Handling every other entry
			#else:

			#	delta_t = odometry[counter2+1,0]-odometry[counter2,0]
			#	pred_v = sqrt((groundtruth_closest[counter2+1,1]-groundtruth_closest[counter2,1])**2 + (groundtruth_closest[counter2+1,2]-groundtruth_closest[counter2,2])**2) / delta_t
			#	pred_w = (groundtruth_closest[counter2+1,3]-groundtruth_closest[counter2,3]) / delta_t


			#	if fabs(groundtruth_closest[counter2,3]-groundtruth_closest[counter2+1,3]) < 0.01:
			#		actual_velocities = opt.fsolve(kinematic_model_no_w,(pred_v,0),args=counter2,epsfcn=1e-3)#system of non-linear eqns -> actual velocities
			#		#print "Solving for v.","\t",counter2
			#	else:
			#		actual_velocities = opt.fsolve(kinematic_model,(pred_v,pred_w),args=counter2,epsfcn=1e-3)#system of non-linear eqns -> actual velocities
			#		#print "Solving for v and w.","\t",counter2


			#	Y[counter2,0] = actual_velocities[0] #calculates Y matrix once - v,w
			#	Y[counter2,1] = actual_velocities[1]


			#	if (counter2 % 100) == 0:
			#		print "Initializing X and Y arrays, element",counter2,"of",len(odometry)



			#Updates previously determined Y matrix in order to account for inaccuracies in fsolve()
			v = Y[counter2,0] #translation speed
			w = Y[counter2,1] #rotational speed
			if counter2==len(odometry)-1:
				delta_t = odometry[counter2,0]-odometry[counter2-1,0]
			else:
				delta_t = odometry[counter2+1,0]-odometry[counter2,0] #time step


			if fabs(v)>1:
				v = sqrt((groundtruth_closest[counter2+1,1]-groundtruth_closest[counter2,1])**2 + (groundtruth_closest[counter2+1,2]-groundtruth_closest[counter2,2])**2) / delta_t

			if fabs(w)>1:
				w = (groundtruth_closest[counter2+1,3]-groundtruth_closest[counter2,3]) / delta_t


			if fabs(v)>5:
				v = 0
			if fabs(w)>5:
				w = 0


			Y_array[counter2,0] = v
			Y_array[counter2,1] = w



		Xrow = np.array([[X[counter2,0],X[counter2,1]]]) #creates Xrow in the correct format

		W[counter2,counter2] = Kernel_Gaussian(Xrow,q) #W matrix gets updated each time there is a new query point


		counter2 = counter2+1



	#Stores X and Y arrays to .txt file
	#if counter==1:
	#	np.savetxt('Y_array_11.txt', Y, delimiter='\t')
	#	print "Done saving file."


	#Predicted target function output calculating velocities for each query point
	returned_v = Predicted_Target_Function_Output(q,W,X,Y_array)

	vels_pred[counter,0] = q[0,0] #odometry v
	vels_pred[counter,1] = q[1,0] #odometry w
	vels_pred[counter,2] = returned_v[0,0] #predicted target function v
	vels_pred[counter,3] = returned_v[0,1] #predicted target function w

	print counter,"\t","V:",round(vels_pred[counter,0],5),"\t","W:",round(vels_pred[counter,1],5)

	counter = counter+1






#Saving Predicted Velocities to .txt files
np.savetxt('Predicted_Velocities_0.txt', vels_pred, delimiter='\t', fmt='%1.4f')




#Figure 2/3 - Relationship Between Extracted Groundtruth Velocities and Odometry Velocities
v_odo = X[:,0]
w_odo = X[:,1]

v_grnd = Y_array[:,0]
w_grnd = Y_array[:,1]



plt.figure(2)
plt.scatter(v_odo, v_grnd, label='Velocity')
plt.title('Relationship Between Extracted Groundtruth Translational Velocities and Odometry')
plt.xlabel('Odometry Translational Velocities [m/s]')
plt.ylabel('Groundtruth Translational Velocities [m/s]')
plt.legend(loc=0)
plt.grid()
#plt.show()


plt.figure(3)
plt.scatter(w_odo, w_grnd, label='Angular Velocity')
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

			if fabs(v_odo - vels_pred[j,0])<0.001 and fabs(w_odo - vels_pred[j,1])<0.001:
				v = vels_pred[j,2]
				w = vels_pred[j,3]

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

			if fabs(v_odo - vels_pred[j,0])<0.001 and fabs(w_odo - vels_pred[j,1])<0.001:
				v = vels_pred[j,2]
				w = vels_pred[j,3]

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
#np.savetxt('Robot_Position_Predicted_13.txt', robot_position_pred, delimiter='\t', fmt='%1.4f')



#Predicted robot position for each query point, based on LWLR on velocities
x_obs = robot_position_pred[:,0]
y_obs = robot_position_pred[:,1]


#Figure 4 - Performance of LWLR on Robot Dataset
plt.figure(4)
plt.plot(x1, y1, label='Robot Groundtruth')
plt.plot(x, y, label='Robot Position')
plt.plot(x_obs, y_obs, label='Improved Robot Position')
plt.title('Robot Position Improvement from Learned Velocities using LWLR')
plt.xlabel('X-position [m]')
plt.ylabel('Y-position [m]')
plt.legend(loc=0)
plt.grid()
#plt.show()















#Positional Error from Motion Model/Simulated Controller on Entire Robot Dataset before applying Machine Learning
robot_position = np.zeros((odometry.size,3))

i = 0
distance=0
total_distance=0
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


		if v==0 and w==0:
			v=0
			w=0
		elif fabs(v-0.142)<0.005 and w==0:
			v=0.13518
			w=0.00073
		elif fabs(v-0.165)<0.005 and fabs(w-0.902)<0.1:
			v=0.14944
			w=0.53973
		elif fabs(v-0.165)<0.005 and fabs(w+1.003)<0.1:
			v=0.1416
			w=-0.53751
		else:
			v=0
			w=0


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


		if v==0 and w==0:
			v=0
			w=0
		elif fabs(v-0.142)<0.005 and w==0:
			v=0.13518
			w=0
		elif fabs(v-0.165)<0.005 and fabs(w-0.902)<0.1:
			v=0.14944
			w=0.58
		elif fabs(v-0.165)<0.005 and fabs(w+1.003)<0.1:
			v=0.1416
			w=-0.59
		else:
			v=0
			w=0

		

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


	distance = sqrt((groundtruth_closest[i,1]-robot_position[i,0])**2 + (groundtruth_closest[i,2]-robot_position[i,1])**2)
	total_distance = total_distance + distance

	i = i+1

print "\n"
print "Target to Beat - Average Euclidean Distance off Groundtruth:","\t",total_distance_odo/len(odometry),"\n"
print "Current Found - Average Euclidean Distance off Groundtruth:","\t",total_distance/len(odometry),"\t"
print "\t","---> Percent Improvement:","\t",round(((total_distance_odo/len(odometry)-total_distance/len(odometry))/(total_distance_odo/len(odometry)))*100,2),"%"

x_hand = robot_position[:,0]
y_hand = robot_position[:,1]


plt.figure(5)
plt.plot(x1, y1, label='Robot Groundtruth')
plt.plot(x, y, label='Robot Position')
plt.plot(x_hand, y_hand, label='Hand-tuned Robot Position')
plt.title('Robot Position Improvement from Hand-tuned Velocities after using LWLR')
plt.xlabel('X-position [m]')
plt.ylabel('Y-position [m]')
plt.legend(loc=0)
plt.grid()
plt.show()
