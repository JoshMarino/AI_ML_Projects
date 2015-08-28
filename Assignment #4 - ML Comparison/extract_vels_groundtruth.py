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






#Defining Kinematic Model Function to Solve for Actual Velocities from Groundtruth File
def kinematic_model_no_w(variables,counter2):

	(w) = variables

	if (odometry[counter2,0]-groundtruth_closest[counter2,0]) < 0: #round down
		percent = (odometry[counter2,0]-groundtruth_closest[counter2-1,0]) / (groundtruth_closest[counter2,0]-groundtruth_closest[counter2-1,0])
		percent1 = (odometry[counter2+1,0]-groundtruth_closest[counter2,0]) / (groundtruth_closest[counter2+1,0]-groundtruth_closest[counter2,0])

		theta = groundtruth_closest[counter2-1,3] + percent*(groundtruth_closest[counter2,3]-groundtruth_closest[counter2-1,3])

		theta_prime = groundtruth_closest[counter2,3] + percent1*(groundtruth_closest[counter2+1,3]-groundtruth_closest[counter2,3])

	else: #round up
		percent = (odometry[counter2,0]-groundtruth_closest[counter2,0]) / (groundtruth_closest[counter2+1,0]-groundtruth_closest[counter2,0])
		percent1 = (odometry[counter2+1,0]-groundtruth_closest[counter2+1,0]) / (groundtruth_closest[counter2+2,0]-groundtruth_closest[counter2+1,0])

		theta = groundtruth_closest[counter2,3] + percent*(groundtruth_closest[counter2+1,3]-groundtruth_closest[counter2,3])

		theta_prime = groundtruth_closest[counter2+1,3] + percent1*(groundtruth_closest[counter2+2,3]-groundtruth_closest[counter2+1,3])


	delta_t = odometry[counter2+1,0]-odometry[counter2,0]

	#Groundtruth determines no rotation
	first_eq = w


	return [first_eq]


#Defining Kinematic Model Function to Solve for Actual Velocities from Groundtruth File
def kinematic_model(variables,counter2):

	(w) = variables

	if (odometry[counter2,0]-groundtruth_closest[counter2,0]) < 0: #round down
		percent = (odometry[counter2,0]-groundtruth_closest[counter2-1,0]) / (groundtruth_closest[counter2,0]-groundtruth_closest[counter2-1,0])
		percent1 = (odometry[counter2+1,0]-groundtruth_closest[counter2,0]) / (groundtruth_closest[counter2+1,0]-groundtruth_closest[counter2,0])

		theta = groundtruth_closest[counter2-1,3] + percent*(groundtruth_closest[counter2,3]-groundtruth_closest[counter2-1,3])

		theta_prime = groundtruth_closest[counter2,3] + percent1*(groundtruth_closest[counter2+1,3]-groundtruth_closest[counter2,3])

	else: #round up
		percent = (odometry[counter2,0]-groundtruth_closest[counter2,0]) / (groundtruth_closest[counter2+1,0]-groundtruth_closest[counter2,0])
		percent1 = (odometry[counter2+1,0]-groundtruth_closest[counter2+1,0]) / (groundtruth_closest[counter2+2,0]-groundtruth_closest[counter2+1,0])

		theta = groundtruth_closest[counter2,3] + percent*(groundtruth_closest[counter2+1,3]-groundtruth_closest[counter2,3])

		theta_prime = groundtruth_closest[counter2+1,3] + percent1*(groundtruth_closest[counter2+2,3]-groundtruth_closest[counter2+1,3])


	delta_t = odometry[counter2+1,0]-odometry[counter2,0]


	#Solve for v and w
	first_eq = ((theta_prime - theta) / delta_t) - w

	return [first_eq]





#Extract Velocities from Groundtruth File
Y = np.zeros((len(odometry),1))

counter = 0
while counter<3:


	#Stacked while loop to calculate X,Y once and W matrix each iteration of query points
	counter2 = 0
	while counter2<len(odometry)-1:

		#Calculates X and Y matrix fully one time
		if counter==0:
			print counter2
			#Handling first entry
			if counter2==0:
				Y[counter2,0] = odometry[counter2,2]

			#Handling last two entry
			elif counter2==len(odometry)-2 or counter2==len(odometry)-1:

				Y[counter2,0] = odometry[counter2,2]

			#Handling every other entry
			else:

				delta_t = odometry[counter2+1,0]-odometry[counter2,0]
				pred_w = (groundtruth_closest[counter2+1,3]-groundtruth_closest[counter2,3]) / delta_t


				if fabs(groundtruth_closest[counter2,3]-groundtruth_closest[counter2+1,3]) < 0.01:
					actual_velocities = opt.fsolve(kinematic_model_no_w,0,args=counter2,epsfcn=1e-3)#system of non-linear eqns -> actual velocities
					#print "Solving for v.","\t",counter2
				else:
					actual_velocities = opt.fsolve(kinematic_model,pred_w,args=counter2,epsfcn=1e-3)#system of non-linear eqns -> actual velocities
					#print "Solving for v and w.","\t",counter2


				Y[counter2,0] = actual_velocities[0] #calculates Y matrix once - w


				if (counter2 % 100) == 0:
					print "Initializing Y array, element",counter2,"of",len(odometry)


		counter2 = counter2+1



	#Stores Y array to .txt file
	if counter==1:
		np.savetxt('Y_array.txt', Y)
		print "Done saving file."


	counter = counter+1
