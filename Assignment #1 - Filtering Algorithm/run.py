import numpy as np
import math
import matplotlib.pyplot as plt
import random


# Reading data from .dat files into separate arrays
barcodes = np.loadtxt('ds1_Barcodes.dat', dtype='int') # Subject#, Barcode#
odometry = np.loadtxt('ds1_Odometry.dat') # Time, v, w
groundtruth = np.loadtxt('ds1_Groundtruth.dat') # Time, x, y, theta
measurement = np.loadtxt('ds1_Measurement.dat') # Time, Subject#, range, bearing
landmark_groundtruth = np.loadtxt('ds1_Landmark_Groundtruth.dat') # Time, x, y, x-std-dev, y-std-dev


#1 - Design a simulated controller to estimate the position changes of robot in response to commanded speed controls
# Inputs: time, translational speed, rotational speed
# Outputs: 2-D location (x,y) and heading (theta)
	#w = w_current + w_previous
	#x = (v*t)*cos(w*t)
	#y = (v*t)*sin(w*t)
	#theta = w*t + theta_previous



#2 - Verify simulated controller on following sequence of commands
commands = np.loadtxt('commands.txt') #v,w,t

position_changes = np.zeros((5,3)) #x,y,theta
heading = np.zeros((5,1))
position = np.zeros((5+1,3))
i = 0

while i<=4:
	if i==0:
		heading[i,0] = commands[i,2]
	else:
		heading[i,0] = commands[i,2]+heading[i-1,0]

	position_changes[i,0] = commands[i,1]*math.cos(heading[i,0])
	position_changes[i,1] = commands[i,2]*math.sin(heading[i,0])
	position_changes[i,2] = heading[i,0]

	if i==0:
		position[i+1,0] = position_changes[i,0]
		position[i+1,1] = position_changes[i,1]
		position[i+1,2] = position_changes[i,2]
	else:
		position[i+1,0] = position_changes[i,0]+position[i,0]
		position[i+1,1] = position_changes[i,1]+position[i,1]
		position[i+1,2] = position_changes[i,2]
	i = i+1

x = position[:,0]
y = position[:,1]

#plt.figure(1)
#plt.plot(x, y)
#plt.title('Simulated Controller: Sequence of Commands')
#plt.xlabel('X-position [m]')
#plt.ylabel('Y-position [m]')
#plt.show()



#3 - Test simulated controller on robot odometry data and compare to robot ground truth data
#odometry = np.loadtxt('ds1_Odometry.dat') # Time, v, w
robot_position_changes = np.zeros((odometry.size,3))
robot_heading = np.zeros((odometry.size,3))
robot_position = np.zeros((odometry.size+1,3))
i = 0

while i<len(odometry):
	if i==0:
		robot_heading[i,0] = odometry[i,2]*(odometry[i+1,0]-odometry[i,0])
		robot_position_changes[i,0] = odometry[i,1]*(odometry[i+1,0]-odometry[i,0])*math.cos(robot_heading[i,0])
		robot_position_changes[i,1] = odometry[i,1]*(odometry[i+1,0]-odometry[i,0])*math.sin(robot_heading[i,0])
		robot_position_changes[i,2] = robot_heading[i,0]
		robot_position[i,0] = robot_position_changes[i,0] + 0.98038490
		robot_position[i,1] = robot_position_changes[i,1] - 4.99232180
		robot_position[i,2] = robot_position_changes[i,2] + 1.44849633
		i = i+1
	else:
		robot_heading[i,0] = odometry[i,2]*(odometry[i,0]-odometry[i-1,0]) + robot_heading[i-1,0]
		robot_position_changes[i,0] = odometry[i,1]*(odometry[i,0]-odometry[i-1,0])*math.cos(robot_heading[i,0])
		robot_position_changes[i,1] = odometry[i,1]*(odometry[i,0]-odometry[i-1,0])*math.sin(robot_heading[i,0])
		robot_position_changes[i,2] = robot_heading[i,0]
		robot_position[i,0] = robot_position_changes[i,0]+robot_position[i-1,0]
		robot_position[i,1] = robot_position_changes[i,1]+robot_position[i-1,1]
		robot_position[i,2] = robot_position_changes[i,2]+robot_position[i-1,2]
		i = i+1


x = robot_position[:,0]
y = robot_position[:,1]
x1 = groundtruth[:,1]
y1 = groundtruth[:,2]

plt.figure(2)
plt.plot(x, y, label='Robot Position')
plt.plot(x1, y1, label='Robot Ground truth')
plt.title('Simulated Controller: Robot Dataset')
plt.xlabel('X-position [m]')
plt.ylabel('Y-position [m]')
plt.legend()
#plt.show()



#4 - Describe the operation of the UKF Filter algorithm. Include any key insights, assumptions, and requirements. Use equations in your algorithm.




#5 - Design a motion model (from simulated controller) that is appropriate for UKF filter. Report all maths and reasonings. Include images for illustration when appropriate.

#Algorithm motion_model_velocity
#Outputs probability p(x_t | u_t, x_t-1) of being at x_t after executing control u_t, beginning in state x_t-1, asssuming control time delta_t
#Need to determine robot-specific motion parameters alpha1-alpha6
Prob_x_t = np.zeros((odometry.size,1))
Motion_Model = np.zeros((odometry.size,3))

alpha1 = 0.2 #robot-specific motion error parameters ???
alpha2 = 0.2
alpha3 = 0.2
alpha4 = 0.2
alpha5 = 0.2
alpha6 = 0.2

q=0

while q<len(odometry):
	if q==0: #Handling first time step
		if ((0-robot_position[q,1])*math.cos(0) - (0-robot_position[q,0])*math.sin(0))==0: #When denominator is zero
			Error_Free_Controls = 0
		else:
			Error_Free_Controls = (0.5)*(((0-robot_position[q,0])*math.cos(0) + (0-robot_position[q,1])*math.sin(0)) / ((0-robot_position[q,1])*math.cos(0) - (0-robot_position[q,0])*math.sin(0)))

		Curvature_Circle_x = (0.5)*(0+robot_position[q,0]) + (Error_Free_Controls)*(0-robot_position[q,1])
		Curvature_Circle_y = (0.5)*(0+robot_position[q,1]) + (Error_Free_Controls)*(0-robot_position[q,0])
		Curvature_Circle_radius = math.sqrt(np.power((0-Curvature_Circle_x),2) + np.power((0-Curvature_Circle_y),2))
		Heading_Change = math.atan2((robot_position[q,1]-Curvature_Circle_y), (robot_position[q,0]-Curvature_Circle_x)) - math.atan2((0-Curvature_Circle_y), (0-Curvature_Circle_x))
		Translation_Velocity_Motion = ((Heading_Change) / (odometry[q,0]-0))*(Curvature_Circle_radius)
		Angular_Velocity_Motion = ((Heading_Change) / (odometry[q,0]-0))
		Final_Rotation = ((robot_position[q,2]-0) / (odometry[q,0]-0)) - (Angular_Velocity_Motion)
		
		Prob_Translation_Velocity_Error = 1
		Prob_Rotational_Velocity_Error = 1
		Prob_Final_Rotation_Error = 1

		Prob_x_t[q,0] = Prob_Translation_Velocity_Error*Prob_Rotational_Velocity_Error*Prob_Final_Rotation_Error

		q = q+1
	else: #Handling every other time step
		if ((robot_position[q-1,1]-robot_position[q,1])*math.cos(robot_position[q-1,2]) - (robot_position[q-1,0]-robot_position[q,0])*math.sin(robot_position[q-1,2]))==0: #When denominator is zero
			Error_Free_Controls = 0
		else:
			Error_Free_Controls = (0.5)*(((robot_position[q-1,0]-robot_position[q,0])*math.cos(robot_position[q-1,2]) + (robot_position[q-1,1]-robot_position[q,1])*math.sin(robot_position[q-1,2])) / ((robot_position[q-1,1]-robot_position[q,1])*math.cos(robot_position[q-1,2]) - (robot_position[q-1,0]-robot_position[q,0])*math.sin(robot_position[q-1,2])))
		
		Curvature_Circle_x = (0.5)*(robot_position[q-1,0]+robot_position[q,0]) + (Error_Free_Controls)*(robot_position[q-1,1]-robot_position[q,1])
		Curvature_Circle_y = (0.5)*(robot_position[q-1,1]+robot_position[q,1]) + (Error_Free_Controls)*(robot_position[q-1,0]-robot_position[q,0])
		Curvature_Circle_radius = math.sqrt(np.power((robot_position[q-1,0]-Curvature_Circle_x),2) + np.power((robot_position[q-1,1]-Curvature_Circle_y),2))	
		Heading_Change = math.atan2((robot_position[q,1]-Curvature_Circle_y), (robot_position[q,0]-Curvature_Circle_x)) - math.atan2((robot_position[q-1,1]-Curvature_Circle_y), (robot_position[q-1,0]-Curvature_Circle_x))
		Translation_Velocity_Motion = ((Heading_Change) / (odometry[q,0]-odometry[q-1,0]))*(Curvature_Circle_radius)
		Angular_Velocity_Motion = ((Heading_Change) / (odometry[q,0]-odometry[q-1,0]))
		Final_Rotation = ((robot_position[q,2]-robot_position[q-1,2]) / (odometry[q,0]-odometry[q-1,0])) - (Angular_Velocity_Motion)

		if odometry[q,1]==0 and odometry[q,2]==0: #No translational or rotational velocity from previous timestep -> no movement
			Prob_Translation_Velocity_Error = 1
			Prob_Rotational_Velocity_Error = 1
			Prob_Final_Rotation_Error = 1
		else:
			Prob_Translation_Velocity_Error = ((1) / (math.sqrt(2*math.pi*((alpha1)*(math.fabs(odometry[q,1])) + (alpha2)*(math.fabs(odometry[q,2]))))))*math.exp((((-1/2)*((odometry[q,1] - Translation_Velocity_Motion)*(odometry[q,1] - Translation_Velocity_Motion))) / ((alpha1)*(math.fabs(odometry[q,1])) + (alpha2)*(math.fabs(odometry[q,2])))))
			Prob_Rotational_Velocity_Error = ((1) / (math.sqrt(2*math.pi*((alpha3)*(math.fabs(odometry[q,1])) + (alpha4)*(math.fabs(odometry[q,2]))))))*math.exp((((-1/2)*(((odometry[q,2] - Angular_Velocity_Motion)*(odometry[q,2] - Angular_Velocity_Motion))) / ((alpha3)*(math.fabs(odometry[q,1])) + (alpha4)*(math.fabs(odometry[q,2]))))))
			Prob_Final_Rotation_Error = ((1) / (math.sqrt(2*math.pi*((alpha5)*(math.fabs(odometry[q,1])) + (alpha6)*(math.fabs(odometry[q,2]))))))*math.exp((((-1/2)*(((Final_Rotation)*(Final_Rotation))) / ((alpha5)*(math.fabs(odometry[q,1])) + (alpha6)*(math.fabs(odometry[q,2]))))))

		Prob_x_t[q,0] = Prob_Translation_Velocity_Error*Prob_Rotational_Velocity_Error*Prob_Final_Rotation_Error

		q = q+1


#Algorithm sample_motion_model_velocity
New_Pose_Noise = np.zeros((odometry.size,3*12))
Mean = np.zeros((odometry.size,3))
Covariance = np.zeros((odometry.size,3))

b = 0

while b<len(odometry):
	if b==0: #Handling first time step
		Control_V_Noise = odometry[b,1] + (1./6.)*(alpha1*math.fabs(odometry[b,1])+alpha2*math.fabs(odometry[b,2]))*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)
		Control_W_Noise = odometry[b,2] + (1./6.)*(alpha3*math.fabs(odometry[b,1])+alpha4*math.fabs(odometry[b,2]))*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)
		Control_Final_Rotation_Noise = (1./6.)*(alpha5*math.fabs(odometry[b,1])+alpha6*math.fabs(odometry[b,2]))*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)*random.uniform(-1,1)

		if Control_W_Noise==0: #When denominator is zero
			New_Pose_Noise[b,0] = 0
			New_Pose_Noise[b,1] = 0
		else:		
			New_Pose_Noise[b,0] = 0 - (Control_V_Noise/Control_W_Noise)*math.sin(0) + (Control_V_Noise/Control_W_Noise)*math.sin(0+Control_W_Noise*(odometry[b+1,0]-odometry[b,0]))
			New_Pose_Noise[b,1] = 0 + (Control_V_Noise/Control_W_Noise)*math.cos(0) - (Control_V_Noise/Control_W_Noise)*math.cos(0+Control_W_Noise*(odometry[b+1,0]-odometry[b,0]))

		New_Pose_Noise[b,2] = 0 + Control_W_Noise*(odometry[b+1,0]-odometry[b,0]) + Control_Final_Rotation_Noise*(odometry[b+1,0]-odometry[b,0])
		b = b+1

	else: #Handling every other time step
		h = 0
		Sum_x = 0
		Sum_y = 0
		Sum_theta = 0

		#Obtaining 12 samples from p(x_t | u_t, x_t-1)
		while h<12:
			Control_V_Noise = odometry[b,1] + (1./6.)*(alpha1*math.fabs(odometry[b,1])+alpha2*math.fabs(odometry[b,2]))*(random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1))
			Control_W_Noise = odometry[b,2] + (1./6.)*(alpha3*math.fabs(odometry[b,1])+alpha4*math.fabs(odometry[b,2]))*(random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1))
			Control_Final_Rotation_Noise = (1./6.)*(alpha5*math.fabs(odometry[b,1])+alpha6*math.fabs(odometry[b,2]))*(random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1)+random.uniform(-1,1))

			if Control_W_Noise==0: #When denominator is zero
				New_Pose_Noise[b,(3*h)] = robot_position[b-1,0]
				New_Pose_Noise[b,(3*h+1)] = robot_position[b-1,1]
			else:
				New_Pose_Noise[b,(3*h)] = robot_position[b-1,0] - (Control_V_Noise/Control_W_Noise)*math.sin(robot_position[b-1,2]) + (Control_V_Noise/Control_W_Noise)*math.sin(robot_position[b-1,2]+Control_W_Noise*(odometry[b,0]-odometry[b-1,0]))
				New_Pose_Noise[b,(3*h+1)] = robot_position[b-1,1] + (Control_V_Noise/Control_W_Noise)*math.cos(robot_position[b-1,2]) - (Control_V_Noise/Control_W_Noise)*math.cos(robot_position[b-1,2]+Control_W_Noise*(odometry[b,0]-odometry[b-1,0]))
		
			New_Pose_Noise[b,(3*h+2)] = robot_position[b-1,2] + Control_W_Noise*(odometry[b,0]-odometry[b-1,0]) + Control_Final_Rotation_Noise*(odometry[b,0]-odometry[b-1,0])
			
			#Used for finding mean of data for each time step
			Sum_x = Sum_x + New_Pose_Noise[b,(3*h)]
			Sum_y = Sum_y + New_Pose_Noise[b,(3*h+1)]
			Sum_theta = Sum_theta + New_Pose_Noise[b,(3*h+2)]

			h = h+1

		#Calculating the Mean for each Time Step
		Mean[b,0] = Sum_x/12. #Calculating mean_x
		Mean[b,1] = Sum_y/12. #Calculating mean_y
		Mean[b,2] = Sum_theta/12. #Calculating mean_theta

		#Calculating the Covariance for each Time Step
		d = 0
		Sum_Error_x = 0
		Sum_Error_y = 0
		Sum_Error_theta = 0
		while d<12:
			Sum_Error_x = Sum_Error_x + np.power((New_Pose_Noise[b,(3*d)]-Mean[b,0]),2)
			Sum_Error_y = Sum_Error_y + np.power((New_Pose_Noise[b,(3*d+1)]-Mean[b,1]),2)
			Sum_Error_theta = Sum_Error_theta + np.power((New_Pose_Noise[b,(3*d+2)]-Mean[b,2]),2)

			d = d+1

		Covariance[b,0] = np.power((math.sqrt((1./12.)*(Sum_Error_x))),2)
		Covariance[b,1] = np.power((math.sqrt((1./12.)*(Sum_Error_y))),2)
		Covariance[b,2] = np.power((math.sqrt((1./12.)*(Sum_Error_theta))),2)

		b = b+1





#6 - Design a measurement model that is approporiate for UKF filter. Report all maths and reasonings. Include images for illustration when appropriate.
Measurement = np.zeros((len(measurement),5)) 
Prob_f_t = np.zeros((len(measurement),1)) 

e = 0 

while e<len(measurement):
	#Barcode read is for a robot ... do nothing
	if barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0] == 5 or barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0] == 14 or barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0] == 41 or barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0] == 32 or barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0] == 23:
		e = e+1

	#Barcode read is for a landmark
	else:
		#Range and heading determined from robot_position to known landmark
		Measurement[e,0] = math.sqrt(np.power((landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),1]-robot_position[e,0]),2) + np.power((landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),2]-robot_position[e,1]),2)) #range
		Measurement[e,1] = math.atan2((landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),2]-robot_position[e,1]), (landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),1]-robot_position[e,0])) #heading
		Measurement[e,2] = measurement[e,0] #time stamp
		Measurement[e,3] = landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),1] + measurement[e,2]*math.cos(robot_position[e,2]+measurement[e,3]-math.pi) #x-pos of robot based on sensor reading and landmark position
		Measurement[e,4] = landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),2] + measurement[e,2]*math.sin(robot_position[e,2]+measurement[e,3]-math.pi)  #y-pos of robot based on sensor reading and landmark position

		#Calculation of standard deviation in the range and heading measurements
		std_dev_range = (Measurement[e,0] + math.sqrt(np.power((landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),3]),2)+np.power((landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),4]),2))) / Measurement[e,0] #(measurement_range + std_dev_range) / (measurement_range)
		std_dev_heading = Measurement[e,1] - math.atan2((landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),2]-robot_position[e,1]+landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),4]), (landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),1]-robot_position[e,0]+landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),3])) #measurement_heading - math.atan2((landmark_y - robot_y + std_dev_y), (landmark_x - robot_x + std_dev_x)

		#Calculation of probability for a normal distribution
		Prob_range = (1./math.sqrt(2*math.pi*np.power(std_dev_range,2)))*math.exp(((-0.5)*((np.power((measurement[e,2]-Measurement[e,0]),2)) / (np.power(std_dev_range,2)))))
		Prob_heading  = (1./math.sqrt(2*math.pi*np.power(std_dev_heading,2)))*math.exp(((-0.5)*((np.power((measurement[e,3]-Measurement[e,1]),2)) / (np.power(std_dev_heading,2)))))

		#Probability that simulated controller is accurate based on sensor reading of landmark position
		Prob_f_t[e,0] = (Prob_range)*(Prob_heading) #removed probability_feature since the robot reads a barcode ID (p=1)
		
		e = e+1





#7 - Implement the full filter.

#Creating variables used for sigma points, predicted mean, and predicted covariance
n = 1 #n-dimensional Gaussian mean/covariance
alpha = math.exp(-3) #scaling parameter
k = 0 #scaling parameter
Beta = 2 #used to incorporate prior knowledge of the distribution of sigma points ... B=2 is optimal for Gaussian distributions
Lambda = (alpha*alpha)*(n+k) - n

Estimation_Mean = np.zeros((len(odometry),((2*n)+1))) #creating empty matrix to hold estimation mean
Estimation_Covariance = np.zeros((len(odometry),((2*n)+1))) #creating empty matrix to hold estimation covariance

Sigma_Points_x = np.zeros((len(odometry),((2*n)+1))) #creating empty matrix to hold sigma points of previous belief
Sigma_Points_y = np.zeros((len(odometry),((2*n)+1))) 
Sigma_Points_theta = np.zeros((len(odometry),((2*n)+1))) 

Sigma_Points_Prop_0 = np.zeros((len(odometry),3)) #creating empty matrix to hold sigma points propogated through noise-free state prediction (motion model)
Sigma_Points_Prop_1 = np.zeros((len(odometry),3)) 
Sigma_Points_Prop_2 = np.zeros((len(odometry),3)) 

Weight_Mean = np.zeros((len(odometry),3)) #creating empty matrix to hold weights associated with mean and covariance of Gaussian
Weight_Covariance = np.zeros((len(odometry),3)) 

Predicted_Mean = np.zeros((len(odometry),3)) #creating empty matrix to hold predicted mean and covariance of propogated sigma points
Predicted_Covariance = np.zeros((len(odometry),3))

Prediction_Noise = np.zeros((len(odometry),3)) #creating empty matrix to additive noise for predicted variance

Sigma_Points_New_x = np.zeros((len(odometry),((2*n)+1))) #creating empty matrix to hold new set of sigma points extracted from predicted Gaussian
Sigma_Points_New_y = np.zeros((len(odometry),((2*n)+1))) 
Sigma_Points_New_theta = np.zeros((len(odometry),((2*n)+1))) 

Measurement = np.zeros((len(measurement),3)) #creating empty matrix to hold measurements determined from robot_position to a known landmark
Prob_f_t = np.zeros((len(measurement),1))  #creating empty matrix to hold probability that current position is accurate based on sensor reading of landmark position

Sigma_Points_Observ_0 = np.zeros((len(odometry),3)) #creating empty matrix to hold observed sigma points from measurement model
Sigma_Points_Observ_1 = np.zeros((len(odometry),3)) 
Sigma_Points_Observ_2 = np.zeros((len(odometry),3)) 

Predicted_Observation = np.zeros((len(odometry),3)) #creating empty matrix to hold predicted observation and uncertainty
Predicted_Uncertainty = np.zeros((len(odometry),3)) 

Measurement_Noise = np.zeros((len(odometry),3)) #creating empty matrix to additive noise for predicted uncertainty

Cross_Covariance = np.zeros((len(odometry),3)) #creating empty matirx to hold cross-covariance between state and observation

Kalman_Gain = np.zeros((len(odometry),1)) #creating empty matirx to hold kalman gain

Estimation_Mean = np.zeros((len(odometry),3)) #creating empty matirx to hold estimation mean and covariance
Estimation_Covariance = np.zeros((len(odometry),3)) 


w = 0

while w<len(odometry):

	#Handling first time step ... do nothing
	if w==0:
		Estimation_Mean[w,0] = robot_position[w,0] #filling in estimation mean for first time step, to only be used in next time step
		Estimation_Mean[w,1] = robot_position[w,1]
		Estimation_Mean[w,2] = robot_position[w,2]

		Estimation_Covariance[w,0] = 0 #filling in estimation covariance for first time step, to only be used in next time step
		Estimation_Covariance[w,1] = 0
		Estimation_Covariance[w,2] = 0

		w=w+1 

	#Handling every other time step, after t=0
	else:	

		#Deterministically extracting sigma points & propogating through noise-free state prediction (motion model)
		y = 0
		while y<3:
			#Deterministically extracting 1st set of sigma points for n=0
			if y==0:
				Sigma_Points_x[w-1,y] = Estimation_Mean[w-1,0]
				Sigma_Points_y[w-1,y] = Estimation_Mean[w-1,1]
				Sigma_Points_theta[w-1,y] = Estimation_Mean[w-1,2]

				#Propogating through noise-free motion model for 1st set of Sigma Points
				if odometry[w,2]==0:
					Sigma_Points_Prop_0[w,0] = Sigma_Points_x[w-1,y] + (odometry[w,1]*(odometry[w,0]-odometry[w-1,0]))*math.cos(Sigma_Points_theta[w-1,y]) #x
					Sigma_Points_Prop_0[w,1] = Sigma_Points_y[w-1,y] + (odometry[w,1]*(odometry[w,0]-odometry[w-1,0]))*math.sin(Sigma_Points_theta[w-1,y]) #y
					Sigma_Points_Prop_0[w,2] = Sigma_Points_theta[w-1,y] #theta
				else:
					Sigma_Points_Prop_0[w,0] = Sigma_Points_x[w-1,y] - (odometry[w,1])/(odometry[w,2])*math.sin(Sigma_Points_theta[w-1,y]) + (odometry[w,1]/odometry[w,2])*math.sin(Sigma_Points_theta[w-1,y] + odometry[w,2]*(odometry[w,0]-odometry[w-1,0])) #x
					Sigma_Points_Prop_0[w,1] = Sigma_Points_y[w-1,y] + (odometry[w,1]/odometry[w,2])*math.cos(Sigma_Points_theta[w-1,y]) - (odometry[w,1]/odometry[w,2])*math.cos(Sigma_Points_theta[w-1,y] + odometry[w,2]*(odometry[w,0]-odometry[w-1,0])) #y
					Sigma_Points_Prop_0[w,2] = Sigma_Points_theta[w-1,y] + odometry[w,2]*(odometry[w,0]-odometry[w-1,0]) #theta

				#Calculating weights for computing the mean and recovering the covariance for n=0
				Weight_Mean[w,y] = (Lambda / (n + Lambda))
				Weight_Covariance[w,y] = (Lambda / (n + Lambda)) + (1 - alpha*alpha + Beta)
			
			#Deterministically extracting 2nd set of sigma points for n=1
			elif y==1:
				Sigma_Points_x[w-1,y] = Estimation_Mean[w-1,0] + math.sqrt((n+Lambda)*(Estimation_Covariance[w-1,0]))
				Sigma_Points_y[w-1,y] = Estimation_Mean[w-1,1] + math.sqrt((n+Lambda)*(Estimation_Covariance[w-1,1]))
				Sigma_Points_theta[w-1,y] = Estimation_Mean[w-1,2] + math.sqrt((n+Lambda)*(Estimation_Covariance[w-1,2]))

				#Propogating through noise-free motion model for 2nd set of Sigma Points
				if odometry[w,2]==0:
					Sigma_Points_Prop_1[w,0] = Sigma_Points_x[w-1,y] + (odometry[w,1]*(odometry[w,0]-odometry[w-1,0]))*math.cos(Sigma_Points_theta[w-1,y]) #x
					Sigma_Points_Prop_1[w,1] = Sigma_Points_y[w-1,y] + (odometry[w,1]*(odometry[w,0]-odometry[w-1,0]))*math.sin(Sigma_Points_theta[w-1,y]) #y
					Sigma_Points_Prop_1[w,2] = Sigma_Points_theta[w-1,y] #theta
				else:
					Sigma_Points_Prop_1[w,0] = Sigma_Points_x[w-1,y] - (odometry[w,1]/odometry[w,2])*math.sin(Sigma_Points_theta[w-1,y]) + (odometry[w,1]/odometry[w,2])*math.sin(Sigma_Points_theta[w-1,y] + odometry[w,2]*(odometry[w,0]-odometry[w-1,0])) #x
					Sigma_Points_Prop_1[w,1] = Sigma_Points_y[w-1,y] + (odometry[w,1]/odometry[w,2])*math.cos(Sigma_Points_theta[w-1,y]) - (odometry[w,1]/odometry[w,2])*math.cos(Sigma_Points_theta[w-1,y] + odometry[w,2]*(odometry[w,0]-odometry[w-1,0])) #y
					Sigma_Points_Prop_1[w,2] = Sigma_Points_theta[w-1,y] + odometry[w,2]*(odometry[w,0]-odometry[w-1,0]) #theta

				#Calculating weights for computing the mean and recovering the covariance for n=1
				Weight_Mean[w,y] = (1 / (2*(n + Lambda)))
				Weight_Covariance[w,y] = (1 / (2*(n + Lambda)))
				
			#Deterministically extracting 3rd set of sigma points for n=2		
			else:
				Sigma_Points_x[w-1,y] = Estimation_Mean[w-1,0] - math.sqrt((n+Lambda)*(Estimation_Covariance[w-1,0]))
				Sigma_Points_y[w-1,y] = Estimation_Mean[w-1,1] - math.sqrt((n+Lambda)*(Estimation_Covariance[w-1,1]))
				Sigma_Points_theta[w-1,y] = Estimation_Mean[w-1,2] - math.sqrt((n+Lambda)*(Estimation_Covariance[w-1,2]))

				#Propogating through noise-free motion model for 3rd set of Sigma Points
				if odometry[w,2]==0:
					Sigma_Points_Prop_2[w,0] = Sigma_Points_x[w-1,y] + (odometry[w,1]*(odometry[w,0]-odometry[w-1,0]))*math.cos(Sigma_Points_theta[w-1,y]) #x
					Sigma_Points_Prop_2[w,1] = Sigma_Points_y[w-1,y] + (odometry[w,1]*(odometry[w,0]-odometry[w-1,0]))*math.sin(Sigma_Points_theta[w-1,y]) #y
					Sigma_Points_Prop_2[w,2] = Sigma_Points_theta[w-1,y] #theta
				else:
					Sigma_Points_Prop_2[w,0] = Sigma_Points_x[w-1,y] - (odometry[w,1]/odometry[w,2])*math.sin(Sigma_Points_theta[w-1,y]) + (odometry[w,1]/odometry[w,2])*math.sin(Sigma_Points_theta[w-1,y] + odometry[w,2]*(odometry[w,0]-odometry[w-1,0])) #x
					Sigma_Points_Prop_2[w,1] = Sigma_Points_y[w-1,y] + (odometry[w,1]/odometry[w,2])*math.cos(Sigma_Points_theta[w-1,y]) - (odometry[w,1]/odometry[w,2])*math.cos(Sigma_Points_theta[w-1,y] + odometry[w,2]*(odometry[w,0]-odometry[w-1,0])) #y
					Sigma_Points_Prop_2[w,2] = Sigma_Points_theta[w-1,y] + odometry[w,2]*(odometry[w,0]-odometry[w-1,0]) #theta

				#Calculating weights for computing the mean and recovering the covariance for n=2
				Weight_Mean[w,y] = (1 / (2*(n + Lambda)))
				Weight_Covariance[w,y] = (1 / (2*(n + Lambda)))
			y = y+1


		#Predicted Mean and Predicted Covariance
		Predicted_Mean[w] = Weight_Mean[w,0]*Sigma_Points_Prop_0[w] + Weight_Mean[w,1]*Sigma_Points_Prop_1[w] + Weight_Mean[w,2]*Sigma_Points_Prop_2[w]

		Prediction_Noise[w,0] = alpha1*math.fabs(odometry[w,1]) + alpha2*math.fabs(odometry[w,2]) #???
		Prediction_Noise[w,1] = alpha3*math.fabs(odometry[w,1]) + alpha4*math.fabs(odometry[w,2]) #???
		Prediction_Noise[w,2] = alpha5*math.fabs(odometry[w,1]) + alpha6*math.fabs(odometry[w,2]) #???

		Subtract0 = np.array([[(Sigma_Points_Prop_0[w,0]-Predicted_Mean[w,0]), (Sigma_Points_Prop_0[w,1]-Predicted_Mean[w,1]), (Sigma_Points_Prop_0[w,2]-Predicted_Mean[w,2])]])
		Subtract1 = np.array([[(Sigma_Points_Prop_1[w,0]-Predicted_Mean[w,0]), (Sigma_Points_Prop_1[w,1]-Predicted_Mean[w,1]), (Sigma_Points_Prop_1[w,2]-Predicted_Mean[w,2])]])
		Subtract2 = np.array([[(Sigma_Points_Prop_2[w,0]-Predicted_Mean[w,0]), (Sigma_Points_Prop_2[w,1]-Predicted_Mean[w,1]), (Sigma_Points_Prop_2[w,2]-Predicted_Mean[w,2])]])

		Predicted_Covariance[w] = Weight_Covariance[w,0]*np.power(Subtract0,2) + Weight_Covariance[w,1]*np.multiply(Subtract0,Subtract1) + Weight_Covariance[w,2]*np.multiply(Subtract0,Subtract2) + 3*Prediction_Noise[w]


		






		#New set of sigma points from predicted Gaussian
		y = 0
		while y<3:

			#Deterministically extracting 1st set of sigma points for n=0
			if y==0:
				Sigma_Points_New_x[w,y] = Predicted_Mean[w,0]
				Sigma_Points_New_y[w,y] = Predicted_Mean[w,1]
				Sigma_Points_New_theta[w,y] = Predicted_Mean[w,2]

				#Search measurement array for time stamp closest to current time stamp in odometry
				time = measurement[(((np.abs(measurement-odometry[w,0])).argmin())/4.), 0] #time
				e = (((np.abs(measurement-odometry[w,0])).argmin())/4.) #index

				#Resulting observating sigma points calculated by passing new sigma points through measurement model
				reading = 'false'
				while reading == 'false':
					#Barcode read is for a robot ... do nothing (robots moving)
					if barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0] == 1 or barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0] == 2 or barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0] == 3 or barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0] == 4 or barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0] == 5:
						reading = 'false'
						e = e+1
					#Barcode read is for a landmark
					else:
						#Range and heading determined from robot_position to known landmark
						Measurement[e,0] = measurement[e,0] #time stamp	
						Measurement[e,1] = math.sqrt(np.power((landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),1]-Sigma_Points_New_x[w,y]),2) + np.power((landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),2]-Sigma_Points_New_y[w,y]),2)) #range
						Measurement[e,2] = math.atan2((landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),2]-Sigma_Points_New_y[w,y]), (landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),1]-Sigma_Points_New_x[w,y])) - Sigma_Points_New_theta[w,y] #heading	

						#Propogating new sigma points through measurement model for 1st set of Sigma Points
						Sigma_Points_Observ_0[w,0] = landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),1] + Measurement[e,1]*math.cos(Sigma_Points_New_theta[w,y]+Measurement[e,2]-math.pi) #x-pos of robot based on sensor reading and landmark position
						Sigma_Points_Observ_0[w,1] = landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),2] + Measurement[e,1]*math.sin(Sigma_Points_New_theta[w,y]+Measurement[e,2]-math.pi) #y-pos of robot based on sensor reading and landmark position
						Sigma_Points_Observ_0[w,2] = Sigma_Points_New_theta[w,y] #cannot determine orientation of robot from measurement of landmark

						reading = 'true'
						y = y+1

			#Deterministically extracting 2nd set of sigma points for n=1
			elif y==1:
				Sigma_Points_New_x[w,y] = Predicted_Mean[w,0] + math.sqrt((n+Lambda)*(Predicted_Covariance[w,0]))
				Sigma_Points_New_y[w,y] = Predicted_Mean[w,1] + math.sqrt((n+Lambda)*(Predicted_Covariance[w,1]))
				Sigma_Points_New_theta[w,y] = Predicted_Mean[w,2] + math.sqrt((n+Lambda)*(Predicted_Covariance[w,2]))

				#Search measurement array for time stamp closest to current time stamp in odometry
				time = measurement[(((np.abs(measurement-odometry[w,0])).argmin())/4.), 0] #time
				e = (((np.abs(measurement-odometry[w,0])).argmin())/4.) #index

				#Resulting observating sigma points calculated by passing new sigma points through measurement model
				reading = 'false'
				while reading == 'false':
					#Barcode read is for a robot ... do nothing (robots moving)
					if barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0] == 1 or barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0] == 2 or barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0] == 3 or barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0] == 4 or barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0] == 5:
						reading = 'false'
						e = e+1
					#Barcode read is for a landmark
					else:
						#Range and heading determined from robot_position to known landmark
						Measurement[e,0] = measurement[e,0] #time stamp	
						Measurement[e,1] = math.sqrt(np.power((landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),1]-Sigma_Points_New_x[w,y]),2) + np.power((landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),2]-Sigma_Points_New_y[w,y]),2)) #range
						Measurement[e,2] = math.atan2((landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),2]-Sigma_Points_New_y[w,y]), (landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),1]-Sigma_Points_New_x[w,y])) - Sigma_Points_New_theta[w,y] #heading	

						#Propogating new sigma points through measurement model for 2nd set of Sigma Points
						Sigma_Points_Observ_1[w,0] = landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),1] + Measurement[e,1]*math.cos(Sigma_Points_New_theta[w,y]+Measurement[e,2]-math.pi) #x-pos of robot based on sensor reading and landmark position
						Sigma_Points_Observ_1[w,1] = landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),2] + Measurement[e,1]*math.sin(Sigma_Points_New_theta[w,y]+Measurement[e,2]-math.pi) #y-pos of robot based on sensor reading and landmark position
						Sigma_Points_Observ_1[w,2] = Sigma_Points_New_theta[w,y] #cannot determine orientation of robot from measurement of landmark

						reading = 'true'
						y = y+1

			#Deterministically extracting 3rd set of sigma points for n=2		
			else:
				Sigma_Points_New_x[w,y] = Predicted_Mean[w,0] - math.sqrt((n+Lambda)*(Predicted_Covariance[w,0]))
				Sigma_Points_New_y[w,y] = Predicted_Mean[w,1] - math.sqrt((n+Lambda)*(Predicted_Covariance[w,1]))
				Sigma_Points_New_theta[w,y] = Predicted_Mean[w,2] - math.sqrt((n+Lambda)*(Predicted_Covariance[w,2]))

				#Search measurement array for time stamp closest to current time stamp in odometry
				time = measurement[(((np.abs(measurement-odometry[w,0])).argmin())/4.), 0] #time
				e = (((np.abs(measurement-odometry[w,0])).argmin())/4.) #index

				#Resulting observating sigma points calculated by passing new sigma points through measurement model
				reading = 'false'
				while reading == 'false':
					#Barcode read is for a robot ... do nothing (robots moving)
					if barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0] == 1 or barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0] == 2 or barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0] == 3 or barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0] == 4 or barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0] == 5:
						reading = 'false'
						e = e+1
					#Barcode read is for a landmark
					else:
						#Range and heading determined from robot_position to known landmark
						Measurement[e,0] = measurement[e,0] #time stamp	
						Measurement[e,1] = math.sqrt(np.power((landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),1]-Sigma_Points_New_x[w,y]),2) + np.power((landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),2]-Sigma_Points_New_y[w,y]),2)) #range
						Measurement[e,2] = math.atan2((landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),2]-Sigma_Points_New_y[w,y]), (landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),1]-Sigma_Points_New_x[w,y])) - Sigma_Points_New_theta[w,y] #heading	

						#Propogating new sigma points through measurement model for 3rd set of Sigma Points
						Sigma_Points_Observ_2[w,0] = landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),1] + Measurement[e,1]*math.cos(Sigma_Points_New_theta[w,y]+Measurement[e,2]-math.pi) #x-pos of robot based on sensor reading and landmark position
						Sigma_Points_Observ_2[w,1] = landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),2] + Measurement[e,1]*math.sin(Sigma_Points_New_theta[w,y]+Measurement[e,2]-math.pi) #y-pos of robot based on sensor reading and landmark position
						Sigma_Points_Observ_2[w,2] = Sigma_Points_New_theta[w,y] #cannot determine orientation of robot from measurement of landmark

						reading = 'true'
						y = y+1


		#Predicted Observation and Uncertainty
		Predicted_Observation[w] = Weight_Mean[w,0]*Sigma_Points_Observ_0[w] + Weight_Mean[w,1]*Sigma_Points_Observ_1[w] + Weight_Mean[w,2]*Sigma_Points_Observ_2[w]

		Measurement_Noise[w,0] = landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),3]
		Measurement_Noise[w,1] = landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),4]
		Measurement_Noise[w,2] = alpha5*math.fabs(odometry[w,1]) + alpha6*math.fabs(odometry[w,2]) 

		Subtract0 = np.array([[(Sigma_Points_Observ_0[w,0]-Predicted_Observation[w,0]), (Sigma_Points_Observ_0[w,1]-Predicted_Mean[w,1]), (Sigma_Points_Observ_0[w,2]-Predicted_Observation[w,2])]])
		Subtract1 = np.array([[(Sigma_Points_Observ_1[w,0]-Predicted_Observation[w,0]), (Sigma_Points_Observ_1[w,1]-Predicted_Mean[w,1]), (Sigma_Points_Observ_1[w,2]-Predicted_Observation[w,2])]])
		Subtract2 = np.array([[(Sigma_Points_Observ_2[w,0]-Predicted_Observation[w,0]), (Sigma_Points_Observ_2[w,1]-Predicted_Mean[w,1]), (Sigma_Points_Observ_2[w,2]-Predicted_Observation[w,2])]])

		Predicted_Uncertainty[w] = Weight_Covariance[w,0]*np.power(Subtract0,2) + Weight_Covariance[w,1]*np.multiply(Subtract0,Subtract1) + Weight_Covariance[w,2]*np.multiply(Subtract0,Subtract2) + 3*Measurement_Noise[w]


		#Cross Covariance between State and Observation
		Subtract0 = np.array([[(Sigma_Points_New_x[w,0]-Predicted_Mean[w,0]), (Sigma_Points_New_y[w,0]-Predicted_Mean[w,1]), (Sigma_Points_New_theta[w,0]-Predicted_Mean[w,2])]])
		Subtract1 = np.array([[(Sigma_Points_New_x[w,1]-Predicted_Mean[w,0]), (Sigma_Points_New_y[w,1]-Predicted_Mean[w,1]), (Sigma_Points_New_theta[w,1]-Predicted_Mean[w,2])]])
		Subtract2 = np.array([[(Sigma_Points_New_x[w,2]-Predicted_Mean[w,0]), (Sigma_Points_New_y[w,2]-Predicted_Mean[w,1]), (Sigma_Points_New_theta[w,2]-Predicted_Mean[w,2])]])

		Subtract3 = np.array([[(Sigma_Points_Observ_0[w,0]-Predicted_Observation[w,0]), (Sigma_Points_Observ_0[w,1]-Predicted_Mean[w,1]), (Sigma_Points_Observ_0[w,2]-Predicted_Observation[w,2])]])
		Subtract4 = np.array([[(Sigma_Points_Observ_1[w,0]-Predicted_Observation[w,0]), (Sigma_Points_Observ_1[w,1]-Predicted_Mean[w,1]), (Sigma_Points_Observ_1[w,2]-Predicted_Observation[w,2])]])
		Subtract5 = np.array([[(Sigma_Points_Observ_2[w,0]-Predicted_Observation[w,0]), (Sigma_Points_Observ_2[w,1]-Predicted_Mean[w,1]), (Sigma_Points_Observ_2[w,2]-Predicted_Observation[w,2])]])

		Cross_Covariance[w] = Weight_Covariance[w,0]*np.multiply(Subtract0,Subtract3) + Weight_Covariance[w,1]*np.multiply(Subtract1,Subtract3) + Weight_Covariance[w,2]*np.multiply(Subtract2,Subtract3)


		#Calculating of Kalman gain
		Kalman_Gain[w,0] = Cross_Covariance[w,0]*Predicted_Uncertainty[w,0] + Cross_Covariance[w,1]*Predicted_Uncertainty[w,1] + Cross_Covariance[w,2]*Predicted_Uncertainty[w,2]

		
		#Estimation Update Step for Mean and Uncertainty
		Estimation_Mean[w,0] = Predicted_Mean[w,0] + Kalman_Gain[w,0]*(((landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),1] + measurement[e,2]*math.cos(robot_heading[w,2]+measurement[e,3]-math.pi)) - (Predicted_Observation[w,0])))
		Estimation_Mean[w,1] = Predicted_Mean[w,1] + Kalman_Gain[w,0]*(((landmark_groundtruth[((barcodes[np.nonzero(barcodes[:,1] == (measurement[e,1])),0])-(6)),2] + measurement[e,2]*math.sin(robot_heading[w,2]+measurement[e,3]-math.pi)) - (Predicted_Observation[w,0])))
		Estimation_Mean[w,2] = Predicted_Mean[w,2]

		Estimation_Covariance[w,0] = Predicted_Covariance[w,0] - Kalman_Gain[w,0]*Predicted_Uncertainty[w,0]*Kalman_Gain[w,0]
		Estimation_Covariance[w,1] = Predicted_Covariance[w,1] - Kalman_Gain[w,0]*Predicted_Uncertainty[w,1]*Kalman_Gain[w,0]
		Estimation_Covariance[w,2] = Predicted_Covariance[w,2] - Kalman_Gain[w,0]*Predicted_Uncertainty[w,2]*Kalman_Gain[w,0]


		w = w+1



x = robot_position[:,0]
y = robot_position[:,1]
x1 = Estimation_Mean[:,0]
y1 = Estimation_Mean[:,1]
x2 = groundtruth[:,1]
y2 = groundtruth[:,2]

plt.figure(3)
plt.plot(x, y, label='Robot Position')
plt.plot(x1, y1, label='UKF Estimation Mean')
plt.plot(x2, y2, label='Robot Ground truth')
plt.title('UKF Filter - Full')
plt.xlabel('X-position [m]')
plt.ylabel('Y-position [m]')
plt.legend()
plt.show()
