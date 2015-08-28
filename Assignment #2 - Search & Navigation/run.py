import numpy as np
import math
import matplotlib.pyplot as plt
import random
from math import sin,cos,atan2,sqrt,fabs,pi
from matplotlib.patches import Rectangle


# Reading landmark data from .dat files into numpy array
landmark = np.loadtxt('ds1_Landmark_Groundtruth.dat') # Time, x, y, x-std-dev, y-std-dev








#1 - Build a grid with cell sizes of 1x1m & mark each cell that contains a landmark as occupied.
Landmark_pos = np.zeros((len(landmark),2))
Landmark_cen = np.zeros((len(landmark),2))

i = 0

#Reading in landmark position to np array
while i<len(landmark):
	Landmark_pos[i,0] = landmark[i,1]
	Landmark_pos[i,1] = landmark[i,2]
	i = i+1

#If landmark is in a grid, defining a new np array to hold the values for the center of the grid
x=-2
while x<=5:
	y=-6
	while y<=6:
		i = 0
		while i<len(landmark):
			if x<Landmark_pos[i,0] and Landmark_pos[i,0]<(x+1) and y<Landmark_pos[i,1] and Landmark_pos[i,1]<(y+1):
				Landmark_cen[i,0] = x+0.5
				Landmark_cen[i,1] = y+0.5
			i = i+1
		y=y+1
	x=x+1


Landmark_x = Landmark_pos[:,0]
Landmark_y = Landmark_pos[:,1]


#Figure #1: Large Grid Cells - Occupied
fig1 = plt.figure(1)
#Setting up axis ranges and making it square
ax = fig1.gca()
ax.set_xticks(np.arange(-2,6,1))
ax.set_yticks(np.arange(-6,7,1))
plt.axis('equal')
#Plotting scatter points for each landmark position
plt.scatter(Landmark_x, Landmark_y, s=10)
#Plotting rectangles in each grid cell where a landmark is occupied
ax1 = plt.gca()
x=-2
while x<=5:
	y=-6
	while y<=6:
		i = 0
		while i<len(landmark):
			if x<Landmark_pos[i,0] and Landmark_pos[i,0]<(x+1) and y<Landmark_pos[i,1] and Landmark_pos[i,1]<(y+1):
				ax1.add_patch(Rectangle((Landmark_cen[i,0]-0.5,Landmark_cen[i,1]-0.5),1,1,facecolor="red",alpha=0.7))
			i = i+1
		y=y+1
	x=x+1
#Labeling graph and setting axis limits
plt.title('Large Grid Cells - Occupied')
plt.xlabel('X-position [m]')
plt.ylabel('Y-position [m]')
plt.xlim([-2,5])
plt.ylim([-6,6])
plt.grid()
#plt.show()










#2 - Implement an "online" version of A*, for which the set of expanded nodes contains only the current cell in which the robot is located. Design an admissable heuristic.

#Heuristic function definition to calculate f(n)
def Heuristic(Cx,Cy,Gx,Gy,prev_cost,landmark_space,grid_size):
	True_Cost = 0
	i = 0
	while i<len(landmark): #checking each landmark to see if in grid cell
		if fabs(Cx-Landmark_pos[i,0])<landmark_space and fabs(Cy-Landmark_pos[i,1])<landmark_space: #neighbor cell occupied
			True_Cost += 1000
		elif fabs(Cx+grid_size/2-Landmark_pos[i,0])<landmark_space and fabs(Cy+grid_size/2-Landmark_pos[i,1])<landmark_space: #passing through obstacle
			True_Cost += 1000
		elif fabs(Cx+grid_size/2-Landmark_pos[i,0])<landmark_space and fabs(Cy-grid_size/2-Landmark_pos[i,1])<landmark_space: #passing through obstacle
			True_Cost += 1000
		elif fabs(Cx-grid_size/2-Landmark_pos[i,0])<landmark_space and fabs(Cy+grid_size/2-Landmark_pos[i,1])<landmark_space: #passing through obstacle
			True_Cost += 1000
		elif fabs(Cx-grid_size/2-Landmark_pos[i,0])<landmark_space and fabs(Cy-grid_size/2-Landmark_pos[i,1])<landmark_space: #passing through obstacle
			True_Cost += 1000
		elif Cx>5 or Cx<-2 or Cy>6 or Cy<-6: #robot trying to move off grid
			True_Cost += 1000
		else: #unoccupied neighbor cell
			True_Cost += 1
		i = i+1

	if True_Cost>1000:
		True_Cost1 = 1000 #landmark in grid cell or moving off grid
	else:
		True_Cost1 = 1 #no landmark

	g = prev_cost #previous path cost
	h = sqrt((Cx-Gx)**2+(Cy-Gy)**2) + True_Cost1 #new Euclidean distance plus true cost
	f = g + h #previous path cost + new cost for moving up

	return (f, h)


#Determining best way to move to next node from 8 neighbor nodes
def Movement(Cx,Cy,Gx,Gy,g,grid_size,landmark_space):
	#Computing f(n)=g(n)+h(n) for each neighbor cell
	f_up, h_up = Heuristic(Cx,Cy+grid_size,Gx,Gy,g,landmark_space,grid_size) #up
	f_down, h_down = Heuristic(Cx,Cy-grid_size,Gx,Gy,g,landmark_space,grid_size) #down
	f_left, h_left = Heuristic(Cx-grid_size,Cy,Gx,Gy,g,landmark_space,grid_size) #left
	f_right, h_right = Heuristic(Cx+grid_size,Cy,Gx,Gy,g,landmark_space,grid_size) #right
	f_up_left, h_up_left = Heuristic(Cx-grid_size,Cy+grid_size,Gx,Gy,g,landmark_space,grid_size) #up-left
	f_up_right, h_up_right = Heuristic(Cx+grid_size,Cy+grid_size,Gx,Gy,g,landmark_space,grid_size) #up-right
	f_down_left, h_down_left = Heuristic(Cx-grid_size,Cy-grid_size,Gx,Gy,g,landmark_space,grid_size) #down-left
	f_down_right, h_down_right = Heuristic(Cx+grid_size,Cy-grid_size,Gx,Gy,g,landmark_space,grid_size) #down-right


	#Finding smallest heuristic value from its 8 neighbors
	f_poss = [f_up,f_down,f_left,f_right,f_up_left,f_up_right,f_down_left,f_down_right]
	f_min_index = f_poss.index(min(f_poss))
	if f_min_index==0: C = [Cx,Cy+grid_size]; g = f_up; h = h_up;
	elif f_min_index==1: C = [Cx,Cy-grid_size]; g = f_down; h = h_down;
	elif f_min_index==2: C = [Cx-grid_size,Cy]; g = f_left; h = h_left;
	elif f_min_index==3: C = [Cx+grid_size,Cy]; g = f_right; h = h_right;
	elif f_min_index==4: C = [Cx-grid_size,Cy+grid_size]; g = f_up_left; h = h_up_left;
	elif f_min_index==5: C = [Cx+grid_size,Cy+grid_size]; g = f_up_right; h = h_up_right;
	elif f_min_index==6: C = [Cx-grid_size,Cy-grid_size]; g = f_down_left; h = h_down_left;
	else: C = [Cx+grid_size,Cy-grid_size]; g = f_down_right; h = h_down_right;

	return (C, g, h)












#3 - Plan paths between the following sets of start (S) and goal (G) positions. Provide a visual display of results.
grid_size = 1
landmark_space = grid_size/2.

S1 = [0.5,-1.5] #start position
G1 = [0.5,1.5] #goal position
C1 = S1 #current node
g1 = 0 #path cost from start -> node
h1 = sqrt((S1[0]-G1[0])**2+(S1[1]-G1[1])**2) #estimate on cheapest path from node -> goal using Euclidean distance
Movement1 = C1 #tracking movement

S2 = [4.5,3.5] #start position
G2 = [4.5,-1.5] #goal position
C2 = S2 #current node
g2 = 0 #path cost from start -> node
h2 = sqrt((S2[0]-G2[0])**2+(S2[1]-G2[1])**2) #estimate on cheapest path from node -> goal using Euclidean distance
Movement2 = C2 #tracking movement

S3 = [-0.5,5.5] #start position
G3 = [1.5,-3.5] #goal position
C3 = S3 #current node
g3 = 0 #path cost from start -> node
h3 = sqrt((S3[0]-G3[0])**2+(S3[1]-G3[1])**2) #estimate on cheapest path from node -> goal using Euclidean distance
Movement3 = C3 #tracking movement


while h1>1:
	#Calculating heurisitic function f=g+h for set 1
	C1, g1, h1 = Movement(C1[0],C1[1],G1[0],G1[1],g1,grid_size,landmark_space)
	#Tracking movement for set 1
	Movement1 = np.vstack((Movement1,C1));

Movement1_x = Movement1[:,0]
Movement1_y = Movement1[:,1]

while h2>1:
	#Calculating heurisitic function f=g+h for set 2
	C2, g2, h2 = Movement(C2[0],C2[1],G2[0],G2[1],g2,grid_size,landmark_space)
	#Tracking movement for set 2
	Movement2 = np.vstack((Movement2,C2));

Movement2_x = Movement2[:,0]
Movement2_y = Movement2[:,1]

while h3>1:
	#Calculating heurisitic function f=g+h for set 3
	C3, g3, h3 = Movement(C3[0],C3[1],G3[0],G3[1],g3,grid_size,landmark_space)
	#Tracking movement for set 3
	Movement3 = np.vstack((Movement3,C3));

Movement3_x = Movement3[:,0]
Movement3_y = Movement3[:,1]


#Figure #2: Large Grid Cells - Planned Path
fig2 = plt.figure(2)
#Setting up axis ranges and making it square
ax = fig2.gca()
ax.set_xticks(np.arange(-2,6,grid_size))
ax.set_yticks(np.arange(-6,7,grid_size))
plt.axis('equal')
#Plotting scatter points for each landmark position
plt.scatter(Landmark_x, Landmark_y, s=10)
#Plotting rectangles in each grid cell where a landmark is occupied
ax1 = plt.gca()
x=-2
while x<=5:
	y=-6
	while y<=6:
		i = 0
		while i<len(landmark):
			if x<Landmark_pos[i,0] and Landmark_pos[i,0]<(x+grid_size) and y<Landmark_pos[i,1] and Landmark_pos[i,1]<(y+grid_size):
				ax1.add_patch(Rectangle((Landmark_cen[i,0]-(grid_size/2.),Landmark_cen[i,1]-(grid_size/2.)),1,1,facecolor="red",alpha=0.7))
			i = i+1
		y=y+grid_size
	x=x+grid_size
#Labeling graph and setting axis limits
plt.plot(Movement1_x,Movement1_y,label='Set 1')
plt.plot(Movement2_x,Movement2_y,label='Set 2')
plt.plot(Movement3_x,Movement3_y,label='Set 3')
plt.title('Large Grid Cells - Planned Paths')
plt.xlabel('X-position [m]')
plt.ylabel('Y-position [m]')
plt.legend()
plt.xlim([-2,5])
plt.ylim([-6,6])
plt.grid()
#plt.show()












#4 - Decrease cell size to 0.1m x 0.1m, while inflating space each landmark occupies by 0.3m in all directions (approx. to square or circle)

#If landmark is in a grid, defining a new np array to hold the values for the center of the grid
x=-2
while x<=5:
	y=-6
	while y<=6:
		i = 0
		while i<len(landmark):
			if x<Landmark_pos[i,0] and Landmark_pos[i,0]<(x+0.1) and y<Landmark_pos[i,1] and Landmark_pos[i,1]<(y+0.1):
				Landmark_cen[i,0] = x+0.05
				Landmark_cen[i,1] = y+0.05
			i = i+1
		y=y+0.1
	x=x+0.1


grid_size = 0.1
landmark_space = 0.3











#5 - Plan paths between the following sets of start (S) and goal (G) positions. Provide a visual display of your results that shows the occupied cells and planned paths.
S4 = [2.45,-3.55] #start position
G4 = [0.95,-1.55] #goal position
C4 = S4 #current node
g4 = 0 #path cost from start -> node
h4 = sqrt((S4[0]-G4[0])**2+(S4[1]-G4[1])**2) #estimate on cheapest path from node -> goal using Manhattan distance
Movement4 = C4 #tracking movement

S5 = [4.95,-0.05] #start position
G5 = [2.45,0.25] #goal position
C5 = S5 #current node
g5 = 0 #path cost from start -> node
h5 = sqrt((S5[0]-G5[0])**2+(S5[1]-G5[1])**2) #estimate on cheapest path from node -> goal using Manhattan distance
Movement5 = C5 #tracking movement

S6 = [-0.55,1.45] #start position
G6 = [1.95,3.95] #goal position
C6 = S6 #current node
g6 = 0 #path cost from start -> node
h6 = sqrt((S6[0]-G6[0])**2+(S6[1]-G6[1])**2) #estimate on cheapest path from node -> goal using Manhattan distance
Movement6 = C6 #tracking movement


#Path planning for set #1 (4)
while h4>1.01:
	#Calculating heurisitic function f=g+h for set 4
	C4, g4, h4 = Movement(C4[0],C4[1],G4[0],G4[1],g4,grid_size,landmark_space)
	#Tracking movement for set 4
	Movement4 = np.vstack((Movement4,C4));

Movement4_x = Movement4[:,0]
Movement4_y = Movement4[:,1]


#Path planning for set #2 (5)
i=0
while h5>1.01:
	#Calculating heurisitic function f=g+h for set 1
	C5, g5, h5 = Movement(C5[0],C5[1],G5[0],G5[1],g5,grid_size,landmark_space)

	if i>1:
		#When stuck behind obstacle, clear obstacle, then continue with online A*
		if fabs(Movement5[i,0] - Movement5[i-2,0]) < 0.001 and fabs(Movement5[i,1] - Movement5[i-2,1]) < 0.001:
			#Calculating heuristic value for each neighboring cell before moved into oscillatory state
			f_up, h_up = Heuristic(Movement5[i-1,0],Movement5[i-1,1]+grid_size,G5[0],G5[1],g5,landmark_space,grid_size) #up
			f_down, h_down = Heuristic(Movement5[i-1,0],Movement5[i-1,1]-grid_size,G5[0],G5[1],g5,landmark_space,grid_size) #down
			f_left, h_left = Heuristic(Movement5[i-1,0]-grid_size,Movement5[i-1,1],G5[0],G5[1],g5,landmark_space,grid_size) #left
			f_right, h_right = Heuristic(Movement5[i-1,0]+grid_size,Movement5[i-1,1],G5[0],G5[1],g5,landmark_space,grid_size) #right
			f_up_left, h_up_left = Heuristic(Movement5[i-1,0]-grid_size,Movement5[i-1,1]+grid_size,G5[0],G5[1],g5,landmark_space,grid_size) #up-left
			f_up_right, h_up_right = Heuristic(Movement5[i-1,0]+grid_size,Movement5[i-1,1]+grid_size,G5[0],G5[1],g5,landmark_space,grid_size) #up-right
			f_down_left, h_down_left = Heuristic(Movement5[i-1,0]-grid_size,Movement5[i-1,1]-grid_size,G5[0],G5[1],g5,landmark_space,grid_size) #down-left
			f_down_right, h_down_right = Heuristic(Movement5[i-1,0]+grid_size,Movement5[i-1,1]-grid_size,G5[0],G5[1],g5,landmark_space,grid_size) #down-right

			if f_up>1000: obstacle="up"; f_obs=f_up; h_obs=h_up;
			if f_down>1000: obstacle="down"; f_obs=f_down; h_obs=h_down;
			if f_left>1000: obstacle="left"; f_obs=f_left; h_obs=h_left;
			if f_right>1000: obstacle="right"; f_obs=f_right; h_obs=h_right;

			j = i+1

			#Moving robot out of the way of the obstacle
			while j<i+5:
				C5 = [Movement5[j-1,0]+(Movement5[j-1,0]-Movement5[j-3,0]),Movement5[j-1,1]+(Movement5[j-1,1]-Movement5[j-3,1])]; 
				g5 = f_obs; 
				h5 = h_obs;

				f_up, h_up = Heuristic(Movement5[j-1,0],Movement5[j-1,1]+grid_size,G5[0],G5[1],g5-1000,landmark_space,grid_size) #up
				f_down, h_down = Heuristic(Movement5[j-1,0],Movement5[j-1,1]-grid_size,G5[0],G5[1],g5-1000,landmark_space,grid_size) #down
				f_left, h_left = Heuristic(Movement5[j-1,0]-grid_size,Movement5[j-1,1],G5[0],G5[1],g5-1000,landmark_space,grid_size) #left
				f_right, h_right = Heuristic(Movement5[j-1,0]+grid_size,Movement5[j-1,1],G5[0],G5[1],g5-1000,landmark_space,grid_size) #right
				f_up_left, h_up_left = Heuristic(Movement5[j-1,0]-grid_size,Movement5[j-1,1]+grid_size,G5[0],G5[1],g5-1000,landmark_space,grid_size) #up-left
				f_up_right, h_up_right = Heuristic(Movement5[j-1,0]+grid_size,Movement5[j-1,1]+grid_size,G5[0],G5[1],g5-1000,landmark_space,grid_size) #up-right
				f_down_left, h_down_left = Heuristic(Movement5[j-1,0]-grid_size,Movement5[j-1,1]-grid_size,G5[0],G5[1],g5-1000,landmark_space,grid_size) #down-left
				f_down_right, h_down_right = Heuristic(Movement5[j-1,0]+grid_size,Movement5[j-1,1]-grid_size,G5[0],G5[1],g5-1000,landmark_space,grid_size) #down-right

				if f_up>1000: obstacle="up"; f_obs=f_up; h_obs=h_up;
				elif f_down>1000: obstacle="down"; f_obs=f_down; h_obs=h_down;
				elif f_left>1000: obstacle="left"; f_obs=f_left; h_obs=h_left;
				elif f_right>1000: obstacle="right"; f_obs=f_right; h_obs=h_right;

				Movement5 = np.vstack((Movement5,C5));

				j = j+1

			i = j-1

	#Tracking movement for set 5
	Movement5 = np.vstack((Movement5,C5));
	i=i+1

Movement5_x = Movement5[:,0]
Movement5_y = Movement5[:,1]


#Path planning for set #3 (6)
while h6>1.01:
	#Calculating heurisitic function f=g+h for set 6
	C6, g6, h6 = Movement(C6[0],C6[1],G6[0],G6[1],g6,grid_size,landmark_space)
	#Tracking movement for set 6
	Movement6 = np.vstack((Movement6,C6));

Movement6_x = Movement6[:,0]
Movement6_y = Movement6[:,1]



#Figure #3: Small Grid Cells - Planned Path
fig3 = plt.figure(3)
#Setting up axis ranges and making it square/equal
ax = fig3.gca()
ax.set_xticks(np.arange(-2,6,1))
ax.set_yticks(np.arange(-6,7,0.5))
plt.axis('equal')
#Plotting scatter points for each landmark position
plt.scatter(Landmark_x, Landmark_y, s=10)
#Plotting rectangles in each grid cell where a landmark is occupied
ax1 = plt.gca()
x=-2
while x<=5:
	y=-6
	while y<=6:
		i = 0
		while i<len(landmark):
			if x<Landmark_pos[i,0] and Landmark_pos[i,0]<(x+0.3) and y<Landmark_pos[i,1] and Landmark_pos[i,1]<(y+0.3):
				ax1.add_patch(Rectangle((Landmark_cen[i,0]-(0.3),Landmark_cen[i,1]-(0.3)),0.6,0.6,facecolor="red",alpha=0.7))
			i = i+1
		y=y+grid_size
	x=x+grid_size
#Labeling graph and setting axis limits
plt.plot(Movement4_x,Movement4_y,label='Set 1') #Set #1
plt.plot(Movement5_x,Movement5_y,label='Set 2') #Set #2
plt.plot(Movement6_x,Movement6_y,label='Set 3') #Set #3
plt.title('Small Grid Cells - Planned Paths')
plt.xlabel('X-position [m]')
plt.ylabel('Y-position [m]')
plt.legend()
plt.xlim([-2,5])
plt.ylim([-6,6])
plt.grid()
#plt.show()











#Part 6 - Design an inverse kinematic controller able to drive a path generated by pseudo-A* implementation, based on model developed in Step #1 of Homework #0. 

#Velocity function to return set of velocities to get from current position to target position
def Velocity(x,y,x_new,y_new,max_accel,time_step):
	#Maximum translational acceleration, per time step
	dv_max = max_accel #m/s/s
	dt = time_step #sec

	#Distance traveled from current position to target position
	total_distance = sqrt((x-x_new)**2 + (y-y_new)**2)

	Velocity = [0]
	distance_traveled = 0
	j = 0

	#Determining number of time steps to get halfway to target position
	while fabs(distance_traveled) <= total_distance/2.:
		v_new = dv_max*dt*j
		distance_traveled = distance_traveled + v_new*dt
		j = j+1
	dv_actual = (total_distance) / ((sum(range(j))+sum(range(j-1)))*dt*dt)


	distance_traveled = 0
	i = 0
	v_new = 0

	#Tracking velocities to get from current position to halfway between current and target position
	while i<j:
		v_new = dv_actual*dt*i
		distance_traveled = distance_traveled + v_new*dt
		Velocity = np.vstack((Velocity,v_new))
		i = i+1

	#Tracking velocities to get from halway between current and target position to target position
	while i<(2*j-1):
		v_new = v_new-dv_actual*dt
		distance_traveled = distance_traveled + v_new*dt
		Velocity = np.vstack((Velocity,v_new))
		i = i+1

	return Velocity



#Rotational velocity function to return set of rotational velocities to rotate from current heading to heading for target position
def Rotational_Velocity(x,y,x_new,y_new,theta_prev,theta_init,max_accel,time_step):
	#Maximum rotational acceleration, per time step
	dw_max = max_accel #m/s/s
	dt = time_step #sec

	#Rotation traveled from current position to target position
	total_rotation = atan2((y_new-y),(x_new-x)) - (theta_prev - theta_init)

	Rot_Vel = [0]
	rotation_traveled = 0
	j = 0

	#Determining number of time steps to get halfway to target position
	while fabs(rotation_traveled) <= fabs(total_rotation/2.):
		w_new = dw_max*dt*j
		rotation_traveled = rotation_traveled + w_new*dt
		j = j+1
	dw_actual = (total_rotation) / ((sum(range(j))+sum(range(j-1)))*dt*dt)


	rotation_traveled = 0
	i = 0
	w_new = 0

	#Tracking velocities to get from current position to halfway between current and target position
	while i<j:
		w_new = dw_actual*dt*i
		rotation_traveled = rotation_traveled + w_new*dt
		Rot_Vel = np.vstack((Rot_Vel,w_new))
		i = i+1

	#Tracking velocities to get from halway between current and target position to target position
	while i<(2*j-1):
		w_new = w_new-dw_actual*dt
		rotation_traveled = rotation_traveled + w_new*dt
		Rot_Vel = np.vstack((Rot_Vel,w_new))
		i = i+1

	return (Rot_Vel, rotation_traveled)










#Part 7 - Drive the paths generated in Step #5. The robot should always start with v=0, w=0, theta=-pi/2. Provide a visual display of your results, including heading and position of your robot at each step of the execution.
Heading4 = np.zeros((len(Movement4),1)) #empty array for storing the heading of robot
Heading5 = np.zeros((len(Movement5),1))
Heading6 = np.zeros((len(Movement6),1))

dv_max = 0.288
dw_max = 5.579
delta_t = 0.1
theta_init = -pi/2
Heading4[0] = theta_init


#Figure #4: Small Grid Cells - Driving Planned Path
fig4 = plt.figure(4)
#Setting up axis ranges and making it square
ax = fig4.gca()
ax.set_xticks(np.arange(-2,6,1))
ax.set_yticks(np.arange(-6,7,0.5))
plt.axis('equal')
#Plotting scatter points for each landmark position
plt.scatter(Landmark_x, Landmark_y, s=10)
#Plotting rectangles in each grid cell where a landmark is occupied
ax1 = plt.gca()
x=-2
while x<=5:
	y=-6
	while y<=6:
		i = 0
		while i<len(landmark):
			if x<Landmark_pos[i,0] and Landmark_pos[i,0]<(x+0.3) and y<Landmark_pos[i,1] and Landmark_pos[i,1]<(y+0.3):
				ax1.add_patch(Rectangle((Landmark_cen[i,0]-(0.3),Landmark_cen[i,1]-(0.3)),0.6,0.6,facecolor="red",alpha=0.7))
			i = i+1
		y=y+grid_size
	x=x+grid_size
#Labeling graph and setting axis limits
plt.ion()
plt.plot(Movement4_x,Movement4_y,label='Set 1')
plt.plot(Movement5_x,Movement5_y,label='Set 2')
plt.plot(Movement6_x,Movement6_y,label='Set 3')
plt.title('Small Grid Cells - Driving Planned Paths')
plt.xlabel('X-position [m]')
plt.ylabel('Y-position [m]')
plt.legend()
plt.xlim([-2,5])
plt.ylim([-6,6])
plt.grid()
plt.show()




#Driving path for each node of the A* path planning algorithm
#Set #1 (4)
i = 0
while i<len(Movement4)-1:
	#Calling the Rotational Velocity function with inputs of x,y,theta,x_target,y_target and outputting array of rotational velocities to accomplish this
	Rot_Vel4, rotation_traveled4 = Rotational_Velocity(Movement4[i,0],Movement4[i,1],Movement4[i+1,0],Movement4[i+1,1],Heading4[i],theta_init,dw_max,delta_t)
	Heading4[i+1] = atan2((Movement4[i+1,1]-Movement4[i,1]),(Movement4[i+1,0]-Movement4[i,0])) + theta_init #updating current heading from theta=0

	#Setting up two arrays to hold robot heading and position for each movement
	robot_heading4 = np.zeros((len(Rot_Vel4),1))
	robot_position4 = np.zeros((len(Rot_Vel4),3))

	#Calculating and storing heading of robot based on array of rotational velocities
	j4 = 0
	while j4<len(Rot_Vel4):
		if j4==0: #first element of array when w=0
			robot_heading4[j4,0] = Rot_Vel4[j4]*delta_t

			robot_position4[j4,0] = Movement4[i,0]
			robot_position4[j4,1] = Movement4[i,1]
			robot_position4[j4,2] = robot_heading4[j4,0]

			j4 = j4+1
		else: #all others
			robot_heading4[j4,0] = Rot_Vel4[j4]*delta_t + robot_heading4[j4-1,0]

			robot_position4[j4,0] = Movement4[i,0]
			robot_position4[j4,1] = Movement4[i,1]
			robot_position4[j4,2] = robot_position4[j4-1,2] + robot_heading4[j4,0]

			j4 = j4+1


	#Calling the Velocity function with inputs of x,y,x_target,y_target and outputting array of velocities to accomplish this
	Vel4 = Velocity(Movement4[i,0],Movement4[i,1],Movement4[i+1,0],Movement4[i+1,1],dv_max,delta_t)

	#Setting up two arrays to hold robot position changes and overall position for each movement
	robot_position_changes41 = np.zeros((len(Vel4),3))
	robot_position41 = np.zeros((len(Vel4),3))

	#Calculating and storing pose of robot based on array of velocities
	k4 = 0
	while k4<len(Vel4):
		if k4==0: #first element of array when v=0
			robot_position_changes41[k4,0] = Vel4[k4]*delta_t*cos(Heading4[i])
			robot_position_changes41[k4,1] = Vel4[k4]*delta_t*sin(Heading4[i])
			robot_position_changes41[k4,2] = 0
			robot_position41[k4,0] = robot_position_changes41[k4,0] + Movement4[i,0] #accounting for initial position
			robot_position41[k4,1] = robot_position_changes41[k4,1] + Movement4[i,1]
			robot_position41[k4,2] = robot_heading4[j4-1,0]

			#Plotting movement for planned path
			plt.scatter(robot_position41[k4,0],robot_position41[k4,1],label='Path Driven #1')
			plt.draw()
			plt.pause(0.000001)

			k4 = k4+1
		else: #all others
			robot_position_changes41[k4,0] = Vel4[k4]*delta_t*cos(Heading4[i+1]-theta_init)
			robot_position_changes41[k4,1] = Vel4[k4]*delta_t*sin(Heading4[i+1]-theta_init)
			robot_position_changes41[k4,2] = 0
			robot_position41[k4,0] = robot_position_changes41[k4,0]+robot_position41[k4-1,0]
			robot_position41[k4,1] = robot_position_changes41[k4,1]+robot_position41[k4-1,1]
			robot_position41[k4,2] = Heading4[i]

			#Plotting movement for planned path
			plt.scatter(robot_position41[k4,0],robot_position41[k4,1],label='Path Driven #1')
			plt.draw()
			plt.pause(0.000001)

			k4 = k4 + 1
	i = i+1


#Set #2 (5)
i = 0
while i<len(Movement5)-1:
	#Calling the Rotational Velocity function with inputs of x,y,theta,x_target,y_target and outputting array of rotational velocities to accomplish this
	Rot_Vel5, rotation_traveled5 = Rotational_Velocity(Movement5[i,0],Movement5[i,1],Movement5[i+1,0],Movement5[i+1,1],Heading5[i],theta_init,dw_max,delta_t)
	Heading5[i+1] = atan2((Movement5[i+1,1]-Movement5[i,1]),(Movement5[i+1,0]-Movement5[i,0])) + theta_init #updating current heading from theta=0

	#Setting up two arrays to hold robot heading and position for each movement
	robot_heading5 = np.zeros((len(Rot_Vel5),1))
	robot_position5 = np.zeros((len(Rot_Vel5),3))

	#Calculating and storing heading of robot based on array of rotational velocities
	j5 = 0
	while j5<len(Rot_Vel5):
		if j5==0: #first element of array when w=0
			robot_heading5[j5,0] = Rot_Vel5[j5]*delta_t

			robot_position5[j5,0] = Movement5[i,0]
			robot_position5[j5,1] = Movement5[i,1]
			robot_position5[j5,2] = robot_heading5[j5,0]

			j5 = j5+1
		else: #all others
			robot_heading5[j5,0] = Rot_Vel5[j5]*delta_t + robot_heading5[j5-1,0]

			robot_position5[j5,0] = Movement5[i,0]
			robot_position5[j5,1] = Movement5[i,1]
			robot_position5[j5,2] = robot_position5[j5-1,2] + robot_heading5[j5,0]

			j5 = j5+1

	#Calling the Velocity function with inputs of x,y,x_target,y_target and outputting array of velocities to accomplish this
	Vel5 = Velocity(Movement5[i,0],Movement5[i,1],Movement5[i+1,0],Movement5[i+1,1],dv_max,delta_t)

	#Setting up two arrays to hold robot position changes and overall position for each movement
	robot_position_changes51 = np.zeros((len(Vel5),3))
	robot_position51 = np.zeros((len(Vel5),3))

	#Calculating and storing pose of robot based on array of velocities
	k5 = 0
	while k5<len(Vel5):
		if k5==0: #first element of array when v=0
			robot_position_changes51[k5,0] = Vel5[k5]*delta_t*cos(Heading5[i])
			robot_position_changes51[k5,1] = Vel5[k5]*delta_t*sin(Heading5[i])
			robot_position_changes51[k5,2] = 0
			robot_position51[k5,0] = robot_position_changes51[k5,0] + Movement5[i,0] #accounting for initial position
			robot_position51[k5,1] = robot_position_changes51[k5,1] + Movement5[i,1]
			robot_position51[k5,2] = robot_heading5[j5-1,0]

			#Plotting movement for planned path
			plt.scatter(robot_position51[k5,0],robot_position51[k5,1],label='Path Driven #2')
			plt.draw()
			plt.pause(0.000001)

			k5 = k5+1
		else: #all others
			robot_position_changes51[k5,0] = Vel5[k5]*delta_t*cos(Heading5[i+1]-theta_init)
			robot_position_changes51[k5,1] = Vel5[k5]*delta_t*sin(Heading5[i+1]-theta_init)
			robot_position_changes51[k5,2] = 0
			robot_position51[k5,0] = robot_position_changes51[k5,0]+robot_position51[k5-1,0]
			robot_position51[k5,1] = robot_position_changes51[k5,1]+robot_position51[k5-1,1]
			robot_position51[k5,2] = Heading5[i]

			#Plotting movement for planned path
			plt.scatter(robot_position51[k5,0],robot_position51[k5,1],label='Path Driven #2')
			plt.draw()
			plt.pause(0.000001)

			k5 = k5 + 1
	i = i+1


#Set #3 (6)
i = 0
while i<len(Movement6)-1:
	#Calling the Rotational Velocity function with inputs of x,y,theta,x_target,y_target and outputting array of rotational velocities to accomplish this
	Rot_Vel6, rotation_traveled6 = Rotational_Velocity(Movement6[i,0],Movement6[i,1],Movement6[i+1,0],Movement6[i+1,1],Heading6[i],theta_init,dw_max,delta_t)
	Heading6[i+1] = atan2((Movement6[i+1,1]-Movement6[i,1]),(Movement6[i+1,0]-Movement6[i,0])) + theta_init #updating current heading from theta=0

	#Setting up two arrays to hold robot heading and position for each movement
	robot_heading6 = np.zeros((len(Rot_Vel6),1))
	robot_position6 = np.zeros((len(Rot_Vel6),3))

	#Calculating and storing heading of robot based on array of rotational velocities
	j6 = 0
	while j6<len(Rot_Vel6):
		if j6==0: #first element of array when w=0
			robot_heading6[j6,0] = Rot_Vel6[j6]*delta_t

			robot_position6[j6,0] = Movement6[i,0]
			robot_position6[j6,1] = Movement6[i,1]
			robot_position6[j6,2] = robot_heading6[j6,0]

			j6 = j6+1
		else: #all others
			robot_heading6[j6,0] = Rot_Vel6[j6]*delta_t + robot_heading6[j6-1,0]

			robot_position6[j6,0] = Movement6[i,0]
			robot_position6[j6,1] = Movement6[i,1]
			robot_position6[j6,2] = robot_position6[j6-1,2] + robot_heading5[j6,0]

			j6 = j6+1

	#Calling the Velocity function with inputs of x,y,x_target,y_target and outputting array of velocities to accomplish this
	Vel6 = Velocity(Movement6[i,0],Movement6[i,1],Movement6[i+1,0],Movement6[i+1,1],dv_max,delta_t)

	#Setting up two arrays to hold robot position changes and overall position for each movement
	robot_position_changes61 = np.zeros((len(Vel6),3))
	robot_position61 = np.zeros((len(Vel6),3))

	#Calculating and storing pose of robot based on array of velocities
	k6 = 0
	while k6<len(Vel6):
		if k6==0: #first element of array when v=0
			robot_position_changes61[k6,0] = Vel6[k6]*delta_t*cos(Heading6[i])
			robot_position_changes61[k6,1] = Vel6[k6]*delta_t*sin(Heading6[i])
			robot_position_changes61[k6,2] = 0
			robot_position61[k6,0] = robot_position_changes61[k6,0] + Movement6[i,0] #accounting for initial position
			robot_position61[k6,1] = robot_position_changes61[k6,1] + Movement6[i,1]
			robot_position61[k6,2] = robot_heading6[j6-1,0]

			#Plotting movement for planned path
			plt.scatter(robot_position61[k6,0],robot_position61[k6,1],label='Path Driven #2')
			plt.draw()
			plt.pause(0.000001)

			k6 = k6+1
		else: #all others
			robot_position_changes61[k6,0] = Vel6[k6]*delta_t*cos(Heading6[i+1]-theta_init)
			robot_position_changes61[k6,1] = Vel6[k6]*delta_t*sin(Heading6[i+1]-theta_init)
			robot_position_changes61[k6,2] = 0
			robot_position61[k6,0] = robot_position_changes61[k6,0]+robot_position61[k6-1,0]
			robot_position61[k6,1] = robot_position_changes61[k6,1]+robot_position61[k6-1,1]
			robot_position61[k6,2] = Heading6[i]

			#Plotting movement for planned path
			plt.scatter(robot_position61[k6,0],robot_position61[k6,1],label='Path Driven #2')
			plt.draw()
			plt.pause(0.000001)

			k6 = k6 + 1
	i = i+1









#Part #8 - Plan the paths while driving them.
S4 = [2.45,-3.55] #start position
G4 = [0.95,-1.55] #goal position
C4 = S4 #current node
g4 = 0 #path cost from start -> node
h4 = sqrt((S4[0]-G4[0])**2+(S4[1]-G4[1])**2) #estimate on cheapest path from node -> goal using Manhattan distance
Movement4 = C4 #tracking movement

S5 = [4.95,-0.05] #start position
G5 = [2.45,0.25] #goal position
C5 = S5 #current node
g5 = 0 #path cost from start -> node
h5 = sqrt((S5[0]-G5[0])**2+(S5[1]-G5[1])**2) #estimate on cheapest path from node -> goal using Manhattan distance
Movement5 = C5 #tracking movement

S6 = [-0.55,1.45] #start position
G6 = [1.95,3.95] #goal position
C6 = S6 #current node
g6 = 0 #path cost from start -> node
h6 = sqrt((S6[0]-G6[0])**2+(S6[1]-G6[1])**2) #estimate on cheapest path from node -> goal using Manhattan distance
Movement6 = C6 #tracking movement


dv_max = 0.288
dw_max = 5.579
delta_t = 0.1
theta_init = -pi/2


Heading4 = [theta_init]
Heading5 = [theta_init]
Heading6 = [theta_init]


#Figure #5: Small Grid Cells - Planning while Driving
fig5 = plt.figure(5)
#Setting up axis ranges and making it square
ax = fig5.gca()
ax.set_xticks(np.arange(-2,6,1))
ax.set_yticks(np.arange(-6,7,0.5))
plt.axis('equal')
#Plotting scatter points for each landmark position
plt.scatter(Landmark_x, Landmark_y, s=10)
#Plotting rectangles in each grid cell where a landmark is occupied
ax1 = plt.gca()
x=-2
while x<=5:
	y=-6
	while y<=6:
		i = 0
		while i<len(landmark):
			if x<Landmark_pos[i,0] and Landmark_pos[i,0]<(x+0.3) and y<Landmark_pos[i,1] and Landmark_pos[i,1]<(y+0.3):
				ax1.add_patch(Rectangle((Landmark_cen[i,0]-(0.3),Landmark_cen[i,1]-(0.3)),0.6,0.6,facecolor="red",alpha=0.7))
			i = i+1
		y=y+grid_size
	x=x+grid_size
#Labeling graph and setting axis limits
plt.ion()
plt.title('Small Grid Cells - Driving Planned Paths')
plt.xlabel('X-position [m]')
plt.ylabel('Y-position [m]')
plt.legend()
plt.xlim([-2,5])
plt.ylim([-6,6])
plt.grid()
plt.show()



#Path planning for set #1 (4)
i = 0
while h4>1.01:
	#Calculating heurisitic function f=g+h for set #1
	C4, g4, h4 = Movement(C4[0],C4[1],G4[0],G4[1],g4,grid_size,landmark_space)

	#Tracking and plotting movement for set #1
	Movement4 = np.vstack((Movement4,C4));

	Movement4_x = Movement4[:,0]
	Movement4_y = Movement4[:,1]

	plt.plot(Movement4_x,Movement4_y,label='Set 1')
	plt.draw()
	plt.pause(0.000001)


	#Calling the Rotational Velocity function with inputs of x,y,theta,x_target,y_target and outputting array of rotational velocities to accomplish this
	Rot_Vel4, rotation_traveled4 = Rotational_Velocity(Movement4[i,0],Movement4[i,1],Movement4[i+1,0],Movement4[i+1,1],Heading4[i],theta_init,dw_max,delta_t)
	Heading4 = np.vstack((Heading4,atan2((Movement4[i+1,1]-Movement4[i,1]),(Movement4[i+1,0]-Movement4[i,0])) + theta_init)) #updating current heading from theta=0

	#Setting up two arrays to hold robot heading and position for each movement
	robot_heading4 = np.zeros((len(Rot_Vel4),1))
	robot_position4 = np.zeros((len(Rot_Vel4),3))

	#Calculating and storing heading of robot based on array of rotational velocities
	j4 = 0
	while j4<len(Rot_Vel4):
		if j4==0: #first element of array when w=0
			robot_heading4[j4,0] = Rot_Vel4[j4]*delta_t

			robot_position4[j4,0] = Movement4[i,0]
			robot_position4[j4,1] = Movement4[i,1]
			robot_position4[j4,2] = robot_heading4[j4,0]

			j4 = j4+1
		else: #all others
			robot_heading4[j4,0] = Rot_Vel4[j4]*delta_t + robot_heading4[j4-1,0]

			robot_position4[j4,0] = Movement4[i,0]
			robot_position4[j4,1] = Movement4[i,1]
			robot_position4[j4,2] = robot_position4[j4-1,2] + robot_heading4[j4,0]

			j4 = j4+1


	#Calling the Velocity function with inputs of x,y,x_target,y_target and outputting array of velocities to accomplish this
	Vel4 = Velocity(Movement4[i,0],Movement4[i,1],Movement4[i+1,0],Movement4[i+1,1],dv_max,delta_t)

	#Setting up two arrays to hold robot position changes and overall position for each movement
	robot_position_changes41 = np.zeros((len(Vel4),3))
	robot_position41 = np.zeros((len(Vel4),3))

	#Calculating and storing pose of robot based on array of velocities
	k4 = 0
	while k4<len(Vel4):
		if k4==0: #first element of array when v=0
			robot_position_changes41[k4,0] = Vel4[k4]*delta_t*cos(Heading4[i])
			robot_position_changes41[k4,1] = Vel4[k4]*delta_t*sin(Heading4[i])
			robot_position_changes41[k4,2] = 0
			robot_position41[k4,0] = robot_position_changes41[k4,0] + Movement4[i,0] #accounting for initial position
			robot_position41[k4,1] = robot_position_changes41[k4,1] + Movement4[i,1]
			robot_position41[k4,2] = robot_heading4[j4-1,0]

			#Plotting movement for planned path
			plt.scatter(robot_position41[k4,0],robot_position41[k4,1],label='Path Driven #1')
			plt.draw()
			plt.pause(0.000001)

			k4 = k4+1
		else: #all others
			robot_position_changes41[k4,0] = Vel4[k4]*delta_t*cos(Heading4[i+1]-theta_init)
			robot_position_changes41[k4,1] = Vel4[k4]*delta_t*sin(Heading4[i+1]-theta_init)
			robot_position_changes41[k4,2] = 0
			robot_position41[k4,0] = robot_position_changes41[k4,0]+robot_position41[k4-1,0]
			robot_position41[k4,1] = robot_position_changes41[k4,1]+robot_position41[k4-1,1]
			robot_position41[k4,2] = Heading4[i]

			#Plotting movement for planned path
			plt.scatter(robot_position41[k4,0],robot_position41[k4,1],label='Path Driven #1')
			plt.draw()
			plt.pause(0.000001)

			k4 = k4+1

	i = i+1


#Path planning for set #2 (5)
i=0
while h5>1.01:
	#Calculating heurisitic function f=g+h for set 1
	C5, g5, h5 = Movement(C5[0],C5[1],G5[0],G5[1],g5,grid_size,landmark_space)

	if i>1:
		#When stuck behind obstacle, clear obstacle, then continue with online A*
		if fabs(Movement5[i,0] - Movement5[i-2,0]) < 0.001 and fabs(Movement5[i,1] - Movement5[i-2,1]) < 0.001:
			#Calculating heuristic value for each neighboring cell before moved into oscillatory state
			f_up, h_up = Heuristic(Movement5[i-1,0],Movement5[i-1,1]+grid_size,G5[0],G5[1],g5,landmark_space,grid_size) #up
			f_down, h_down = Heuristic(Movement5[i-1,0],Movement5[i-1,1]-grid_size,G5[0],G5[1],g5,landmark_space,grid_size) #down
			f_left, h_left = Heuristic(Movement5[i-1,0]-grid_size,Movement5[i-1,1],G5[0],G5[1],g5,landmark_space,grid_size) #left
			f_right, h_right = Heuristic(Movement5[i-1,0]+grid_size,Movement5[i-1,1],G5[0],G5[1],g5,landmark_space,grid_size) #right
			f_up_left, h_up_left = Heuristic(Movement5[i-1,0]-grid_size,Movement5[i-1,1]+grid_size,G5[0],G5[1],g5,landmark_space,grid_size) #up-left
			f_up_right, h_up_right = Heuristic(Movement5[i-1,0]+grid_size,Movement5[i-1,1]+grid_size,G5[0],G5[1],g5,landmark_space,grid_size) #up-right
			f_down_left, h_down_left = Heuristic(Movement5[i-1,0]-grid_size,Movement5[i-1,1]-grid_size,G5[0],G5[1],g5,landmark_space,grid_size) #down-left
			f_down_right, h_down_right = Heuristic(Movement5[i-1,0]+grid_size,Movement5[i-1,1]-grid_size,G5[0],G5[1],g5,landmark_space,grid_size) #down-right

			if f_up>1000: obstacle="up"; f_obs=f_up; h_obs=h_up;
			if f_down>1000: obstacle="down"; f_obs=f_down; h_obs=h_down;
			if f_left>1000: obstacle="left"; f_obs=f_left; h_obs=h_left;
			if f_right>1000: obstacle="right"; f_obs=f_right; h_obs=h_right;

			j = i+1

			#Moving robot out of the way of the obstacle
			while j<i+5:
				C5 = [Movement5[j-1,0]+(Movement5[j-1,0]-Movement5[j-3,0]),Movement5[j-1,1]+(Movement5[j-1,1]-Movement5[j-3,1])]; 
				g5 = f_obs; 
				h5 = h_obs;

				f_up, h_up = Heuristic(Movement5[j-1,0],Movement5[j-1,1]+grid_size,G5[0],G5[1],g5-1000,landmark_space,grid_size) #up
				f_down, h_down = Heuristic(Movement5[j-1,0],Movement5[j-1,1]-grid_size,G5[0],G5[1],g5-1000,landmark_space,grid_size) #down
				f_left, h_left = Heuristic(Movement5[j-1,0]-grid_size,Movement5[j-1,1],G5[0],G5[1],g5-1000,landmark_space,grid_size) #left
				f_right, h_right = Heuristic(Movement5[j-1,0]+grid_size,Movement5[j-1,1],G5[0],G5[1],g5-1000,landmark_space,grid_size) #right
				f_up_left, h_up_left = Heuristic(Movement5[j-1,0]-grid_size,Movement5[j-1,1]+grid_size,G5[0],G5[1],g5-1000,landmark_space,grid_size) #up-left
				f_up_right, h_up_right = Heuristic(Movement5[j-1,0]+grid_size,Movement5[j-1,1]+grid_size,G5[0],G5[1],g5-1000,landmark_space,grid_size) #up-right
				f_down_left, h_down_left = Heuristic(Movement5[j-1,0]-grid_size,Movement5[j-1,1]-grid_size,G5[0],G5[1],g5-1000,landmark_space,grid_size) #down-left
				f_down_right, h_down_right = Heuristic(Movement5[j-1,0]+grid_size,Movement5[j-1,1]-grid_size,G5[0],G5[1],g5-1000,landmark_space,grid_size) #down-right

				if f_up>1000: obstacle="up"; f_obs=f_up; h_obs=h_up;
				elif f_down>1000: obstacle="down"; f_obs=f_down; h_obs=h_down;
				elif f_left>1000: obstacle="left"; f_obs=f_left; h_obs=h_left;
				elif f_right>1000: obstacle="right"; f_obs=f_right; h_obs=h_right;

				Movement5 = np.vstack((Movement5,C5));

				Movement5_x = Movement5[:,0]
				Movement5_y = Movement5[:,1]

				plt.plot(Movement5_x,Movement5_y,label='Set 1')
				plt.draw()
				plt.pause(0.000001)


				#Calling the Rotational Velocity function with inputs of x,y,theta,x_target,y_target and outputting array of rotational velocities to accomplish this
				Rot_Vel5, rotation_traveled5 = Rotational_Velocity(Movement5[j-1,0],Movement5[j-1,1],Movement5[j-1+1,0],Movement5[j-1+1,1],Heading5[i],theta_init,dw_max,delta_t)
				Heading5 = np.vstack((Heading5,atan2((Movement5[j-1+1,1]-Movement5[j-1,1]),(Movement5[j-1+1,0]-Movement5[j-1,0])) + theta_init)) #updating current heading from theta=0

				#Setting up two arrays to hold robot heading and position for each movement
				robot_heading5 = np.zeros((len(Rot_Vel5),1))
				robot_position5 = np.zeros((len(Rot_Vel5),3))

				#Calculating and storing heading of robot based on array of rotational velocities
				j5 = 0
				while j5<len(Rot_Vel5):
					if j5==0: #first element of array when w=0
						robot_heading5[j5,0] = Rot_Vel5[j5]*delta_t

						robot_position5[j5,0] = Movement5[j-1,0]
						robot_position5[j5,1] = Movement5[j-1,1]
						robot_position5[j5,2] = robot_heading5[j5,0]

						j5 = j5+1
					else: #all others
						robot_heading5[j5,0] = Rot_Vel5[j5]*delta_t + robot_heading5[j5-1,0]

						robot_position5[j5,0] = Movement5[j-1,0]
						robot_position5[j5,1] = Movement5[j-1,1]
						robot_position5[j5,2] = robot_position5[j5-1,2] + robot_heading5[j5,0]

						j5 = j5+1
	
				#Calling the Velocity function with inputs of x,y,x_target,y_target and outputting array of velocities to accomplish this
				Vel5 = Velocity(Movement5[j-1,0],Movement5[j-1,1],Movement5[j-1+1,0],Movement5[j-1+1,1],dv_max,delta_t)

				#Setting up two arrays to hold robot position changes and overall position for each movement
				robot_position_changes51 = np.zeros((len(Vel5),3))
				robot_position51 = np.zeros((len(Vel5),3))

				#Calculating and storing pose of robot based on array of velocities
				k5 = 0
				while k5<len(Vel5):
					if k5==0: #first element of array when v=0
						robot_position_changes51[k5,0] = Vel5[k5]*delta_t*cos(Heading5[i])
						robot_position_changes51[k5,1] = Vel5[k5]*delta_t*sin(Heading5[i])
						robot_position_changes51[k5,2] = 0
						robot_position51[k5,0] = robot_position_changes51[k5,0] + Movement5[j-1,0] #accounting for initial position
						robot_position51[k5,1] = robot_position_changes51[k5,1] + Movement5[j-1,1]
						robot_position51[k5,2] = robot_heading5[j5-1,0]

						#Plotting movement for planned path
						plt.scatter(robot_position51[k5,0],robot_position51[k5,1],label='Path Driven #2')
						plt.draw()
						plt.pause(0.000001)

						k5 = k5+1
					else: #all others
						robot_position_changes51[k5,0] = Vel5[k5]*delta_t*cos(Heading5[j-1+1]-theta_init)
						robot_position_changes51[k5,1] = Vel5[k5]*delta_t*sin(Heading5[j-1+1]-theta_init)
						robot_position_changes51[k5,2] = 0
						robot_position51[k5,0] = robot_position_changes51[k5,0]+robot_position51[k5-1,0]
						robot_position51[k5,1] = robot_position_changes51[k5,1]+robot_position51[k5-1,1]
						robot_position51[k5,2] = Heading5[i]

						#Plotting movement for planned path
						plt.scatter(robot_position51[k5,0],robot_position51[k5,1],label='Path Driven #2')
						plt.draw()
						plt.pause(0.000001)

						k5 = k5+1

				j = j+1

			i = j-1

	#Tracking and plotting movement for set #2
	Movement5 = np.vstack((Movement5,C5));

	Movement5_x = Movement5[:,0]
	Movement5_y = Movement5[:,1]

	plt.plot(Movement5_x,Movement5_y,label='Set 1')
	plt.draw()
	plt.pause(0.000001)


	#Calling the Rotational Velocity function with inputs of x,y,theta,x_target,y_target and outputting array of rotational velocities to accomplish this
	Rot_Vel5, rotation_traveled5 = Rotational_Velocity(Movement5[i,0],Movement5[i,1],Movement5[i+1,0],Movement5[i+1,1],Heading5[i],theta_init,dw_max,delta_t)
	Heading5 = np.vstack((Heading5,atan2((Movement5[i+1,1]-Movement5[i,1]),(Movement5[i+1,0]-Movement5[i,0])) + theta_init)) #updating current heading from theta=0

	#Setting up two arrays to hold robot heading and position for each movement
	robot_heading5 = np.zeros((len(Rot_Vel5),1))
	robot_position5 = np.zeros((len(Rot_Vel5),3))

	#Calculating and storing heading of robot based on array of rotational velocities
	j5 = 0
	while j5<len(Rot_Vel5):
		if j5==0: #first element of array when w=0
			robot_heading5[j5,0] = Rot_Vel5[j5]*delta_t

			robot_position5[j5,0] = Movement5[i,0]
			robot_position5[j5,1] = Movement5[i,1]
			robot_position5[j5,2] = robot_heading5[j5,0]

			j5 = j5+1
		else: #all others
			robot_heading5[j5,0] = Rot_Vel5[j5]*delta_t + robot_heading5[j5-1,0]

			robot_position5[j5,0] = Movement5[i,0]
			robot_position5[j5,1] = Movement5[i,1]
			robot_position5[j5,2] = robot_position5[j5-1,2] + robot_heading5[j5,0]

			j5 = j5+1

	#Calling the Velocity function with inputs of x,y,x_target,y_target and outputting array of velocities to accomplish this
	Vel5 = Velocity(Movement5[i,0],Movement5[i,1],Movement5[i+1,0],Movement5[i+1,1],dv_max,delta_t)

	#Setting up two arrays to hold robot position changes and overall position for each movement
	robot_position_changes51 = np.zeros((len(Vel5),3))
	robot_position51 = np.zeros((len(Vel5),3))

	#Calculating and storing pose of robot based on array of velocities
	k5 = 0
	while k5<len(Vel5):
		if k5==0: #first element of array when v=0
			robot_position_changes51[k5,0] = Vel5[k5]*delta_t*cos(Heading5[i])
			robot_position_changes51[k5,1] = Vel5[k5]*delta_t*sin(Heading5[i])
			robot_position_changes51[k5,2] = 0
			robot_position51[k5,0] = robot_position_changes51[k5,0] + Movement5[i,0] #accounting for initial position
			robot_position51[k5,1] = robot_position_changes51[k5,1] + Movement5[i,1]
			robot_position51[k5,2] = robot_heading5[j5-1,0]

			#Plotting movement for planned path
			plt.scatter(robot_position51[k5,0],robot_position51[k5,1],label='Path Driven #2')
			plt.draw()
			plt.pause(0.000001)

			k5 = k5+1
		else: #all others
			robot_position_changes51[k5,0] = Vel5[k5]*delta_t*cos(Heading5[i+1]-theta_init)
			robot_position_changes51[k5,1] = Vel5[k5]*delta_t*sin(Heading5[i+1]-theta_init)
			robot_position_changes51[k5,2] = 0
			robot_position51[k5,0] = robot_position_changes51[k5,0]+robot_position51[k5-1,0]
			robot_position51[k5,1] = robot_position_changes51[k5,1]+robot_position51[k5-1,1]
			robot_position51[k5,2] = Heading5[i]

			#Plotting movement for planned path
			plt.scatter(robot_position51[k5,0],robot_position51[k5,1],label='Path Driven #2')
			plt.draw()
			plt.pause(0.000001)

			k5 = k5+1

	i = i+1



#Path planning for set #3 (6)
i = 0
while h6>1.01:
	#Calculating heurisitic function f=g+h for set 6
	C6, g6, h6 = Movement(C6[0],C6[1],G6[0],G6[1],g6,grid_size,landmark_space)

	#Tracking and plotting movement for set #3
	Movement6 = np.vstack((Movement6,C6));

	Movement6_x = Movement6[:,0]
	Movement6_y = Movement6[:,1]

	plt.plot(Movement6_x,Movement6_y,label='Set 1')
	plt.draw()
	plt.pause(0.000001)


	#Calling the Rotational Velocity function with inputs of x,y,theta,x_target,y_target and outputting array of rotational velocities to accomplish this
	Rot_Vel6, rotation_traveled6 = Rotational_Velocity(Movement6[i,0],Movement6[i,1],Movement6[i+1,0],Movement6[i+1,1],Heading6[i],theta_init,dw_max,delta_t)
	Heading6 = np.vstack((Heading6,atan2((Movement6[i+1,1]-Movement6[i,1]),(Movement6[i+1,0]-Movement6[i,0])) + theta_init)) #updating current heading from theta=0

	#Setting up two arrays to hold robot heading and position for each movement
	robot_heading6 = np.zeros((len(Rot_Vel6),1))
	robot_position6 = np.zeros((len(Rot_Vel6),3))

	#Calculating and storing heading of robot based on array of rotational velocities
	j6 = 0
	while j6<len(Rot_Vel6):
		if j6==0: #first element of array when w=0
			robot_heading6[j6,0] = Rot_Vel6[j6]*delta_t

			robot_position6[j6,0] = Movement6[i,0]
			robot_position6[j6,1] = Movement6[i,1]
			robot_position6[j6,2] = robot_heading6[j6,0]

			j6 = j6+1
		else: #all others
			robot_heading6[j6,0] = Rot_Vel6[j6]*delta_t + robot_heading6[j6-1,0]

			robot_position6[j6,0] = Movement6[i,0]
			robot_position6[j6,1] = Movement6[i,1]
			robot_position6[j6,2] = robot_position6[j6-1,2] + robot_heading6[j6,0]

			j6 = j6+1

	#Calling the Velocity function with inputs of x,y,x_target,y_target and outputting array of velocities to accomplish this
	Vel6 = Velocity(Movement6[i,0],Movement6[i,1],Movement6[i+1,0],Movement6[i+1,1],dv_max,delta_t)

	#Setting up two arrays to hold robot position changes and overall position for each movement
	robot_position_changes61 = np.zeros((len(Vel6),3))
	robot_position61 = np.zeros((len(Vel6),3))

	#Calculating and storing pose of robot based on array of velocities
	k6 = 0
	while k6<len(Vel6):
		if k6==0: #first element of array when v=0
			robot_position_changes61[k6,0] = Vel6[k6]*delta_t*cos(Heading6[i])
			robot_position_changes61[k6,1] = Vel6[k6]*delta_t*sin(Heading6[i])
			robot_position_changes61[k6,2] = 0
			robot_position61[k6,0] = robot_position_changes61[k6,0] + Movement6[i,0] #accounting for initial position
			robot_position61[k6,1] = robot_position_changes61[k6,1] + Movement6[i,1]
			robot_position61[k6,2] = robot_heading6[j6-1,0]

			#Plotting movement for planned path
			plt.scatter(robot_position61[k6,0],robot_position61[k6,1],label='Path Driven #2')
			plt.draw()
			plt.pause(0.000001)

			k6 = k6+1
		else: #all others
			robot_position_changes61[k6,0] = Vel6[k6]*delta_t*cos(Heading6[i+1]-theta_init)
			robot_position_changes61[k6,1] = Vel6[k6]*delta_t*sin(Heading6[i+1]-theta_init)
			robot_position_changes61[k6,2] = 0
			robot_position61[k6,0] = robot_position_changes61[k6,0]+robot_position61[k6-1,0]
			robot_position61[k6,1] = robot_position_changes61[k6,1]+robot_position61[k6-1,1]
			robot_position61[k6,2] = Heading6[i]

			#Plotting movement for planned path
			plt.scatter(robot_position61[k6,0],robot_position61[k6,1],label='Path Driven #2')
			plt.draw()
			plt.pause(0.000001)

			k6 = k6+1

	i = i+1








#Part #9 - Drive while planning for the start/goal positions of Step #3 - large grid cells. Provide a visual display of results.

#Setting grid size and associated values back to 1x1 m.
x=-2
while x<=5:
	y=-6
	while y<=6:
		i = 0
		while i<len(landmark):
			if x<Landmark_pos[i,0] and Landmark_pos[i,0]<(x+1) and y<Landmark_pos[i,1] and Landmark_pos[i,1]<(y+1):
				Landmark_cen[i,0] = x+0.5
				Landmark_cen[i,1] = y+0.5
			i = i+1
		y=y+1
	x=x+1

Landmark_x = Landmark_pos[:,0]
Landmark_y = Landmark_pos[:,1]

grid_size = 1
landmark_space = grid_size/2.


S1 = [0.5,-1.5] #start position
G1 = [0.5,1.5] #goal position
C1 = S1 #current node
g1 = 0 #path cost from start -> node
h1 = sqrt((S1[0]-G1[0])**2+(S1[1]-G1[1])**2) #estimate on cheapest path from node -> goal using Manhattan distance
Movement1 = C1 #tracking movement

S2 = [4.5,3.5] #start position
G2 = [4.5,-1.5] #goal position
C2 = S2 #current node
g2 = 0 #path cost from start -> node
h2 = sqrt((S2[0]-G2[0])**2+(S2[1]-G2[1])**2) #estimate on cheapest path from node -> goal using Manhattan distance
Movement2 = C2 #tracking movement

S3 = [-0.5,5.5] #start position
G3 = [1.5,-3.5] #goal position
C3 = S3 #current node
g3 = 0 #path cost from start -> node
h3 = sqrt((S3[0]-G3[0])**2+(S3[1]-G3[1])**2) #estimate on cheapest path from node -> goal using Manhattan distance
Movement3 = C3 #tracking movement


dv_max = 0.288
dw_max = 5.579
delta_t = 0.1
theta_init = -pi/2


Heading1 = [theta_init]
Heading2 = [theta_init]
Heading3 = [theta_init]


#Figure #6: Large Grid Cells - Planning while Driving
fig6 = plt.figure(6)
#Setting up axis ranges and making it square
ax = fig6.gca()
ax.set_xticks(np.arange(-2,6,1))
ax.set_yticks(np.arange(-6,7,1))
plt.axis('equal')
#Plotting scatter points for each landmark position
plt.scatter(Landmark_x, Landmark_y, s=10)
#Plotting rectangles in each grid cell where a landmark is occupied
ax1 = plt.gca()
x=-2
while x<=5:
	y=-6
	while y<=6:
		i = 0
		while i<len(landmark):
			if x<Landmark_pos[i,0] and Landmark_pos[i,0]<(x+grid_size) and y<Landmark_pos[i,1] and Landmark_pos[i,1]<(y+grid_size):
				ax1.add_patch(Rectangle((Landmark_cen[i,0]-(grid_size/2.),Landmark_cen[i,1]-(grid_size/2.)),1,1,facecolor="red",alpha=0.7))
			i = i+1
		y=y+grid_size
	x=x+grid_size
#Labeling graph and setting axis limits
plt.ion()
plt.title('Small Grid Cells - Driving Planned Paths')
plt.xlabel('X-position [m]')
plt.ylabel('Y-position [m]')
plt.legend()
plt.xlim([-2,5])
plt.ylim([-6,6])
plt.grid()
plt.show()



#Path planning for set #1
i = 0
while h1>1:
	#Calculating heurisitic function f=g+h for set #1
	C1, g1, h1 = Movement(C1[0],C1[1],G1[0],G1[1],g1,grid_size,landmark_space)

	#Tracking and plotting movement for set #1
	Movement1 = np.vstack((Movement1,C1));

	Movement1_x = Movement1[:,0]
	Movement1_y = Movement1[:,1]

	plt.plot(Movement1_x,Movement1_y,label='Set 1')
	plt.draw()
	plt.pause(0.000001)


	#Calling the Rotational Velocity function with inputs of x,y,theta,x_target,y_target and outputting array of rotational velocities to accomplish this
	Rot_Vel1, rotation_traveled1 = Rotational_Velocity(Movement1[i,0],Movement1[i,1],Movement1[i+1,0],Movement1[i+1,1],Heading1[i],theta_init,dw_max,delta_t)
	Heading1 = np.vstack((Heading1,atan2((Movement1[i+1,1]-Movement1[i,1]),(Movement1[i+1,0]-Movement1[i,0])) + theta_init)) #updating current heading from theta=0

	#Setting up two arrays to hold robot heading and position for each movement
	robot_heading1 = np.zeros((len(Rot_Vel1),1))
	robot_position1 = np.zeros((len(Rot_Vel1),3))

	#Calculating and storing heading of robot based on array of rotational velocities
	j1 = 0
	while j1<len(Rot_Vel1):
		if j1==0: #first element of array when w=0
			robot_heading1[j1,0] = Rot_Vel1[j1]*delta_t

			robot_position1[j1,0] = Movement1[i,0]
			robot_position1[j1,1] = Movement1[i,1]
			robot_position1[j1,2] = robot_heading1[j1,0]

			j1 = j1+1
		else: #all others
			robot_heading1[j1,0] = Rot_Vel1[j1]*delta_t + robot_heading1[j1-1,0]

			robot_position1[j1,0] = Movement1[i,0]
			robot_position1[j1,1] = Movement1[i,1]
			robot_position1[j1,2] = robot_position1[j1-1,2] + robot_heading1[j1,0]

			j1 = j1+1


	#Calling the Velocity function with inputs of x,y,x_target,y_target and outputting array of velocities to accomplish this
	Vel1 = Velocity(Movement1[i,0],Movement1[i,1],Movement1[i+1,0],Movement1[i+1,1],dv_max,delta_t)

	#Setting up two arrays to hold robot position changes and overall position for each movement
	robot_position_changes11 = np.zeros((len(Vel1),3))
	robot_position11 = np.zeros((len(Vel1),3))

	#Calculating and storing pose of robot based on array of velocities
	k1 = 0
	while k1<len(Vel1):
		if k1==0: #first element of array when v=0
			robot_position_changes11[k1,0] = Vel1[k1]*delta_t*cos(Heading1[i])
			robot_position_changes11[k1,1] = Vel1[k1]*delta_t*sin(Heading1[i])
			robot_position_changes11[k1,2] = 0
			robot_position11[k1,0] = robot_position_changes11[k1,0] + Movement1[i,0] #accounting for initial position
			robot_position11[k1,1] = robot_position_changes11[k1,1] + Movement1[i,1]
			robot_position11[k1,2] = robot_heading1[j1-1,0]

			#Plotting movement for planned path
			plt.scatter(robot_position11[k1,0],robot_position11[k1,1],label='Path Driven #1')
			plt.draw()
			plt.pause(0.000001)

			k1 = k1+1
		else: #all others
			robot_position_changes11[k1,0] = Vel1[k1]*delta_t*cos(Heading1[i+1]-theta_init)
			robot_position_changes11[k1,1] = Vel1[k1]*delta_t*sin(Heading1[i+1]-theta_init)
			robot_position_changes11[k1,2] = 0
			robot_position11[k1,0] = robot_position_changes11[k1,0]+robot_position11[k1-1,0]
			robot_position11[k1,1] = robot_position_changes11[k1,1]+robot_position11[k1-1,1]
			robot_position11[k1,2] = Heading1[i]

			#Plotting movement for planned path
			plt.scatter(robot_position11[k1,0],robot_position11[k1,1],label='Path Driven #1')
			plt.draw()
			plt.pause(0.000001)

			k1 = k1+1

	i = i+1



#Path planning for set #2
i = 0
while h2>1:
	#Calculating heurisitic function f=g+h for set #2
	C2, g2, h2 = Movement(C2[0],C2[1],G2[0],G2[1],g2,grid_size,landmark_space)

	#Tracking and plotting movement for set #2
	Movement2 = np.vstack((Movement2,C2));

	Movement2_x = Movement2[:,0]
	Movement2_y = Movement2[:,1]

	plt.plot(Movement2_x,Movement2_y,label='Set 2')
	plt.draw()
	plt.pause(0.000001)


	#Calling the Rotational Velocity function with inputs of x,y,theta,x_target,y_target and outputting array of rotational velocities to accomplish this
	Rot_Vel2, rotation_traveled2 = Rotational_Velocity(Movement2[i,0],Movement2[i,1],Movement2[i+1,0],Movement2[i+1,1],Heading2[i],theta_init,dw_max,delta_t)
	Heading2 = np.vstack((Heading2,atan2((Movement2[i+1,1]-Movement2[i,1]),(Movement2[i+1,0]-Movement2[i,0])) + theta_init)) #updating current heading from theta=0

	#Setting up two arrays to hold robot heading and position for each movement
	robot_heading2 = np.zeros((len(Rot_Vel2),1))
	robot_position2 = np.zeros((len(Rot_Vel2),3))

	#Calculating and storing heading of robot based on array of rotational velocities
	j2 = 0
	while j2<len(Rot_Vel2):
		if j2==0: #first element of array when w=0
			robot_heading2[j2,0] = Rot_Vel2[j2]*delta_t

			robot_position2[j2,0] = Movement2[i,0]
			robot_position2[j2,1] = Movement2[i,1]
			robot_position2[j2,2] = robot_heading2[j2,0]

			j2 = j2+1
		else: #all others
			robot_heading2[j2,0] = Rot_Vel2[j2]*delta_t + robot_heading2[j2-1,0]

			robot_position2[j2,0] = Movement2[i,0]
			robot_position2[j2,1] = Movement2[i,1]
			robot_position2[j2,2] = robot_position2[j2-1,2] + robot_heading2[j2,0]

			j2 = j2+1


	#Calling the Velocity function with inputs of x,y,x_target,y_target and outputting array of velocities to accomplish this
	Vel2 = Velocity(Movement2[i,0],Movement2[i,1],Movement2[i+1,0],Movement2[i+1,1],dv_max,delta_t)

	#Setting up two arrays to hold robot position changes and overall position for each movement
	robot_position_changes21 = np.zeros((len(Vel2),3))
	robot_position21 = np.zeros((len(Vel2),3))

	#Calculating and storing pose of robot based on array of velocities
	k2 = 0
	while k2<len(Vel2):
		if k2==0: #first element of array when v=0
			robot_position_changes21[k2,0] = Vel2[k2]*delta_t*cos(Heading2[i])
			robot_position_changes21[k2,1] = Vel2[k2]*delta_t*sin(Heading2[i])
			robot_position_changes21[k2,2] = 0
			robot_position21[k2,0] = robot_position_changes21[k2,0] + Movement2[i,0] #accounting for initial position
			robot_position21[k2,1] = robot_position_changes21[k2,1] + Movement2[i,1]
			robot_position21[k2,2] = robot_heading2[j2-1,0]

			#Plotting movement for planned path
			plt.scatter(robot_position21[k2,0],robot_position21[k2,1],label='Path Driven #2')
			plt.draw()
			plt.pause(0.000001)

			k2 = k2+1
		else: #all others
			robot_position_changes21[k2,0] = Vel2[k2]*delta_t*cos(Heading2[i+1]-theta_init)
			robot_position_changes21[k2,1] = Vel2[k2]*delta_t*sin(Heading2[i+1]-theta_init)
			robot_position_changes21[k2,2] = 0
			robot_position21[k2,0] = robot_position_changes21[k2,0]+robot_position21[k2-1,0]
			robot_position21[k2,1] = robot_position_changes21[k2,1]+robot_position21[k2-1,1]
			robot_position21[k2,2] = Heading2[i]

			#Plotting movement for planned path
			plt.scatter(robot_position21[k2,0],robot_position21[k2,1],label='Path Driven #2')
			plt.draw()
			plt.pause(0.000001)

			k2 = k2+1

	i = i+1


#Path planning for set #3
i = 0
while h3>1:
	#Calculating heurisitic function f=g+h for set #3
	C3, g3, h3 = Movement(C3[0],C3[1],G3[0],G3[1],g3,grid_size,landmark_space)

	#Tracking and plotting movement for set #3
	Movement3 = np.vstack((Movement3,C3));

	Movement3_x = Movement3[:,0]
	Movement3_y = Movement3[:,1]

	plt.plot(Movement3_x,Movement3_y,label='Set 3')
	plt.draw()
	plt.pause(0.000001)


	#Calling the Rotational Velocity function with inputs of x,y,theta,x_target,y_target and outputting array of rotational velocities to accomplish this
	Rot_Vel3, rotation_traveled3 = Rotational_Velocity(Movement3[i,0],Movement3[i,1],Movement3[i+1,0],Movement3[i+1,1],Heading3[i],theta_init,dw_max,delta_t)
	Heading3 = np.vstack((Heading3,atan2((Movement3[i+1,1]-Movement3[i,1]),(Movement3[i+1,0]-Movement3[i,0])) + theta_init)) #updating current heading from theta=0

	#Setting up two arrays to hold robot heading and position for each movement
	robot_heading3 = np.zeros((len(Rot_Vel3),1))
	robot_position3 = np.zeros((len(Rot_Vel3),3))

	#Calculating and storing heading of robot based on array of rotational velocities
	j3 = 0
	while j3<len(Rot_Vel3):
		if j3==0: #first element of array when w=0
			robot_heading3[j3,0] = Rot_Vel3[j3]*delta_t

			robot_position3[j3,0] = Movement3[i,0]
			robot_position3[j3,1] = Movement3[i,1]
			robot_position3[j3,2] = robot_heading3[j3,0]

			j3 = j3+1
		else: #all others
			robot_heading3[j3,0] = Rot_Vel3[j3]*delta_t + robot_heading3[j3-1,0]

			robot_position3[j3,0] = Movement3[i,0]
			robot_position3[j3,1] = Movement3[i,1]
			robot_position3[j3,2] = robot_position3[j3-1,2] + robot_heading3[j3,0]

			j3 = j3+1


	#Calling the Velocity function with inputs of x,y,x_target,y_target and outputting array of velocities to accomplish this
	Vel3 = Velocity(Movement3[i,0],Movement3[i,1],Movement3[i+1,0],Movement3[i+1,1],dv_max,delta_t)

	#Setting up two arrays to hold robot position changes and overall position for each movement
	robot_position_changes31 = np.zeros((len(Vel3),3))
	robot_position31 = np.zeros((len(Vel3),3))

	#Calculating and storing pose of robot based on array of velocities
	k3 = 0
	while k3<len(Vel3):
		if k3==0: #first element of array when v=0
			robot_position_changes31[k3,0] = Vel3[k3]*delta_t*cos(Heading3[i])
			robot_position_changes31[k3,1] = Vel3[k3]*delta_t*sin(Heading3[i])
			robot_position_changes31[k3,2] = 0
			robot_position31[k3,0] = robot_position_changes31[k3,0] + Movement3[i,0] #accounting for initial position
			robot_position31[k3,1] = robot_position_changes31[k3,1] + Movement3[i,1]
			robot_position31[k3,2] = robot_heading3[j3-1,0]

			#Plotting movement for planned path
			plt.scatter(robot_position31[k3,0],robot_position31[k3,1],label='Path Driven #3')
			plt.draw()
			plt.pause(0.000001)

			k3 = k3+1
		else: #all others
			robot_position_changes31[k3,0] = Vel3[k3]*delta_t*cos(Heading3[i+1]-theta_init)
			robot_position_changes31[k3,1] = Vel3[k3]*delta_t*sin(Heading3[i+1]-theta_init)
			robot_position_changes31[k3,2] = 0
			robot_position31[k3,0] = robot_position_changes31[k3,0]+robot_position31[k3-1,0]
			robot_position31[k3,1] = robot_position_changes31[k3,1]+robot_position31[k3-1,1]
			robot_position31[k3,2] = Heading3[i]

			#Plotting movement for planned path
			plt.scatter(robot_position31[k3,0],robot_position31[k3,1],label='Path Driven #3')
			plt.draw()
			plt.pause(0.000001)

			k3 = k3+1

	i = i+1

