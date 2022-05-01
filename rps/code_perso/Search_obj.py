'''
Réecriture

 Thomas Boursa
 09/03/2022
'''

#Import Robotarium Utilities

from os import stat
import random
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

#Other Imports
import numpy as np

# Experiment Constants
iterations = 5000 #Run the simulation/experiment for 5000 steps (5000*0.033 ~= 2min 45sec)
N=1 #Number of robots to use, this must stay 4 unless the Laplacian is changed.

Objectif = np.array([[-0.5],[0.8]])
close_enough = 0.1 ; #a quelle distance minimum du waypoint doit etre le follower afin de valider son étape

####Creation du Laplacien####
#On cree un graphe complet => tous les sommets sont adjacent 2 a 2
#followers = -completeGL(N) #N-A car Un leader
followers = -cycle_GL(N)
L = np.zeros((N,N)) #Matrice de zero pour pallier les eventuelles erreures memoires
L = followers
print(L)


[rows,cols] = np.where(L==1)

# For computational/memory reasons, initialize the velocity vector
dxi = np.zeros((2,N))

##On initialise l'état du Leader à 0
state = 0

#On limite l'amplitude de vitesse des robots 
magnitude_limit = 0.15 

#On crée les gains pour l'algorithme de formation de controle 
formation_control_gain = 10
desired_distance = 0.3

#On pose les condiitons initiales des robots  
initial_conditions = np.array([[-1.5],[-0.6],[0]])

#On initialise la classe robotarium 
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=False)
#SI to UNI mapping
_,uni_to_si_states = create_si_to_uni_mapping()#SI barrieere certificat : avoid les collisions

si_to_uni_dyn = create_si_to_uni_dynamics()

si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
#SI controlle de position
leader_controller = create_si_position_controller(velocity_magnitude_limit=0.1)

########################
#######PLOT#############
###########################

# Plotting Parameters
CM = np.random.rand(N,3) # Random Colors
marker_size_goal = determine_marker_size(r,0.2)
font_size_m = 0.1
font_size = determine_font_size(r,font_size_m)
line_width = 5

# Create goal text and markers

#Text with goal identification
goal_caption = ['G{0}'.format(ii) for ii in range(Objectif.shape[1])]
#Plot text for caption
goal_points_text = [r.axes.text(Objectif[0,ii], Objectif[1,ii], goal_caption[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-2)
for ii in range(Objectif.shape[1])]
goal_markers = [r.axes.scatter(Objectif[0,ii], Objectif[1,ii], s=marker_size_goal, marker='s', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width,zorder=-2)
for ii in range(Objectif.shape[1])]

# Plot Graph Connections
x = r.get_poses() # Need robot positions to do this.
print(x)
linked_follower_index = np.empty((2,4))
follower_text = np.empty((3,0))
for jj in range(1,int(len(rows))):

	follower_text = np.append(follower_text,'{0}'.format(jj))

line_follower = [r.axes.plot([x[0,rows[kk]], x[0,cols[kk]]],[x[1,rows[kk]], x[1,cols[kk]]],linewidth=line_width,color='b',zorder=-1)
for kk in range(0,N)]
##line_leader = r.axes.plot([x[0,0],x[0,1]],[x[1,0],x[1,1]],linewidth=line_width,color='r',zorder = -1)
follower_labels = [r.axes.text(x[0,kk],x[1,kk]+0.15,kk,fontsize=font_size, color='b',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
for kk in range(0,N)]
##leader_label = r.axes.text(x[0,0],x[1,0]+0.15,"Leader",fontsize=font_size, color='r',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)



xp=-1.5
yp=0.9

v = 0

########################
#######PLOT#############
###########################

waypoint_check = np.zeros(N)
waypoint = np.zeros((2,N))
Obj_find = 0

r.step()
for t in range(iterations):

	# Get the most recent pose information from the Robotarium. The time delay is
	# approximately 0.033s
	x = r.get_poses()
	xi = uni_to_si_states(x)

	# Update Plot Handles
	for q in range(N):
		follower_labels[q].set_position([xi[0,q],xi[1,q]+0.15])
		follower_labels[q].set_fontsize(determine_font_size(r,font_size_m))
		line_follower[q][0].set_data([x[0,rows[q]], x[0,cols[q]]],[x[1,rows[q]], x[1,cols[q]]])


	# This updates the marker sizes if the figure window size is changed. 
    # This should be removed when submitting to the Robotarium.

	#Algorithm

	for i in range(0,N):

		# Zero velocities and get the topological neighbors of agent i
		dxi[:,[i]]=np.zeros((2,1))
		neighbors = topological_neighbors(L,i)
		






		if (waypoint_check[i] == 0) and Obj_find ==0:
			if v==0 :
				xp = xp+.1
				yp = yp
				v=1
			if v==1 : 
				xp=xp
				yp=-yp
				v=0
			waypoint[[0],[i]] = xp
			waypoint[[1],[i]] = yp
			##on etabli un nouveau waypoint
			##dxi[:,[i]] = leader_controller(x[:2,[0]], waypoint[:,[i]])
			waypoint_check[i] = 1
		
		if np.linalg.norm(x[:2,[i]]-waypoint[:,[i]]) < close_enough and Obj_find ==0:
			print("FIND",i)
			waypoint_check[i] = 0
			print(waypoint)
		
		if np.linalg.norm(x[:2,[i]]-Objectif) < desired_distance and Obj_find ==0:
			##print("FIND IT")
			Obj_find = 1
			dxi[:,[i]] = [[0],[0]]
			for j in neighbors:
				waypoint[:,[j]] = xi[:,[i]]
				waypoint_check[j] = 1
				dxi[:,[j]] += leader_controller(x[:2,[j]], waypoint[:,[j]])
				print("voisin" , j)


		else:
			#Use barriers and convert single-integrator to unicycle commands
			dxi[:,[i]] = leader_controller(x[:2,[i]], waypoint[:,[i]])
	dxi = si_barrier_cert(dxi, x[:2,:])
	dxu = si_to_uni_dyn(dxi,x)
	r.set_velocities(np.arange(N),dxu)

	
	##print(waypoint)
	# Iterate the simulation
	r.step()

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()


def SayInfo(voisin) :
	j =topological_neighbors(L,voisin)
	

