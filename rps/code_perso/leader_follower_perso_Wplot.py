'''
Réecriture

 Thomas Boursa
 04/03/2022
'''

#Import Robotarium Utilities
from os import stat
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
N=4 #Number of robots to use, this must stay 4 unless the Laplacian is changed.

waypoints = np.array([[-1,-1,1,1],[0.8, -0.8, -0.8, 0.8]]) #les points parlesquelle va passer le Leader
close_enough = 0.03 ; #a quelle distance minimum du waypoint doit etre le follower afin de valider son étape

####Creation du Laplacien####
#On cree un graphe complet => tous les sommets sont adjacent 2 a 2
followers = -completeGL(N-1) #N-A car Un leader
L = np.zeros((N,N)) #Matrice de zero pour pallier les eventuelles erreures memoires
L[1:N,1:N] = followers
L[1,1] = L[1,1]+1
L[1,0] = -1

#Trouver les connexions 
"""
Ici, la matrice ressmble a :
 
 	    1   2   3   4         <=Robot numero (1 => Leader)
	 
 1 	  [ 0.  0.  0.  0.]
 2    [-1. -1.  1.  1.]
 3    [ 0.  1. -2.  1.]
 4    [ 0.  1.  1. -2.]


 Il y a une connexion quand il y a un 1
 On s'apercoit que les trois leaders sont liés 
"""
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
initial_conditions = np.array([[0, 0.5, 0.3, -0.1],[0.5, 0.5, 0.2, 0],[0, 0, 0, 0]])

#On initialise la classe robotarium 
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=True)

#######Utilisations des outils de conversion SI to UNI et barriere colisions#######
#On stocke ces fonctions utiles dans des variables
#SI to UNI mapping
_,uni_to_si_states = create_si_to_uni_mapping()
"""
Cette fct renvoie 2 fonction :
si_to_uni_dyn => prend vitesse SI et transform en controle UNI
et
uni_to_si_states => Prend etats UNI transforme en SI state
"""
si_to_uni_dyn = create_si_to_uni_dynamics()
"""
Cette fct renvoie 1 fonction :
uni_to_si_dyn => Converti UNI to SI dynamic => renvoie 2xN array of SI controle input

"""

#SI barrieere certificat : avoid les collisions
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
#SI controlle de position
leader_controller = create_si_position_controller(velocity_magnitude_limit=0.1)
"""Creates a position controller for single integrators.  Drives a single integrator to a point
    using a propoertional controller.

    x_velocity_gain - the gain impacting the x (horizontal) velocity of the single integrator
    y_velocity_gain - the gain impacting the y (vertical) velocity of the single integrator
    velocity_magnitude_limit - the maximum magnitude of the produce velocity vector (should be less than the max linear speed of the platform)

    -> function : si_position_controller(xi, positions):

        xi: 2xN numpy array (of single-integrator states of the robots)
        points: 2xN numpy array (of desired points each robot should achieve)

        -> 2xN numpy array (of single-integrator control inputs)
	"""


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
goal_caption = ['G{0}'.format(ii) for ii in range(waypoints.shape[1])]
#Plot text for caption
waypoint_text = [r.axes.text(waypoints[0,ii], waypoints[1,ii], goal_caption[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-2)
for ii in range(waypoints.shape[1])]
g = [r.axes.scatter(waypoints[0,ii], waypoints[1,ii], s=marker_size_goal, marker='s', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width,zorder=-2)
for ii in range(waypoints.shape[1])]

# Plot Graph Connections
x = r.get_poses() # Need robot positions to do this.
linked_follower_index = np.empty((2,3))
follower_text = np.empty((3,0))
for jj in range(1,int(len(rows)/2)+1):
	linked_follower_index[:,[jj-1]] = np.array([[rows[jj]],[cols[jj]]])
	follower_text = np.append(follower_text,'{0}'.format(jj))

line_follower = [r.axes.plot([x[0,rows[kk]], x[0,cols[kk]]],[x[1,rows[kk]], x[1,cols[kk]]],linewidth=line_width,color='b',zorder=-1)
 for kk in range(1,N)]
line_leader = r.axes.plot([x[0,0],x[0,1]],[x[1,0],x[1,1]],linewidth=line_width,color='r',zorder = -1)
follower_labels = [r.axes.text(x[0,kk],x[1,kk]+0.15,follower_text[kk-1],fontsize=font_size, color='b',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)
for kk in range(1,N)]
leader_label = r.axes.text(x[0,0],x[1,0]+0.15,"Leader",fontsize=font_size, color='r',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=0)

r.step()
########################
#######PLOT#############
###########################



#Boucle

for t in range(iterations) : 
	#obtenir la position 
	x = r.get_poses()
	xi = uni_to_si_states(x) #Conversion etat UNI en SI
########################
#######PLOT#############
###########################


		# Update Plot Handles
	for q in range(N-1):
		follower_labels[q].set_position([xi[0,q+1],xi[1,q+1]+0.15])
		follower_labels[q].set_fontsize(determine_font_size(r,font_size_m))
		line_follower[q][0].set_data([x[0,rows[q+1]], x[0,cols[q+1]]],[x[1,rows[q+1]], x[1,cols[q+1]]])
	leader_label.set_position([xi[0,0],xi[1,0]+0.15])
	leader_label.set_fontsize(determine_font_size(r,font_size_m))
	line_leader[0].set_data([x[0,0],x[0,1]],[x[1,0],x[1,1]])

	# This updates the marker sizes if the figure window size is changed. 
    # This should be removed when submitting to the Robotarium.
	for q in range(waypoints.shape[1]):
		waypoint_text[q].set_fontsize(determine_font_size(r,font_size_m))
		g[q].set_sizes([determine_marker_size(r,0.2)])

########################
#######PLOT#############
###########################




	#Algo :

	###Follower###
	for i in range(1,N) :
		#velocité = 0 et obtention des voisins de i 
		dxi[:,[i]]=np.zeros((2,1))
		neighbors = topological_neighbors(L,i)

		for j in neighbors : 
			dxi[:,[i]] += formation_control_gain*(np.power(np.linalg.norm(x[:2,[j]]-x[:2,[i]]), 2)-np.power(desired_distance, 2))*(x[:2,[j]]-x[:2,[i]])
	
	###Leader####
	waypoint = waypoints[:,state].reshape((2,1))

	dxi[:,[0]] = leader_controller(x[:2,[0]],waypoint) # On donne la position du Leader (indice 0) et son goal ( waypoint)
	if np.linalg.norm(x[:2,[0]]-waypoint) < close_enough:
		state = (state + 1)%4  
		#Si le leader est assez proche de son waypoint, il passe au suivant 
		#Si il atteint le dernier waypoint, il revient au premier grace a %4 (modulo 4)
	
	#On garde le SI controller sous les magnitudes spécifiées
	#Seuil de condtrol input
	norms  = np.linalg.norm(dxi,2,0)
	idxs_to_normalize = (norms > magnitude_limit)
	dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

	#Utilisation des barieres certificats et conversion SI to UNI commands
	dxi = si_barrier_cert(dxi,x[:2,:])
	dxu = si_to_uni_dyn(dxi,x)

	#Ajustement de la velocité des robots 
	r.set_velocities(np.arange(N),dxu)

	#iteration 
	r.step()

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()