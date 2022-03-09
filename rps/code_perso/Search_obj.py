'''
Réecriture

 Thomas Boursa
 09/03/2022
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

Objectif = np.array([[-1],[0.8]])
close_enough = 0.03 ; #a quelle distance minimum du waypoint doit etre le follower afin de valider son étape

####Creation du Laplacien####
#On cree un graphe complet => tous les sommets sont adjacent 2 a 2
followers = -completeGL(N-1) #N-A car Un leader
L = np.zeros((N,N)) #Matrice de zero pour pallier les eventuelles erreures memoires
L[1:N,1:N] = followers
L[1,1] = L[1,1]+1
L[1,0] = -1
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
#SI to UNI mapping
_,uni_to_si_states = create_si_to_uni_mapping()#SI barrieere certificat : avoid les collisions
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




