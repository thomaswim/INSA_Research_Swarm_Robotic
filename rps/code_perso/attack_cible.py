"""
Si l’emplacement de la cible n’est pas connu : 

	-> Déplacement aléatoirement

Si l’emplacement de la cible est connu :

	Envoyer l’emplacement de la cible à tous mes voisins

	-> Si la cible n’est pas à portée
		Déplacement vers la cible 

	-> Si la cible est à portée 
		Arrêter le mouvement (le robot va alors attaquer la cible)
"""
#Import Robotarium Utilities
import graphlib
from os import stat
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

#Other Imports
import numpy as np
import random as random

# Experiment Constants
iterations = 5000 #Run the simulation/experiment for 5000 steps (5000*0.033 ~= 2min 45sec)
N=4 #Number of robots to use, this must stay 4 unless the Laplacian is changed.


#On cree un graphe complet => tous les sommets sont adjacent 2 a 2
L = -completeGL(N) #N-A car Un leader
 #Matrice de zero pour pallier les eventuelles erreures memoires
print(L)
[rows,cols] = np.where(L==1)

# For computational/memory reasons, initialize the velocity vector
dxi = np.zeros((2,N))
#On crée les gains pour l'algorithme de formation de controle 
formation_control_gain = 10
desired_distance = 0.3

#On pose les condiitons initiales des robots  
initial_conditions = np.array([[0, 0.5, 0.3, -0.1],[0.5, 0.5, 0.2, 0],[0, 0, 0, 0]])

#On initialise la classe robotarium 
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=True)

_,uni_to_si_states = create_si_to_uni_mapping()
si_to_uni_dyn = create_si_to_uni_dynamics()
#SI barrieere certificat : avoid les collisions
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
#SI controlle de position
controller = create_si_position_controller(velocity_magnitude_limit=0.1)

cible = np.array([[-1],[0.8]])
close_enough = 0.05

#----------------------------------------------------------
# Plotting Parameters
CM = np.random.rand(N,3) # Random Colors
goal_marker_size_m = 0.2 # taille de la zone a atteindre 
robot_marker_size_m = 0.15 #taille de la zone de départ 
#definition des parametres du dessus
marker_size_goal = determine_marker_size(r,goal_marker_size_m)  
marker_size_robot = determine_marker_size(r, robot_marker_size_m)
#texte de la zone 
font_size = determine_font_size(r,0.1)
line_width = 2
#----------------------------------------------------------

#Plot text for caption
goal_points_text = [r.axes.text(cible[0,ii], cible[1,ii], cible[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-2)
for ii in range(cible.shape[1])]
goal_markers = [r.axes.scatter(cible[0,ii], cible[1,ii], s=marker_size_goal, marker='s', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width,zorder=-2)
for ii in range(cible.shape[1])]

cible_detected = 0

for t in range(iterations) :
    x = r.get_poses()
    xi = uni_to_si_states(x)





    for i in range(1,N) :

        if (cible_detected == 0) :
            rand = np.array([[random.random()*2-1],[random.random()*2-1]])
            dxi[:,[i]] = controller(x[:2,[i]],rand)
    
    dxi = si_barrier_cert(dxi,x[:2,:])
    dxu = si_to_uni_dyn(dxi,x)

    r.set_velocities(np.arange(N),dxu)

    
    r.step()

    	



	#Algo :
    #iteration 
