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

#---------------- Création du GOAL ------------------

# Plotting Parameters
CM = np.random.rand(N,3) # Random Colors
goal_marker_size_m = 0.1 # taille de la zone a atteindre 

#definition des parametres du dessus
marker_size_goal = determine_marker_size(r,goal_marker_size_m)  

#texte de la zone 
font_size = determine_font_size(r,0.1)
line_width = 5

#on defini nouvelles coordonnées du goal  
goal_points2 = np.array(np.mat('0; -0.5; 0')) 
#definition du goal 
goal_caption2 = ['o'.format(ii) for ii in range(goal_points2.shape[1])] 


#definition graphique du goal 
goal_points_text2 = [r.axes.text(goal_points2[0,ii], goal_points2[1,ii], goal_caption2[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-2)
for ii in range(goal_points2.shape[1])]
goal_markers2 = [r.axes.scatter(goal_points2[0,ii], goal_points2[1,ii], s=marker_size_goal, marker='o', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width,zorder=-2)
for ii in range(goal_points2.shape[1])]    



goal_check = True
t=0
#--------------Boucle----------------------

while (t<iterations and goal_check) : 
    #obtenir la position 
    x = r.get_poses()
    t=t+1
    xi = uni_to_si_states(x) #Conversion etat UNI en SI

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



while (np.size(at_pose(np.vstack((x_si,x[2,:])), goal_points2, rotation_error=100)) != N):

    #on recupere les états
    x = r.get_poses()
    x_si = uni_to_si_states(x)

    # Create single-integrator control inputs
    dxi = single_integrator_position_controller(x_si ,goal_points2[:2][:])

    # Create safe control inputs (i.e., no collisions)
    dxi = si_barrier_cert(dxi, x_si)

    # Transform single integrator velocity commands to unicycle
    dxu = si_to_uni_dyn(dxi, x)
    # Set the velocities by mapping the single-integrator inputs to unciycle inputs
    r.set_velocities(np.arange(N), dxu)
    r.step()
#---------------

#coucou 

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()