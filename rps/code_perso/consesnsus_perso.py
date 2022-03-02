import rps.robotarium as robotarium
from rps.utilities.graph import *
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np

#initialiser le robotarium
N = 10
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)

#nombre iterations
iteration = 1000


# Default Barrier Parameters
safety_radius = 0.17

#on travail avec SI dynamique et on ne veut pas que les robos se collide ou
#sorte du testbed => on utilise les barrieres certificats

si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

#SI to UNI
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

#On genere un graphe Laplacien connecté (graphe circulaire)
L = cycle_GL(N)

#trouver les connexions
[rows,cols] = np.where(L==1)


####################################
####CREATION TEXT ET MARKER############
#####################################

"""
# Plotting Parameters
CM = np.random.rand(N,3) # Random Colors
safety_radius_marker_size = determine_marker_size(r,safety_radius) # Will scale the plotted markers to be the diameter of provided argument (in meters)
font_height_meters = 0.1
font_height_points = determine_font_size(r,font_height_meters) # Will scale the plotted font height to that of the provided argument (in meters)
# Initial plots
x=r.get_poses()
g = r.axes.scatter(x[0,:], x[1,:], s=np.pi/4*safety_radius_marker_size, marker='o', facecolors='none',edgecolors=CM,linewidth=7)
#g = r.axes.plot(x[0,:], x[1,:], markersize=safety_radius_marker_size, linestyle='none', marker='o', markerfacecolor='none', markeredgecolor=color[CM],linewidth=7)
linked_follower_index = np.empty((2,3))
follower_text = np.empty((3,0))

r.step()

"""





#########################################
####### END TEXT ET MARKER #############
######################################""

for k in range(iteration) :
    #on recuperer la position des robots qu'on converti en SI model
    x = r.get_poses()
    x_si = uni_to_si_states(x)
    # Update Plotted Visualization
    """
    g.set_offsets(x[:2,:].T)
    # This updates the marker sizes if the figure window size is changed. 
    # This should be removed when submitting to the Robotarium.
    g.set_sizes([determine_marker_size(r,safety_radius)])
"""
    #on initialise le SI control input avec des zeros (pour pallier le probleme de memoire)
    si_velocities = np.zeros((2,N))

    #pour chaque robot :
    for i in range(N) :
        #on recuperer 'j', le voisin du robot 'i' (encodé dans le laplacien L)
        j =  topological_neighbors(L,i)
        #on applique l'algo consensus
        """ Consensus algo :
        dxi = somme ( xj - xi )
        """
        si_velocities[:, i] = np.sum(x_si[:, j] - x_si[:, i, None], 1)

#on utilise la barriere certificat  pour eviter les collisions
    si_velocities = si_barrier_cert(si_velocities, x_si)     

#transform SI to UNI
    dxu = si_to_uni_dyn(si_velocities,x)

#On ajuste la velocité des robots 1...N
    r.set_velocities(np.arange(N),dxu)
    #on step 
    r.step()

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
