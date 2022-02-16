import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
from numpy import linalg as LA
import time

##ON INTITIALISE LE SIMULATEUR ROBOTARIUM #############

N = 1 #1 ROBOT
initial_conditions = generate_initial_conditions(N) #COND INITIAL ALEATOIRE
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=False) 




 ##ON DEFINIT LES OBJECTIFS ##
error_margin = 0.02 #mARGE D'ERREUR POUR LA POSITION FINALE

# Definie la postion finale des robots (objectif de position)
goal_points = np.array(np.mat('0 ; 0 ; 0'))


##ON APPELLE LES OUTILS NECESSAIRES ##
# Create le suivi de posion d'un agent
single_integrator_position_controller = create_si_position_controller()

# Create des barriere de certification pour eviter les obstacles
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

# Cree un emap des etats de tout les agents ??
_, uni_to_si_states = create_si_to_uni_mapping()


# Create mapping from single integrator velocity commands to unicycle velocity commands ??????
si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()

#obternir la position des robots 
x = r.get_poses()
#obternir la position de chaque robot ?
x_si = uni_to_si_states(x)
#avance la simu
r.step()

#tant que la distance esntre le robot et l'objectif est < margin  et que tout les robots ne sont pas arrivé (fonction at_pose fait tout en meme temps)


##ON BOUCLE LA SIMUATION##
while (np.size(at_pose(np.vstack((x_si,x[2,:])), goal_points, rotation_error=100)) != N):

    #on recupere les états
    x = r.get_poses()
    x_si = uni_to_si_states(x)

    # Create single-integrator control inputs
    dxi = single_integrator_position_controller(x_si ,goal_points[:2][:])

    # Create safe control inputs (i.e., no collisions)
    dxi = si_barrier_cert(dxi, x_si)

    # Transform single integrator velocity commands to unicycle
    dxu = si_to_uni_dyn(dxi, x)
    # Set the velocities by mapping the single-integrator inputs to unciycle inputs
    r.set_velocities(np.arange(N), dxu)
    r.step()

r.call_at_scripts_end()