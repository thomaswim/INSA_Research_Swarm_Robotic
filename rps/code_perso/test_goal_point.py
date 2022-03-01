import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
from numpy import linalg as LA
import time

#----------------------------------------------------------
# FONCTIONS 
#----------------------------------------------------------

#PAS FONCTIONNEL POUR LE MOMENT 
def avance(goal_point) :

    #on recupere les états
    x = r.get_poses()
    x_si = uni_to_si_states(x)

    # Create single-integrator control inputs
    dxi = single_integrator_position_controller(x_si ,goal_point[:2][:])

    # Create safe control inputs (i.e., no collisions)
    dxi = si_barrier_cert(dxi, x_si)

    # Transform single integrator velocity commands to unicycle
    dxu = si_to_uni_dyn(dxi, x)
    # Set the velocities by mapping the single-integrator inputs to unciycle inputs
    r.set_velocities(np.arange(N), dxu)
    r.step()
#----------------------------------------------------------





#----------------------------------------------------------
##ON INTITIALISE LE SIMULATEUR ROBOTARIUM #############
#----------------------------------------------------------

N = 2 #NB ROBOT
initial_conditions = generate_initial_conditions(N) #COND INITIAL ALEATOIRE
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=False)  




 ##ON DEFINIT LES OBJECTIFS ##
error_margin = 0.02 #mARGE D'ERREUR POUR LA POSITION FINALE

# Definie la postion finale des robots (objectif de position)
goal_points = np.array(np.mat('0.5; 0.5; 0'))   #x y et radiant   


##ON APPELLE LES OUTILS NECESSAIRES ##
# Create le suivi de position d'un agent
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

#----------------------------------------------------------
# Plotting Parameters
CM = np.random.rand(N,3) # Random Colors
goal_marker_size_m = 0.2 # taille de la zone a atteindre 
robot_marker_size_m = 0.15 #taille de la zone de départ 
#definition des parametres du dessus
marker_size_goal = determine_marker_size(r,goal_marker_size_m)  
marker_size_robot = determine_marker_size(r, robot_marker_size_m)
#texte de la zone 
font_size = determine_font_size(r,0.2)
line_width = 5
#----------------------------------------------------------
# Create Goal Point Markers
#Text with goal identification
goal_caption = ['A'.format(ii) for ii in range(goal_points.shape[1])] #definition du goal 
#----------------------------------------------------------
#Plot text for caption
goal_points_text = [r.axes.text(goal_points[0,ii], goal_points[1,ii], goal_caption[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-2)
for ii in range(goal_points.shape[1])]
goal_markers = [r.axes.scatter(goal_points[0,ii], goal_points[1,ii], s=marker_size_goal, marker='s', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width,zorder=-2)
for ii in range(goal_points.shape[1])]




#----------------------------------------------------------
#TEST DE 2e POINT GOAL 
#on defini nouvelles coordonnées du goal  
goal_points2 = np.array(np.mat('0; -0.5; 0')) 
#definition du goal 
goal_caption2 = ['B'.format(ii) for ii in range(goal_points2.shape[1])] 
#definition graphique du goal 
goal_points_text2 = [r.axes.text(goal_points2[0,ii], goal_points2[1,ii], goal_caption2[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-2)
for ii in range(goal_points2.shape[1])]
goal_markers2 = [r.axes.scatter(goal_points2[0,ii], goal_points2[1,ii], s=marker_size_goal, marker='s', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width,zorder=-2)
for ii in range(goal_points2.shape[1])]
#----------------------------------------------------------

#----------------------------------------------------------
# Definition de la zone d'apparition du robot 
robot_markers = [r.axes.scatter(x[0,ii], x[1,ii], s=marker_size_robot, marker='o', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width) 
for ii in range(goal_points.shape[1])]
#----------------------------------------------------------

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


print("goal 1 atteint !!")

#----------------------------------------------------------
# Atteinte du 2e point 

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
#----------------------------------------------------------


r.call_at_scripts_end()