#!/usr/bin/env python

import pkg_resources
pkg_resources.require("klampt>=0.7.0")
from klampt import *
from klampt import vis 
from klampt.vis.glrobotprogram import *
from klampt.math import *
from klampt.model import collide
from klampt.io import resource
from klampt.sim import *
from moving_base_control import *
import csv
import importlib
import math
import os
import string
import sys, traceback
import time
import pickle
import colorama

from klampt.math import so3, se3
import numpy as np
from IPython import embed
from mvbb.draw_bbox import draw_GL_frame, draw_bbox
from i16mc import make_moving_base_robot
from mvbb.CollisionCheck import CheckCollision, CollisionTestInterpolate, CollisionTestPose
from mvbb.box_db import MVBBLoader

from plugins import soft_hand

#added by GA
from mvbb.ScalaReduce import DanyReduceScale
from i16mc import make_object
from create_mvbb import MVBBVisualizer, compute_poses, skip_decimate_or_return
# import pydany_bb


objects = {}
robots = ['reflex_col', 'soft_hand', 'reflex']

def make_box(world, x_dim, y_dim, z_dim, mass=0.5):
    """Makes a new axis-aligned box centered at the origin with
    dimensions width x depth x height. The box is a RigidObject
    with automatically determined inertia.
        """
    boxgeom = Geometry3D()
    boxgeom.loadFile("data/objects/cube.tri")

    # box is centered at the origin
    boxgeom.transform([x_dim, 0, 0, 0, y_dim, 0, 0, 0, z_dim], [-x_dim * 0.5, -y_dim * 0.5, -z_dim * 0.5])

    print "Making a box a rigid object"
    bmass = Mass()
    bmass.setMass(mass)
    bmass.setCom([0, 0, 0])
    bmass.setInertia([x_dim / 12, y_dim / 12, z_dim / 12])
    box = world.makeRigidObject("box")
    box.geometry().set(boxgeom)
    box.appearance().setColor(0.6, 0.3, 0.2, 1.0)
    box.setMass(bmass)
    cparams = box.getContactParameters()
    cparams.kFriction = 1.5
    cparams.kStiffness = 100000
    cparams.kDamping = 30000
    cparams.kRestitution = 0.5


    return box

class FilteredMVBBTesterVisualizer(GLRealtimeProgram):
    def __init__(self, box_dims, poses, world, p_T_h, module, box_db, links_to_check = None):
        GLRealtimeProgram.__init__(self, "FilteredMVBBTEsterVisualizer")
        self.world = world
        self.p_T_h = p_T_h
        self.h_T_p = np.linalg.inv(self.p_T_h)
        self.poses = []
        self.hand = None
        self.is_simulating = False
        self.curr_pose = None
        self.robot = self.world.robot(0)
        self.q_0 = self.robot.getConfig()
        self.w_T_o = None
        self.obj = None
        self.box_dims = box_dims
        self.t_0 = None
        self.object_com_z_0 = None
        self.object_fell = None
        self.sim = None
        self.module = module
        self.running = True
        self.db = box_db
        self.links_to_check = links_to_check

        if self.world.numRigidObjects() > 0:
            self.obj = self.world.rigidObject(0)
            self.w_T_o = np.array(se3.homogeneous(self.obj.getTransform()))
            for p in poses:
                if not self.db.has_simulation(self.box_dims, p):
                    self.poses.append(p)
                else:
                    print "Pose", p, "already simulated"
        else:
            "Warning: during initialization of visualizer object is still not loaded in world"
            self.poses = poses



        print "Will simulate", len(self.poses), "poses,"

    def display(self):
        if self.running:
            self.world.drawGL()

            w_T_o = np.array(se3.homogeneous(self.obj.getTransform()))

            for pose in self.poses:
                w_T_p_des = w_T_o.dot(pose)
                w_T_p__des_se3 = se3.from_homogeneous(w_T_p_des)
                draw_GL_frame(w_T_p__des_se3, color=(0.5,0.5,0.5))
            if self.curr_pose is not None:
                w_T_p_des = w_T_o.dot(self.curr_pose)
                w_T_p__des_se3 = se3.from_homogeneous(w_T_p_des)
                draw_GL_frame(w_T_p__des_se3)

            hand_xform = get_moving_base_xform(self.robot)
            w_T_p = np.array(se3.homogeneous(hand_xform)).dot(self.h_T_p)
            w_T_p_se3 = se3.from_homogeneous(w_T_p)
            draw_GL_frame(w_T_p_se3)

            glDisable(GL_LIGHTING)
            glDisable(GL_DEPTH_TEST)
            glEnable(GL_POINT_SMOOTH)
            glColor3f(1,1,0)
            glLineWidth(1.0)
            glPointSize(5.0)
            forceLen = 0.01  # scale of forces
            if self.sim is not None and self.obj is not None and self.robot is not None:
                c_p, c_f = getObjectPhalanxMeanContactPoint(self.sim, self.obj, self.robot, self.links_to_check)
                n_c_p = countContactPoints(c_p)
                if countContactPoints(c_p) > 0:
                    glBegin(GL_POINTS)
                    for i in range(len(c_p)/3):
                        o_c_p_i = c_p[3*i:3*i+3]
                        if np.all([False if math.isnan(p) else True for p in o_c_p_i]):
                            w_c_p_i = se3.apply(se3.from_homogeneous(w_T_o), o_c_p_i)
                            glVertex3f(*w_c_p_i)
                    glEnd()

                    glBegin(GL_LINES)
                    for i in range(len(c_p)/3):
                        o_c_p_i = c_p[3 * i:3 * i + 3]
                        o_c_f_i = c_f[6 * i:6 * i + 3]
                        if np.all([False if math.isnan(f) else True for f in o_c_f_i]):
                            if np.all([False if math.isnan(p) else True for p in o_c_p_i]):
                                w_c_p_i = se3.apply(se3.from_homogeneous(w_T_o), o_c_p_i)
                                w_c_f_i = se3.apply(se3.from_homogeneous(w_T_o), o_c_f_i)
                                glVertex3f(*w_c_p_i)
                                glVertex3f(*vectorops.madd(w_c_p_i, w_c_f_i, forceLen))
                    glEnd()

    def idle(self):
        if not self.running:
            return

        if self.world.numRigidObjects() > 0:
            self.obj = self.world.rigidObject(0)
        else:
            return

        if not self.is_simulating:
            if len(self.poses) > 0:
                self.curr_pose = self.poses.pop(0)

                print "\n\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!"
                print "!!!!!!!!!!!!!!!!!!!!!!!!!!"
                print "!!!!!!!!!!!!!!!!!!!!!!!!!!"
                print "Simulating Next Pose Grasp"
                print "Dims:\n", self.box_dims
                print "Pose:\n", self.curr_pose
                print "!!!!!!!!!!!!!!!!!!!!!!!!!!"
                print "!!!!!!!!!!!!!!!!!!!!!!!!!!"
                print "!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n\n"

            else:
                print "Done testing all", len(self.poses), "poses for object", self.box_dims
                print "Quitting"
                self.running = False
                vis.show(hidden=True)
                return

            w_T_o_se3 = se3.from_homogeneous(self.w_T_o)
            self.obj.setTransform(w_T_o_se3[0], w_T_o_se3[1])
            w_T_h_des_se3 = se3.from_homogeneous(self.w_T_o.dot(self.curr_pose).dot(self.p_T_h))
            self.robot.setConfig(self.q_0)
            set_moving_base_xform(self.robot, w_T_h_des_se3[0], w_T_h_des_se3[1])

            if self.sim is None:
                self.sim = SimpleSimulator(self.world)
                self.hand = self.module.HandEmulator(self.sim, 0, 6, 6)
                self.sim.addEmulator(0, self.hand)
                # the next line latches the current configuration in the PID controller...
                self.sim.controller(0).setPIDCommand(self.robot.getConfig(),
                                                     vectorops.mul(self.robot.getVelocity(),0.0))

                obj_b = self.sim.body(self.obj)
                obj_b.setVelocity([0., 0., 0.],[0., 0., 0.])

                # setup the preshrink
                visPreshrink = False  # turn this to true if you want to see the "shrunken" models used for collision detection
                for l in range(self.robot.numLinks()):
                    self.sim.body(self.robot.link(l)).setCollisionPreshrink(visPreshrink)
                for l in range(self.world.numRigidObjects()):
                    self.sim.body(self.world.rigidObject(l)).setCollisionPreshrink(visPreshrink)

            self.object_com_z_0 = getObjectGlobalCom(self.obj)[2]
            self.object_fell = False
            self.t_0 = self.sim.getTime()
            self.is_simulating = True

        if self.is_simulating:
            t_lift = 1.3 # when to lift
            d_lift = 1.0 # duration
            # print "t:", self.sim.getTime() - self.t_0
            object_com_z = getObjectGlobalCom(self.obj)[2]
            w_T_h_curr_se3 = get_moving_base_xform(self.robot)
            w_T_h_des_se3 = se3.from_homogeneous(self.w_T_o.dot(self.curr_pose).dot(self.p_T_h))


            if self.sim.getTime() - self.t_0 == 0:
                # print "Closing hand"
                self.hand.setCommand([1.0])
            elif (self.sim.getTime() - self.t_0) >= t_lift and (self.sim.getTime() - self.t_0) <= t_lift+d_lift:
                # print "Lifting"
                t_i = w_T_h_des_se3[1]
                t_f = vectorops.add(t_i, (0,0,0)) # previously was (0,0,0.2)
                #we could reduce the amount of time for each simulation
                u = np.min((self.sim.getTime() - self.t_0 - t_lift, 1.))
                send_moving_base_xform_PID(self.sim.controller(0), w_T_h_des_se3[0], vectorops.interpolate(t_i, t_f ,u))

            if (self.sim.getTime() - self.t_0) >= t_lift: # wait for a lift before checking if object fell
                d_hand = w_T_h_curr_se3[1][2] - w_T_h_des_se3[1][2]
                d_com = object_com_z - self.object_com_z_0
                if d_hand - d_com > 0.1:
                    self.object_fell = True
                    print "!!!!!!!!!!!!!!!!!!"
                    print "Grasp Unsuccessful"
                    print "!!!!!!!!!!!!!!!!!!"

            self.sim.simulate(0.01)
            self.sim.updateWorld()

            if not vis.shown() or (self.sim.getTime() - self.t_0) >= 2.5 or self.object_fell:
                if vis.shown(): # simulation stopped because it was successful
                    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                    print "Saving grasp, status:", "failure" if self.object_fell else "success"
                    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n"
                    w_T_h_curr = np.array(se3.homogeneous(w_T_h_curr_se3))
                    w_T_o_curr = np.array(se3.homogeneous(self.obj.getTransform()))
                    h_T_o = np.linalg.inv(w_T_h_curr).dot(w_T_o_curr)
                    if self.db.n_dofs == self.hand.d_dofs + self.hand.u_dofs:
                        q_grasp = [float('nan')]*self.db.n_dofs if self.object_fell else self.hand.getConfiguration()
                    elif self.db.n_dofs == self.hand.d_dofs + self.hand.u_dofs + self.hand.m_dofs:
                        q_grasp = [float('nan')] * self.db.n_dofs if self.object_fell else self.hand.getFullConfiguration()
                    else:
                        raise Exception('Error: unexcpeted number of joints for hand')

                    c_p, c_f = getObjectPhalanxMeanContactPoint(self.sim, self.obj,
                                                                self.robot, self.links_to_check)

                    try:
                        self.db.save_simulation(self.box_dims, self.curr_pose, h_T_o,
                                                q_grasp, c_p, c_f)
                    except:
                        print "\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
                        print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
                        print "X Error while calling save_simulation X"
                        print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
                        traceback.print_exc()
                        print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"

                self.is_simulating = False
                self.sim = None
                self.robot.setConfig(self.q_0)

def getObjectGlobalCom(obj):
    return se3.apply(obj.getTransform(), obj.getMass().getCom())

def getObjectDims(obj):
    # when margins are zero, the box dims are simply the bounding box dimensions
    BB = obj.geometry().getBB()
    dims = vectorops.sub(BB[1],BB[0])
    return tuple(dims)

def countContactPoints(c_p):
    n_c = len(c_p) / 3
    n_contacts = 0
    for i in range(n_c):
        c_p_i = c_p[3*i:3*i+3]
        if np.all([False if math.isnan(c_p_i[i]) else True for i in range(3)]):
            n_contacts += 1
    return n_contacts

def getObjectPhalanxMeanContactPoint(sim, obj, robot, links = None):
    """
    Returns a contact point for each link in the robot, which is the simple arithmetic mean of all contact points,
    expressed in the object frame of reference
    This is not a contact centroid.
    Also returns the total force applied by the link to the object, and the total moment applied to the object
    (with reference pole in the object origin), expressed in the object frame of reference.

    :param obj: object to grasp
    :param robot: robot grasping the object
    :param links: the links to check for collision
    :return: (cps_avg, wrench_avg) where cps_avg is a vector 3*n_links, and wrench_avg is a vector 6*n_links
    """
    oId = obj.getID()
    lIds = []
    _lIds = [] # if links is not None, this contains the set of link Ids which should not be checked
    lId_to_lIndex = {}
    lId_to_i = {}
    w_T_o = obj.getTransform()
    o_T_w = se3.inv(w_T_o)

    # let's first create the map from lId to lIndex, with all links
    links_to_check = range(robot.numLinks())
    for l_ind in links_to_check:
        link = robot.link(l_ind)
        lId_to_lIndex[link.getID()] = l_ind
        if links is not None and l_ind not in links:
            _lIds.append(link.getID())

    if links is not None:
        links_to_check = links
    for i, l_ind in enumerate(links_to_check):
        link = robot.link(l_ind)
        lIds.append(link.getID())
        lId_to_i[link.getID()] = i

    cps_avg = np.array([float('nan')] * 3 * len(lIds))
    wrench_avg = np.array([float('nan')] * 6 * len(lIds))

    for lId in lIds:
        clist = sim.getContacts(oId, lId)
        pavg = [0, 0, 0]
        navg = [0, 0, 0]
        for c in clist:
            pavg = vectorops.add(pavg, c[0:3])
            navg = vectorops.add(navg, c[3:6])

        if len(clist) > 0:
            pavg = vectorops.div(pavg, len(clist))
            navg = vectorops.div(navg, len(clist))
            l_i = lId_to_i[lId]
            cps_avg[l_i*3:l_i*3+3] = se3.apply(o_T_w, pavg)

            w_F = sim.contactForce(oId, lId) # total force applied on object
            w_M_obj = sim.contactTorque(oId, lId) # total moment applied on object about it's origin

            wrench_avg[l_i*6:l_i*6+3] = se3.apply_rotation(o_T_w, w_F)
            wrench_avg[l_i*6+3:l_i*6+6] = se3.apply_rotation(o_T_w, w_M_obj)

            """ here I should first determine a "contact" reference frame
            cp_M_cp = TODO
            if np.all(wrench_avg[l_i*3+3:l_i*3+5] > 1e-12):
                print "\n\n\n\n\n" + colorama.Fore.RED
                print "xxxxxxxxxxxxxxxxxxxxxxxxx"
                print "xxxxxxxxxxxxxxxxxxxxxxxxx"
                print "WARNING: moments on cp for link", robot.link(lId_to_lIndex[lId]).getName(), "are not soft finger model"
                print "xxxxxxxxxxxxxxxxxxxxxxxxx"
                print "xxxxxxxxxxxxxxxxxxxxxxxxx"
                print "\n\n\n\n\n" + colorama.Fore.RESET

            """

    for lId in _lIds:
        clist = sim.getContacts(oId, lId)

        if len(clist) > 0:
            print "\n\n\n\n\n" + colorama.Fore.MAGENTA
            print "xxxxxxxxxxxxxxxxxxxxxxxxx"
            print "xxxxxxxxxxxxxxxxxxxxxxxxx"
            print "ERROR: link", robot.link(lId_to_lIndex[lId]).getName(), "is in contact with", obj.getName(), "but is not checked for collision"
            print "xxxxxxxxxxxxxxxxxxxxxxxxx"
            print "xxxxxxxxxxxxxxxxxxxxxxxxx"
            print "\n\n\n\n\n" + colorama.Fore.RESET

    return (cps_avg, wrench_avg)


def launch_test_mvbb_grasps(robotname, object_list, min_vertices = 0, links_to_check = None):
    """Launches a very simple program that spawns a box with dimensions specified in boxes_db.
    """

    world = WorldModel()
    world.loadElement("data/terrains/plane.env")
    robot = make_moving_base_robot(robotname, world)
    xform = resource.get("default_initial_%s.xform" % robotname, description="Initial hand transform",
                         default=se3.identity(), world=world, doedit=False)

    #for box_dims, poses in box_db.db.items():
    #     if world.numRigidObjects() > 0:
    #         world.remove(world.rigidObject(0))
    #     obj = make_box(world,
    #                    box_dims[0],
    #                    box_dims[1],
    #                    box_dims[2])
    #     poses_filtered = []

    for object_name in object_list:
        obj = None
        for object_set, objects_in_set in objects.items():
            if object_name in objects_in_set:
                if world.numRigidObjects() > 0:
                    world.remove(world.rigidObject(0))
                if object_name in objects['princeton']:
                    print "*************Dentro princeton********************" #need to scale the obj size
                    objfilename = 'data/objects/template_obj_scale_princeton.obj'
                    # print"objfilename", objfilename
                    obj = DanyReduceScale(object_name, world,objfilename,object_set)
                    # print obj
                # elif object_name in objects['thingiverse']:
                #     print "****************"
                #     objfilename = 'data/objects/template_obj_scale_thinginverse.obj'
                #     print"objfilename", objfilename
                #     obj = DanyReduceScale(object_name, world,objfilename,object_set)
                else:    
                    obj = make_object(object_set, object_name, world)
        if obj is None:
            print "Could not find object", object_name
            continue





        R,t = obj.getTransform()
        #obj.setTransform(R, [0, 0, box_dims[2]/2.])
        obj.setTransform(R, [0,0,0]) #[0,0,0] or t? #Added by GA
        object_vertices_or_none, tm_decimated = skip_decimate_or_return(obj, min_vertices, 2000)
        if object_vertices_or_none is None:
            print "??????????????????????????????????????????????????"
            print "??????????????????????????????????????????????????"
            print "??????????????????????????????????????????????????"
            print "skipping object, too few vertices", obj.getName()
            print "??????????????????????????????????????????????????"
            print "??????????????????????????????????????????????????"
            print "??????????????????????????????????????????????????"
            continue
        object_or_vertices = object_vertices_or_none
        poses, poses_variations, boxes = compute_poses(object_or_vertices)
        Ry = np.array(se3.homogeneous((so3.from_axis_angle(((0,1,0), 45.*np.pi/180.)),[0,0,0])))
        Rx = np.array(se3.homogeneous((so3.from_axis_angle(((1,0,0), 45.*np.pi/180.)),[0,0,0])))
        Rz = np.array(se3.homogeneous((so3.from_axis_angle(((0,0,1), 45.*np.pi/180.)),[0,0,0])))
        Tx = np.array(se3.homogeneous((so3.identity(), [-.0,0,0])))
        T = Tx.dot(Rz).dot(Rx).dot(Rx) # object is at origin)
        T = Rz;

        poses_new = []

        for pose in poses:
            poses_new.append(pose.dot(T))
        poses = poses_new


        w_T_o = np.array(se3.homogeneous(obj.getTransform()))
        p_T_h = np.array(se3.homogeneous(xform))
        p_T_h[2][3] += 0.02

        for pose in poses:
            w_T_p = w_T_o.dot(pose)
            w_T_h_des_se3 = se3.from_homogeneous(w_T_p.dot(p_T_h))
            if CollisionTestPose(world, robot, obj, w_T_h_des_se3):
                pose_pp = se3.from_homogeneous(pose)
                t_pp = pose_pp[1]
                q_pp = so3.quaternion(pose_pp[0])
                q_pp = [q_pp[1], q_pp[2], q_pp[3], q_pp[0]]
                print "Pose", t_pp + q_pp, "has been filtered since it is in collision for box", box_dims
            elif  w_T_p[2][3] <= 0.:
                print "Pose", t_pp + q_pp, "has been filtered since it penetrates with table"
            else:
                poses_filtered.append(pose)
        print "Filtered", len(poses)-len(poses_filtered), "out of", len(poses), "poses"
        # embed()
        # create a hand emulator from the given robot name
        module = importlib.import_module('plugins.' + robotname)
        # emulator takes the robot index (0), start link index (6), and start driver index (6)
        program = FilteredMVBBTesterVisualizer(box_dims,
                                               poses_filtered,
                                               world,
                                               p_T_h,
                                               module,
                                               box_db,
                                               links_to_check)

        vis.setPlugin(None)
        vis.setPlugin(program)
        program.reshape(800, 600)

        vis.show()
        # this code manually updates the visualization
        while vis.shown():
            time.sleep(0.1)
    return

if __name__ == '__main__':
    print "I'm in grasp_boxes_batch_ga"
    # box_db = None
    # try:
    #     import os.path
    #     filename = os.path.splitext(sys.argv[1])[0]
    # except:
    #     filename = 'box_db'
    # if not os.path.isfile(filename+'.csv'):
    #     print "Error: file", filename, "doesn't exist"
    #     exit()
    all_objects = []
    for dataset in objects.values():
        all_objects += dataset

    to_check = [
    #ycb and acp
    'sterilite_bin_12qt_bottom', #Dont find
    'melissa_doug_play-time_produce_farm_fresh_fruit_unopened_box', #dont find
    #newobjdany
    "juicerB", #dont find
    'bowlB',
    'panA', #dont find
    'containerD', #don't find
    #princeton
    'orangeusa',#error math
    'colorchess', # error math
    'usa', # error math
    'gastruck', #error line 91 i16mc.py
    'stopsignace', #error line 91
    'openchest', #error math
    'yellowcart', #error math
    'whitemonstertruck', #si blocca il pc
    ]
    to_filter = [
    #ycb and acp
    'play_go_rainbow_stakin_cups_2_orange', #zero feasible poses found
    'spam_12oz', #zero poses found
    '1in_metal_washer', #zero feasible poses found
    'stanley_flathead_screwdriver', #zero feasible poses found
    'plastic_nut_grey', #too small
    'small_black_spring_clamp', #end-effector fails
    'campbells_condensed_tomato_soup', #zero feasible poses found
    'blank_hard_plastic_card', #zero feasible poses found
    'plastic_wine_cup', #zero feasible poses found
    'extra_small_black_spring_clamp',#zero feasible poses found
    'orange_wood_block_1inx1in',#zero feasible poses found   
    'play_go_rainbow_stakin_cups_1_yellow',#one poses and objct fall
    'expo_black_dry_erase_marker_fine',#zero feasible poses found
    'expo_black_dry_erase_marker', #zero feasible poses found
    'champion_copper_plus_spark_plug', # zero feasible poses found
    'sharpie_accent_tank_style_highlighters', # zero feasible poses found
    'dove_beauty_bar', #zero feasible poses found
    'one_with_nature_soap_dead_sea_mud', #zero feasible poses found
    'fireplace2', # too few vertices
    'brownchest', #too few vertices
    'brownandyellowchest', #too few vertices
    'openbrownchest', # zero pose found
    'door', #too few vertices
    'browndoor', #too few vertices
    'graydoor', ##too few vertices
    'blackdoor', #too few vertices
    'whitefireplace', #zero poses found
    'book', #too few verticies
    'redbridge', #zero poses found
    'doorwithwindow', #to few vertices
    'blackdoorwithwindow', #error linea 19 in moving_base_control
    '3stair', #too few verticies
    'flipoverpc', #too few verticies
    'pcblue', #too few verticies
    'bigbridge', #zero feasible poses
    'oldgraypc', #zeros feasible poses
    'redumbrella', #too few verticies
    'knight2', #too few verticies
    'buildingposter',  #too few verticies
    'brownboxwithballs', #zeros poses found
    'longformula1', #zeros
    'trianglesignace', # zeros
    'whitebuildingposter', #too
    'stopsignace2', #too
    'whitesignace', #zeros
    'signace', #zeros
    'gravedark', #zeros
    'postbox', #zeros
    'kettle',
    'standingdoublestaircase', #zeros
    'longship', #zeros
    'colouramerica', #zeros
    'yellowsignace', #zeros
    'blackandgreenonewingsboat', #too few verticies
    ]
    done = [
    #ycb and acp
    'pringles_original',
    'dr_browns_bottle_brush',
    'block_of_wood_6in',
    'stanley_philips_screwdriver',
    'elmers_washable_no_run_school_glue',
    'sterilite_bin_12qt_cap',
    'thick_wood_block_6in',  # SI BLOCCA
    'wilson_100_tennis_ball',
    'melissa_doug_farm_fresh_fruit_orange',
    'play_go_rainbow_stakin_cups_8_orange',
    'play_go_rainbow_stakin_cups_box', #il pc si blocca sempre
    'sponge_with_textured_cover',
    'play_go_rainbow_stakin_cups_blue_4',
    'brine_mini_soccer_ball',
    'learning_resources_one-inch_color_cubes_box',
    'soft_scrub_2lb_4oz', 
    'black_and_decker_lithium_drill_driver_unboxed', 
    '1_and_a_half_in_metal_washer', 
    'melissa_doug_farm_fresh_fruit_strawberry',
    'yellow_plastic_chain', 
    'starkist_chunk_light_tuna',
    'melissa_doug_farm_fresh_fruit_plum',
    'play_go_rainbow_stakin_cups_9_red',
    'stainless_steel_spoon_red_handle',
    'comet_lemon_fresh_bleach',
    'stainless_steel_spatula',
    'cheeze-it_388g',
    'moutain_security_steel_shackle',
    'wescott_orange_grey_scissors',
    'purple_wood_block_1inx1in',
    'stainless_steel_fork_red_handle',
    'play_go_rainbow_stakin_cups_10_blue' ,
    'plastic_bolt_grey',
    'play_go_rainbow_stakin_cups_3_red',
    'sharpie_marker',
    'dark_red_foam_block_with_three_holes',
    'melissa_doug_farm_fresh_fruit_lemon',
    'master_chef_ground_coffee_297g',
    'jell-o_chocolate_flavor_pudding',
    'morton_salt_shaker',
    'wilson_golf_ball',
    'white_rope',
    'red_metal_cup_white_speckles',
    'cheerios_14oz',
    'blue_wood_block_1inx1in',
    'jell-o_strawberry_gelatin_dessert',
    'champion_sports_official_softball',
    'clorox_disinfecting_wipes_35',
    'large_black_spring_clamp',
    'melissa_doug_farm_fresh_fruit_apple',
    'frenchs_classic_yellow_mustard_14oz',
    'windex',
    'stainless_steel_knife_red_handle',
    '2in_metal_washer',
    'black_and_decker_lithium_drill_driver',
    'red_metal_bowl_white_speckles',
    'wearever_cooking_pan_with_lid',
    'block_of_wood_12in',
    'medium_black_spring_clamp',
    'red_metal_plate_white_speckles',
    'moutain_security_steel_shackle_key',
    'play_go_rainbow_stakin_cups_7_yellow',
    'rubbermaid_ice_guard_pitcher_blue',
    'melissa_doug_farm_fresh_fruit_pear',
    'domino_sugar_1lb',
    'yellow_wood_block_1inx1in',
    'stanley_13oz_hammer',
    'penn_raquet_ball',
    'melissa_doug_farm_fresh_fruit_peach',
    'play_go_rainbow_stakin_cups_6_purple',
    'morton_pepper_shaker',
    'play_go_rainbow_stakin_cups_5_green',
    'red_wood_block_1inx1in', 
    'melissa_doug_farm_fresh_fruit_banana',
    'first_years_take_and_toss_straw_cups',
    'kong_duck_dog_toy',
    'oreo_mega_stuf',
    'crayola_64_ct',
    'mommys_helper_outlet_plugs',
    'stanley_66_052',
    'feline_greenies_dental_treats',
    'expo_dry_erase_board_eraser',
    'cheezit_big_original',
    'genuine_joe_plastic_stir_sticks',
    'kong_sitting_frog_dog_toy',
    'safety_works_safety_glasses',
    'kong_air_dog_squeakair_tennis_ball',
    'highland_6539_self_stick_notes',
    'mead_index_cards',
    'mark_twain_huckleberry_finn',
    'rollodex_mesh_collection_jumbo_pencil_cup',
    'laugh_out_loud_joke_book',
    'munchkin_white_hot_duck_bath_toy',
    'kygen_squeakin_eggs_plush_puppies',
    'paper_mate_12_count_mirado_black_warrior',
    #newobjdany
    'bowA',
    'containerA',
    'containerC',
    'kitchenUtensilD',
    'kitchenUtensilA',
    'kitchenUtensilB',
    'kitchenUtensilC',
    'kitchenUtensilE',
    'kitchenUtensilF',
    'kitchenUtensilG',
    'mugD',
    'pot',
    'containerB',
    'jug',
    'wc',
    #princeton
    'longwhitebridge',
    'cashmachineblack',
    'antenna',
    'fireplace',
    'arch',
    'cashmachine',
    'cashmachinewhite',
    'cashmachinebig',
    'bigparabola',
    'blackparabola',
    'yellowbook',
    'parabola',
    'whiteswing',
    'sink',
    'squaresink',
    'whitecontainer',
    'chest',
    'whitedoor',
    'whitebigdoor',
    'graybook',
    'ropebridge',
    'chinesbridge',
    'redandgreenbridge',
    'whitebigbridge',
    'doublegraydoor',
    'brownfireplace',
    'blackfireplace',
    'swing',
    'grayswing',
    'redswing',
    'tank',
    'twocilinder',
    'longblackbridge',
    'longpinksatellite',
    'totalblackwithwhitewheeltank',
    'totalwhitetank',
    'blackbook',
    'biggraydoor',
    'bigredstair',
    'blackshoes',
    'bronwstair',
    'glassblackdoor',
    'longblackcontainership',
    'graydooropen',
    'grayshoes',
    'graystripesshoes',
    'standingxwing',
    'lyingblackshoes', #found a different mesh
    'newbalanceshoes',
    'oldpc',
    'pcmac',
    'boatshoes',
    'oldpcwhitkeyboard',
    'flipoverblacktank',
    'colorcar2',
    'flipoverblackship',
    'pinkstandingreceiver',
    'receiver', #il pc si e' bloccato 
    'redandblueumbrella',
    'yellowandbronwumbrella',
    'redandwoodstair',
    'redstair',
    'totalblackubot',
    'smallyellowumbrella',
    'snickers',
    'snowman',
    'snowman2',
    'flipoverwhitemoto',
    'snowmanparty',
    'snowmanwhithat',
    'snowmanwithtrianglehat',
    'stairwhite',
    'standingreceiver',
    'whitebigshoes',
    'whitereceiver',
    'yellowumbrella',
    '2blacksignace',
    'airsignace',
    'america',
    'bishop',
    'blackandstripeshat',
    'blackoldhat',
    'bluemovingballs',
    'blueskate',
    'bigblueskate',
    'flipoveryellowskate',
    'lyingstaircase',
    'captainhat',
    'yellowtrain',
    'notstraightlineoldcarriage',
    'graystair',
    'standingformula1',
    'yellowformula1', # chiede intervento user
    'miniflagcar',
    'redlocomotive',
    'greenskate',
    'blackandwhitebuildingposter',
    'backtruck',
    'glasstruck',
    'lyinggrayformula1',
    'pharmacyblock',
    'manyblock',
    'lyingpawn',
    'knight3',
    'behindbluecar',
    'degreehat',
    'whitechess',
    'flipoverwhitecar',
    'batmancar',
    'locomotive',
    'greendragon',
    'standingbluecar',
    'lyingoldcarriage',
    'greendragonwithwings',
    'yellowdragon',
    'oldcar',
    'train',
    'behindwhitecar',
    'vintagecar',
    'smallcity',
    'blacktruck',
    'pawn3',
    'queen3',
    'car',
    'oldcarriage',
    'notstandingstaircase',
    'rockamerica2',
    'yellowromanhat',
    'smallcity2',
    'behindtruck',
    'whitehat',
    'king',
    'monstertruck',
    'orangeskate',
    'queen',
    'lyingtrain',
    'oldwhitehat',
    'rockcone',
    'redubot',
    'queen2',
    'yellowandgreenlocomotive',
    'personalcomputerblack',
    'pawn4',
    'notstraightlinecar',
    'redsmallship',
    'ship',
    'blueandredbigcar',
    'standingshuttle',
    'redbigcar',
    'pickup',
    'bronwusa',
    'rockamerica',
    'verysmallboat',
    'standingcar',
    'snowmanwhithatandarm',
    'blueandyellowbike',
    'standingwhitetank',
    'smallwhitetank',
    'whitecontainership',
    'flipoverblackxwing',
    'containership',
    'totalblackship',
    'movingballs',
    'historyhat',
    'bluecar',
    'whiteubot',
    'oldcoach',
    'blueandredcar',
    'sideubot',
    'smallcity3',
    'orangeamerica',
    'bronwcowboyhat',
    'pawn2',
    'backoldcarriage',
    'tankblack',
    'highhatwhite',
    'whitelocomotive',
    'blackformula1',
    'graybuildingposter',
    'blackandgreenship',
    'icedragonwithwings',
    'pinkstandingstaircase',
    'vitangeglasscar',
    'orangetruck',
    'boxwithballs',
    'pterosauri',
    'chess',
    'romansailboat',
    'yellowtram',
    'king2',
    'pawn',
    'redxwing',
    'pilothat',
    'longpinkcar',
    'researchship',
    'whiteflagsignace',
    'whoodboat',
    'browndragonwithwings',
    'yellowlocomotive',
    'bluebike',
    'pole',
    'truck',
    'lyingformula1',
    'blackstopsignace',
    'allcolorcar',
    'pinklocomotive',
    'whiteresearchship',
    'tram',
    'yellowskate',
    'backmonstertruck',
    'vikingssailboat',
    'redformula1',
    'formula1',
    'brownchess',
    'manycontainer',
    'sidewhiteubot',
    'knight',
    'behindformula1',
    'longwhiteship',
    'smallcitycolor',
    'militaryhat',
    'pinktriumph',
    'texasyellowhat',
    'blackandwhitetank',
    'behindblueformula1',
    'moto',
    'redmanbike',
    'longwhitecontainership',
    'ship2',
    'messyboat',
    'bluetank',
    'opencar',
    'flipoveropencar',
    'whiteonewingsboat',
    'blueship',
    'longredcontainership',
    'verysmallonewings',
    'onewingsboat',
    'bigship',
    'graygip',
    'blueandgrayTie',
    'grayubot',
    'longbleandredcar',
    'militarygreencar',
    'yellowcar',
    'redbike',
    'militarytruck',
    'xwing',
    'blackcatamaran',
    'redandyellowbike',
    'azureship',
    'carwithflag',
    'purplexwing',
    'whiteandredlongship',
    'flipoverwhitecar2',
    'yacht',
    'longblacksatellite',
    'stanidngwhitebike',
    'pinksatellite',
    'militaryship',
    'whitebike',
    'etruscansailboat',
    'totalblacktank',
    'blacktank',
    'sailboat',
    'brownsmallship',
    'whitegip',
    'whitetank',
    'orangewingboat',
    'whitetank2',
    'smallboat',
    'yellowtank',
    'behindubot',
    'redmoto',
    'harley',
    'blacktie',
    'flipovergreentank',
    'motogp',
    'whitebigsailboat',
    'greentank',
    'bigsailboat',


    ]

    for obj_name in to_filter +  done + to_check:
        all_objects.pop(all_objects.index(obj_name))

    print "-------------"
    print all_objects
    print "-------------"

    # try:
    objname = sys.argv[1]

    # palm, index (proximal, middle, distal), little, middle, ring, thumb
    links_to_check = np.array(soft_hand.links_to_check) + 6
    # box_db = MVBBLoader(filename, soft_hand.numJoints, len(links_to_check))
    launch_test_mvbb_grasps("soft_hand", [objname], 100, links_to_check)
