#!/usr/bin/env python

# Copyright (c) 2019 Aptiv
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
An example of client-side bounding boxes with basic car controls.

Controls:

    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake

    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

#

import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
import cv2

import weakref
import random
import logging
import math
import pickle
import argparse

import re

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = 1920#//2
VIEW_HEIGHT = 1080#//2
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)
REGION_COLOR = (29,97,193)

#FIXME
VEHICLES_DATA_LIST = []
vehicles_id = []

#FIXME - TESTES
REGION_SPEED = [(200,94),(1550,94),(1700,316),(60,316)]
SPEED_ESTIMATION_REGION_3D = (
    [3.96895000e+02, -1.30008548e+02,  -2.26336597e-02], 
    [4.17000000e+02, -1.30008548e+02, -2.26336597e-02], 
    [4.16500000e+02, -1.24107853e+02, -2.26330453e-02], 
    [3.98710000e+02, -1.24107853e+02,-2.26330453e-02])

#For video 
def inicializate_video(name_video):
    #Center View Point
    name = name_video+'_center.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out_center = cv2.VideoWriter(name, fourcc, 30.0, (1920,1080))
    
    #Right View Point
    name = name_video+'_right.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out_right = cv2.VideoWriter(name, fourcc, 30.0, (1920,1080))
    
    #Left View Point
    name = name_video+'_left.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out_left = cv2.VideoWriter(name, fourcc, 30.0, (1920,1080))
    
    return out_center, out_right, out_left

#REGION_SPEED = [(755,125), (1301,125),(704,308),(1407,308)] #Town05

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def change_weather(world, weather_index):
    #weather_index = 0
    #self._weather_index += -1 if reverse else 1
    #self._weather_index %= len(self._weather_presets)
    weather_presets = find_weather_presets()
    #print('tempo - ', weather_presets)
    preset = weather_presets[weather_index]
    world.set_weather(preset[0])


##NOVO - FAZ A VERICAÇÃO DO PONTO NO MUNDO REAL
def in_region(point):
    #FIXME - Corrigir atribuição
    
    #print('point - ', point)
    region_speed = np.asarray(SPEED_ESTIMATION_REGION_3D)[:,[0,1]]
    region = Polygon(region_speed)
    
    point = [point[0], point[1]]
    point = Point(point)
    
    return region.contains(point)

def save_data(client, output):
    f = open(output+'.pckl', 'wb')
    pickle.dump(VEHICLES_DATA_LIST, f)
    f.close()
    
    ##Save Calibration Parameters
    camera_calibration = {}
    
    ##Center
    camera_calibration['center_e'] = client.camera_c.E
    camera_calibration['center_k'] = client.camera_c.K
    
    ##Right
    camera_calibration['right_e'] = client.camera_r.E
    camera_calibration['right_k'] = client.camera_r.K
    
    ##Left
    camera_calibration['left_e'] = client.camera_l.E
    camera_calibration['left_k'] = client.camera_l.K
    
    ##Save Calibration Parameters - FIXME
    f = open(output+'_camera_calibration.pckl', 'wb')
    pickle.dump(camera_calibration, f)
    f.close()
    

    
def time_to_fps(time_seconds):
    return time_seconds * 30
    

class speed_data(object):
    def __init__(self):
        self.speeds = []
        self.positions = []
        self.frame_init = 0
        self.frame_end = 0
        
    def add_speed(self, value):
        self.speeds.append(value)
        
    def add_position(self, points):
        self.positions.append(points)
        
    def add_frame_init(self, frame_init):
        self.frame_init = frame_init
        
    def add_frame_end(self, frame_end):
        self.frame_end = frame_end
        
    def get_average_speed(self):
        return np.average(self.speeds)
        

class vehicle_data(object):
    def __init__(self, vehicle_id, current_speed, current_point, current_frame):
        self.vehicle_id = vehicle_id
        self.exited = True
        self.data_speeds = []
        self._added_new_instance = False

        self.instance_speed = speed_data()
        self.instance_speed.add_frame_init(current_frame)
        self.instance_speed.add_speed(current_speed)
        self.instance_speed.add_position(current_point)
        
        
    def create_new_instace_speed(self, current_speed, current_point, current_frame):
        self.instance_speed = speed_data()
        self.instance_speed.add_frame_init(current_frame)
        self.instance_speed.add_speed(current_speed)
        self.instance_speed.add_position(current_point)
        self.exited = True

    def add_current_instace_speed(self, current_speed, current_point):
        self.instance_speed.add_speed(current_speed)
        self.instance_speed.add_position(current_point)
        
    def add_instance_frame_init(self, current_frame):
        self.instance_speed.add_frame_init(current_frame)
        
    def add_instance_frame_end(self, current_frame):
        self.instance_speed.add_frame_end(current_frame)

    def get_id(self):
        return self.vehicle_id  

    def get_average_speed(self):
        return self.data_speeds[len(self.data_speeds)-1].get_average_speed()
   
    def end(self, current_frame, current_point):
        self.data_speeds.append(self.instance_speed)
        self.instance_speed.add_position(current_point)
        self.instance_speed.add_frame_end(current_frame)
        self._added_new_instance = False
        self.exited = False
        
        
    #def acabou_sair(self, value):
    #    self.acabou_sair = value
    
    ##CAMERA CALIBRATION | Novo
    def return_3dPoint(point, calibration_camera):
        """
        Returns the 3D point given the 2D point of the image
        """
        
        return np.dot(calibration_camera,point)

# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================

#FIXME - TESTES
#coords_3d = []

class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """
    

    @staticmethod
    def get_bounding_boxes(display, vehicles, camera, current_frame, region_speed):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """
        
        #FIXME
        pygame.font.init()
        myfont = pygame.font.SysFont('Comic Sans MS', 30)
        

        bounding_boxes = []
        for vehicle in vehicles:
            
            if vehicle.is_alive:     
                bbox = ClientSideBoundingBoxes.get_bounding_box(vehicle, camera)
                bounding_boxes.append(bbox)
                bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
                
                points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
                
                bbox_3d = ClientSideBoundingBoxes.get_3Dbounding_box(vehicle)
                points_3d = [(int(bbox_3d[i, 0]), int(bbox_3d[i, 1])) for i in range(8)]
                
               

                if in_region(points_3d[2]): 
                    
                     #FIXME - NOVO
                    bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
                    world_cord = ClientSideBoundingBoxes._vehicle_to_world(bb_cords, vehicle)
                    
                    #FIXME - TESTES
                    #coords_3d.append(world_cord)
                    #f = open('points_3D.pckl', 'wb')
                    #pickle.dump(coords_3d, f)
                    #f.close()
                    #print('[SAVING>>>]')
                    

                    t = vehicle.get_transform()
                    v = vehicle.get_velocity()
                    c = vehicle.get_control() 
                    
                    speed = (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
                    
                    #print('TIPO: ', vehicle.type_id)
                    
                    
                    if len([v for v in VEHICLES_DATA_LIST if v.get_id() == vehicle.id]) == 0:
                        #vehicles_id.append(vehicle.id)
                        vh = vehicle_data(vehicle.id, speed, points, current_frame)
    
                        #vh.add_current_instace_speed(speed, points)
                        vh._added_new_instance = True
                        #vh.add_bboxes(points)
                        VEHICLES_DATA_LIST.append(vh)
                        
                        print('=============// START 1 //==============')
                                    
                    else:
                        #FIXME
                        for i in range(0, len(VEHICLES_DATA_LIST)):
                            if VEHICLES_DATA_LIST[i].get_id() == vehicle.id:
                                if not VEHICLES_DATA_LIST[i]._added_new_instance:
                                     VEHICLES_DATA_LIST[i].create_new_instace_speed(speed, points, current_frame)
                                     VEHICLES_DATA_LIST[i]._added_new_instance = True
                                     print('=============// START 2 //==============')
                                else:                                 
                                     VEHICLES_DATA_LIST[i].add_current_instace_speed(speed, points)
    
                else:
                    for i in range(0, len(VEHICLES_DATA_LIST)):
                        if VEHICLES_DATA_LIST[i].get_id() == vehicle.id and VEHICLES_DATA_LIST[i].exited:
                            VEHICLES_DATA_LIST[i].end(current_frame, points)
                            
                            #text = myfont.render('Speed: '+ str(VEHICLES_DATA_LIST[i].get_average_speed())+' km/h', False, (0, 0, 0))
                            print('=============// END//==============')
                            print('Average speed - ' + str(VEHICLES_DATA_LIST[i].get_average_speed()) +' | ID: '+str(VEHICLES_DATA_LIST[i].get_id()))
                            #display.blit(text, (1567,320))
                                
        return bounding_boxes
    
    @staticmethod
    def draw_region_speed(display, points):
        """
        Draws bounding boxes on pygame display.
        """
        pygame.font.init()
        myfont = pygame.font.SysFont('Comic Sans MS', 30)
        text = myfont.render('Speed measurement region', False, (0, 0, 0))
        
        #FIXME        
        text_point_x = points[0][0] 
        text_point_y = points[0][1]+120
        
        text_point = [text_point_x, text_point_y]
        
        display.blit(text,(text_point))    

        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1], 2)
        pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3], 2)
        display.blit(bb_surface, (0, 0))
        

    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes):
        """
        Draws bounding boxes on pygame display.
        """

        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        
        
        font = pygame.font.SysFont('Comic Sans MS', 30)
        
        t0 = font.render('p_0', False, (0, 0, 0))
        t1 = font.render('p_1', False, (0, 0, 0))
        t2 = font.render('p_2', False, (0, 0, 0))
        t3 = font.render('p_3', False, (0, 0, 0))
        t4 = font.render('p_4', False, (0, 0, 0))
        t5 = font.render('p_5', False, (0, 0, 0))
        t6 = font.render('p_6', False, (0, 0, 0))
        t7 = font.render('p_7', False, (0, 0, 0))
        
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # draw lines
            # base
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
            # top
            pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
            pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
            # base-top
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])
        display.blit(bb_surface, (0, 0))
        
        display.blit(t0,points[0])
        display.blit(t1,points[1]) 
        display.blit(t2,points[2]) 
        display.blit(t3,points[3])
        display.blit(t4,points[4])
        display.blit(t5,points[5])
        display.blit(t6,points[6])
        display.blit(t7,points[7])
        

    @staticmethod
    def get_3Dbounding_box(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        world_cord = ClientSideBoundingBoxes._vehicle_to_world(bb_cords, vehicle)

        return np.transpose(world_cord[:3, :])


    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.K, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)

        return camera_bbox

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """
    
        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self, args):
        self.client = None
        self.world = None
        self.camera_c = None
        self.camera_r = None
        self.vehicles_list = []
        self.tm = None

        self.display = None
        self.image_c = None
        self.image_r = None
        self.image_l = None
        self.capture_c = True
        self.capture_r = True
        self.capture_l = True
        self.synchronous_master = True
        self.show = args.show
        self.out_center, self.out_right, self.out_left = inicializate_video(args.video)

    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        #self.world.apply_settings(settings)
        
        #synchronous_master = True
        #settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def setup_car(self, tm_port):
        """
        Spawns actor-vehicle to be controled.
        """

        # car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        # location = random.choice(self.world.get_map().get_spawn_points())
        # self.car = self.world.spawn_actor(car_bp, location)
        
        #self.vehicles_list = []
        #self.world = world
        #self.client = client
        
        #novo
        number_of_vehicles = 373 #373
        
        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        
        if number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, number_of_vehicles, number_of_spawn_points)
            #args.number_of_vehicles = number_of_spawn_points
            
        # @todo cannot import these directly.
        #SpawnActor = carla.command.SpawnActor
        
        blueprints = []
        #'vehicle.mercedes-benz.coupe'
        models = ['dodge', 'audi', 'model3', 'mini', 'mustang', 
            'lincoln', 'prius', 'nissan', 'crown', 'impala', 'kawasaki', 
            'bmw', 'gazelle', 'diamondback', 'tesla', 'mitsubishi', 'volkswagen', 
            'carlamotors', 'ford', 'seat', 'mercedes', 'vespa', 'toyota', 'harley-davidson',
            'jeep', 'micro', 'chevrolet', 'citroen']
        
        #print("Vehicle = ", self.world.get_blueprint_library().filter('*vehicle*'))
        for vehicle in self.world.get_blueprint_library().filter('*vehicle*'):
            if any(model in vehicle.id for model in models):
                blueprints.append(vehicle)
        
        #blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        #print('bps - ', blueprints)
        

        #novo
        tm = self.client.get_trafficmanager(tm_port)
        tm.set_global_distance_to_leading_vehicle(2.0)
        
        # @todo cannot import these directly.
        #SpawnActor = carla.command.SpawnActor
        #SetAutopilot = carla.command.SetAutopilot
        #SetVehicleLightState = carla.command.SetVehicleLightState
        #FutureActor = carla.command.FutureActor
        

        # --------------
        # Spawn vehicles
        # --------------
        #batch = []

        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break
            blueprint = random.choice(blueprints)

            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            self.vehicles_list.append(vehicle)

            #light_state = vls.NONE
            
            ##novo
           # batch.append(SpawnActor(blueprint, transform)
           #     .then(SetAutopilot(FutureActor, True, tm.get_port()))
           #     .then(SetVehicleLightState(FutureActor, light_state)))
            
        #self.vehicles_list = self.world.get_actors().filter('vehicle.*')
        tm_port = tm.get_port()
        for v in self.vehicles_list:
            v.set_autopilot(True, tm_port)    
            tm.auto_lane_change(v,False)
            
        tm.global_percentage_speed_difference(20.0)

        
    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        #Posição de câmera
        #Town05 = carla.Location(x=27, y=165.0, z=5) 
        camera_transform_c = carla.Transform(carla.Location(x=410.4, y=-115.0, z=7.5), carla.Rotation(pitch=-36,  yaw=-90, roll=0.000000)) #pattern
        camera_transform_r = carla.Transform(carla.Location(x=413.4, y=-110.0, z=7.5), carla.Rotation(pitch=-36,  yaw=-106, roll=0.000000)) #pattern 2
        camera_transform_l = carla.Transform(carla.Location(x=400.4, y=-115.0, z=7.5), carla.Rotation(pitch=-36,  yaw=-48, roll=0.000000))

        #camera_transform = carla.Transform(carla.Location(x=408.4, y=-300.0, z=16.5), carla.Rotation(pitch=-30,  yaw=-170, roll=0.000000))
        
        #Center Camera
        self.camera_c = self.world.spawn_actor(self.camera_blueprint(), camera_transform_c)
        weak_self = weakref.ref(self) #FIXME - Serve para todas as câmeras
        
        self.camera_c.listen(lambda image_c: weak_self().set_image_c(weak_self, image_c))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera_c.K = calibration
        
        transformation_matrix = ClientSideBoundingBoxes.get_matrix(camera_transform_c)
        self.camera_c.E = transformation_matrix
        
        #Right Camera
        self.camera_r = self.world.spawn_actor(self.camera_blueprint(), camera_transform_r)
        #weak_self_r = weakref.ref(self)
        self.camera_r.listen(lambda image_r: weak_self().set_image_r(weak_self, image_r))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera_r.K = calibration
        
        transformation_matrix = ClientSideBoundingBoxes.get_matrix(camera_transform_r)
        self.camera_r.E = transformation_matrix
        
        #Left Camera
        self.camera_l = self.world.spawn_actor(self.camera_blueprint(), camera_transform_l)
        #weak_self_l = weakref.ref(self)
        self.camera_l.listen(lambda image_l: weak_self().set_image_l(weak_self, image_l))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera_l.K = calibration
        
        transformation_matrix = ClientSideBoundingBoxes.get_matrix(camera_transform_l)
        self.camera_l.E = transformation_matrix
        
    #FIXME - TESTES 
    def convert3DRegionsSpeed_to_2d(self, region_3d, calibration_extr, calibration_intr):
        ponto = np.transpose(np.mat(region_3d))
        
        novo_ponto = np.matrix(np.ones((4,4)))
        novo_ponto[0,0] = ponto[0,0]
        novo_ponto[0,1] = ponto[0,1]
        novo_ponto[0,2] = ponto[0,2]
        novo_ponto[0,3] = ponto[0,3]
        
        novo_ponto[1,0] = ponto[1,0]
        novo_ponto[1,1] = ponto[1,1]
        novo_ponto[1,2] = ponto[1,2]
        novo_ponto[1,3] = ponto[1,3]
        
        novo_ponto[2,0] = ponto[2,0]
        novo_ponto[2,1] = ponto[2,1]
        novo_ponto[2,2] = ponto[2,2]
        novo_ponto[2,3] = ponto[2,3]

        world_sensor_matrix = np.linalg.inv(calibration_extr)
        sensor_cords = np.dot(world_sensor_matrix, novo_ponto)
        
        cords_y_minus_z_x = np.concatenate([sensor_cords[1, :], -sensor_cords[2, :], sensor_cords[0, :]])
        polygon = np.transpose(np.dot(calibration_intr, cords_y_minus_z_x))
        camera_polygon = np.concatenate([polygon[:, 0] / polygon[:, 2], polygon[:, 1] / polygon[:, 2], polygon[:, 2]], axis=1)
        
        array_polygon = (np.asarray(camera_polygon))
        array_polygon = array_polygon[:,[0,1]]
        array_polygon =  [(array_polygon[0,0],array_polygon[0,1]), 
                        (array_polygon[1,0],array_polygon[1,1]),
                        (array_polygon[2,0],array_polygon[2,1]),
                        (array_polygon[3,0],array_polygon[3,1])]
        
        return array_polygon

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """

        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False
    
    '''
    @staticmethod
    def set_image(weak_self, img, camera_position):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture_c:
            
            if camera_position == 'center':
                self.image_c = img
                self.capture_c = False
            elif camera_position == 'right':
                self.image_r = img
                self.capture_r = False
            elif camera_position == 'left':
                self.image_l = img
                self.capture_l = False
      '''        
                
    @staticmethod
    def set_image_c(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """
    
        self = weak_self()
        if self.capture_c:
            self.image_c = img
            self.capture_c = False
            
            
    @staticmethod
    def set_image_r(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """
    
        self = weak_self()
        if self.capture_r:
            self.image_r = img
            self.capture_r = False
            
            
    @staticmethod
    def set_image_l(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """
    
        self = weak_self()
        if self.capture_l:
            self.image_l = img
            self.capture_l = False
          
                

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """
        #Image Center
        if self.image_c is not None:
            array = np.frombuffer(self.image_c.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image_c.height, self.image_c.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            
            if self.show:
                surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
                display.blit(surface, (0, 0))
            
            self.out_center.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
            
        if self.image_r is not None:
           array = np.frombuffer(self.image_r.raw_data, dtype=np.dtype("uint8"))
           array = np.reshape(array, (self.image_r.height, self.image_r.width, 4))
           array = array[:, :, :3]
           array = array[:, :, ::-1]

           self.out_right.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
           #cv2.imwrite("teste.jpg",cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
           
        if self.image_l is not None:
           array = np.frombuffer(self.image_l.raw_data, dtype=np.dtype("uint8"))
           array = np.reshape(array, (self.image_l.height, self.image_l.width, 4))
           array = array[:, :, :3]
           array = array[:, :, ::-1]

           self.out_left.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
           
    
            
    def destroy_vehicles(self):
         print('Destroying %d vehicles.\n' % len(self.vehicles_list))
         self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
         
         
    def save_video(self):
        self.out_center.release()
        self.out_right.release()
        self.out_left.release()

    def game_loop(self, args):
        """
        Main program loop.
        """

        try:
            if self.show:
                pygame.init()
    
            self.client = carla.Client(args.host, args.port) #2000
            #self.client = carla.Client('10.131.8.45', 2000)
            self.client.set_timeout(10.0)
            self.world = self.client.load_world('Town04')
            change_weather(self.world, args.weather)
            
            self.tm = self.client.get_trafficmanager(args.tm)
            self.frame_number = 0
    
            self.setup_car(args.tm)
            self.setup_camera()
            
            #FIXME - NOVO
            self.region_speed = self.convert3DRegionsSpeed_to_2d(SPEED_ESTIMATION_REGION_3D, self.camera_c.E, self.camera_c.K)
    
            if self.show:
                self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
                pygame_clock = pygame.time.Clock()
    
            self.set_synchronous_mode(True)
            print('Number of vehicles - ', len(self.vehicles_list))
            #vehicles = self.world.get_actors().filter('vehicle.*')
            #points = [(755,125), (1301,125),(704,308),(1407,308)] #Town05
            #points = [(496,94), (1348,94),(1518,316),(297,316)] #Town04
            current_time = 0
            
            total_fps = time_to_fps(args.time) #108000 Qtd de fps necessária para 1 hrs de video
            
    
            while self.frame_number < total_fps:
                self.world.tick()
                
                self.capture_c = True
                self.capture_r = True
                self.capture_l = True
                #pygame_clock.tick_busy_loop(20)
                if self.show:
                    pygame_clock.tick_busy_loop(30)
                self.render(self.display)

                    #current_time = (pygame.time.get_ticks()/(1000*60))%60
                    
                    #print('time: ', current_time)
                    
                bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(self.display, self.vehicles_list, self.camera_c, self.frame_number, self.region_speed)
                
                #show bounding box e region speed
                if self.show:
                #    ClientSideBoundingBoxes.draw_bounding_boxes(self.display, bounding_boxes)
                    ClientSideBoundingBoxes.draw_region_speed(self.display, self.region_speed)
                    #ClientSideBoundingBoxes.show_points_bbox(self.display, bounding_boxes)
    
                if self.show:
                    pygame.display.flip()
                    pygame.event.pump()
                
                self.frame_number+=1
                print('frame_number - ', self.frame_number)
                
                #if self.control(self.car):
                #    return

        finally:
            self.set_synchronous_mode(False)
            self.camera_c.destroy()
            self.camera_r.destroy()
            self.camera_l.destroy()
            self.destroy_vehicles()
            self.save_video()

        #self.car.destroy()
        if self.show:
            pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """
    argparser = argparse.ArgumentParser(
        description='CARLA Control Simulator - Papa')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--tm',
        metavar='TM',
        default=8000,
        type=int,
        help='TM port to listen to (default: 8000)')
    argparser.add_argument(
        '-v', '--video',
        metavar='V',
        default='video',
        help='name of video to save')
    argparser.add_argument(
        '--time',
        metavar='TIME',
        default='1200',
        type=int,
        help='time in seconds to generate simulator. Pattern: 20 minutes')
    argparser.add_argument(
        '--weather',
        metavar='TIME',
        default='0',
        type=int,
        help='set index weather')
    argparser.add_argument(
        '--show',
        action='store_true',
        help='show image video')
    args = argparser.parse_args()

    try:
        print('-- ', args.video)
        client = BasicSynchronousClient(args)
        client.game_loop(args)
    finally:
        save_data(client, args.video)
        print('EXIT')


if __name__ == '__main__':
    main()
