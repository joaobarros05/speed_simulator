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


import glob
import os
import sys

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
import random


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
REGION_SPEED = [(420,94), (1500,94),(1680,316),(200,316)]

#For video 
def inicializate_video(name_video):
    name = name_video+'.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(name, fourcc, 30.0, (1920,1080))
    return out

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


def in_region(point):
    region = Polygon(REGION_SPEED)
    
    point = Point(point[0], point[1])
    
    return region.contains(point)

def save_data(output):
    f = open(output+'.pckl', 'wb')
    pickle.dump(VEHICLES_DATA_LIST, f)
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

# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================


class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(display, vehicles, camera, current_frame):
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
                
                
                if in_region(points[2]):
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
        
        display.blit(text,(704,318))    

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
        
        # t0 = font.render('p_0', False, (0, 0, 0))
        # t1 = font.render('p_1', False, (0, 0, 0))
        # t2 = font.render('p_2', False, (0, 0, 0))
        # t3 = font.render('p_3', False, (0, 0, 0))
        # t4 = font.render('p_4', False, (0, 0, 0))
        # t5 = font.render('p_5', False, (0, 0, 0))
        # t6 = font.render('p_6', False, (0, 0, 0))
        # t7 = font.render('p_7', False, (0, 0, 0))
        
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
        
        # display.blit(t0,points[0])
        # display.blit(t1,points[1]) 
        # display.blit(t2,points[2]) 
        # display.blit(t3,points[3])
        # display.blit(t4,points[4])
        # display.blit(t5,points[5])
        # display.blit(t6,points[6])
        # display.blit(t7,points[7])
        

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        #print(bb_cords)
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
        self.camera = None
        self.vehicles_list = []
        self.tm = None

        self.display = None
        self.image = None
        self.capture = True
        self.synchronous_master = True
        self.show = args.show
        self.out = inicializate_video(args.video)

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
        number_of_vehicles = 300 #300
        
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
        
        bp_lists = ['vehicle.audi.a2', 'vehicle.audi.tt', 
                    'vehicle.mercedes-benz.coupe', 
                    'vehicle.carlamotors.carlacola',
                    'vehicle.citroen.c3', 'vehicle.mini.cooperst', 
                    'vehicle.bmw.isetta', 'vehicle.nissan.micra', 
                    'vehicle.nissan.patrol', 'vehicle.mustang.mustang',
                    'vehicle.lincoln.mkz2017', 'vehicle.tesla.cybertruck', 
                    'vehicle.toyota.prius', 'vehicle.volkswagen.t2',
                    'vehicle.seat.leon', 'vehicle.audi.etron', 
                    'vehicle.bmw.grandtourer', 'vehicle.tesla.model3', 
                    'vehicle.dodge_charger.police',
                    'vehicle.kawasaki.ninja', 'vehicle.jeep.wrangler_rubicon',
                    'vehicle.yamaha.yzf', 'vehicle.chevrolet.impala',
                    'vehicle.harley-davidson.low_rider']
        
        for bp_id in bp_lists:
            blueprints.append(self.world.get_blueprint_library().filter(bp_id)[0])
        
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
        #cont_v = 0
        tm_port = tm.get_port()
        for v in self.vehicles_list:
            v.set_autopilot(True, tm_port)    
            tm.auto_lane_change(v,False)
            
            #if cont_v <= 50:
            tm.vehicle_percentage_speed_difference(v,-120)
                #cont_v+=1

        tm.global_percentage_speed_difference(2.0)

        
    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        #Posição da câmera
        #Town05 = carla.Location(x=27, y=165.0, z=5)
        #camera_transform = carla.Transform(carla.Location(x=4, y=100, z=5.5), carla.Rotation(pitch=-45,  yaw=270, roll=0.000000)) #pattern #pitch=-45, yaw=0, roll=0.000000
        camera_transform = carla.Transform(carla.Location(x=4, y=100, z=5.5), carla.Rotation(pitch=0,  yaw=270, roll=0.000000))
        self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration

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
    

    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            
            if self.show:
                surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
                display.blit(surface, (0, 0))
            
            self.out.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
            #cv2.imwrite("teste.jpg",cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
            
    def destroy_vehicles(self):
         print('Destroying %d vehicles.\n' % len(self.vehicles_list))
         self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
         
         
    def save_video(self):
        self.out.release()

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
            self.world = self.client.load_world('Town03')
            change_weather(self.world, args.weather)
            
            self.tm = self.client.get_trafficmanager(args.tm)
            self.frame_number = 0
    
            self.setup_car(args.tm)
            self.setup_camera()
    
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
                
                self.capture = True
                #pygame_clock.tick_busy_loop(20)
                if self.show:
                    pygame_clock.tick_busy_loop(30)
                self.render(self.display)
                    
                    #current_time = (pygame.time.get_ticks()/(1000*60))%60
                    
                    #print('time: ', current_time)
                    
                bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(self.display, self.vehicles_list, self.camera, self.frame_number)
               
                #Desenha bounding box e região de estimação
                #if self.show:
                #    ClientSideBoundingBoxes.draw_bounding_boxes(self.display, bounding_boxes)
                #    ClientSideBoundingBoxes.draw_region_speed(self.display, REGION_SPEED)
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
            self.camera.destroy()
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
        save_data(args.video)
        print('EXIT')


if __name__ == '__main__':
    main()
