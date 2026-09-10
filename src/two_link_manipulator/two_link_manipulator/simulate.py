#!/usr/bin/env python
import os

os.environ.pop('WAYLAND_DISPLAY', None)
os.environ.pop('XDG_SESSION_TYPE', None)
os.environ['GLFW_PLATFORM'] = 'x11'

import rclpy    
import mujoco
import mujoco.viewer
import time
import threading
import numpy as np
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from custom_interfaces.srv import Position2D

class SimulateNode(Node):
    def __init__(self):
        super().__init__('simulate_node')
        self.xml_path = os.path.join('/home/luka/projekti/robotic_manipulation_ws/src/two_link_manipulator', 'models', 'simple_two_link_manip', 'two_link_manipulator.xml')
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.paused = False
        self.show_marker = False
        mujoco.mj_forward(self.model, self.data)

        self.srv = self.create_service(Position2D, 'set_goal', self.set_goal_callback)

        self.kp = 100
        self.kd = 200
        self.x_goal = None
        self.y_goal = None

    def key_callback(self, keycode):
            if chr(keycode) == ' ':
                self.paused = not self.paused
            elif chr(keycode) == '6':
                self.show_marker = not self.show_marker;

    def set_goal_callback(self, request, response):
        self.x_goal = request.x
        self.y_goal = request.y
        return response

    def inverse_kinematics(self, x, y):
        q2_1 = np.arccos((x**2 + y**2)/2 - 1)
        q1_1 = np.arctan2(y * (1 + np.cos(q2_1)) - x * np.sin(q2_1), x * (1 + np.cos(q2_1)) + y * np.sin(q2_1)) 

        q2_2 = -np.arccos((x**2 + y**2)/2 - 1)
        q1_2 = np.arctan2(y * (1 + np.cos(q2_2)) - x * np.sin(q2_2), x * (1 + np.cos(q2_2)) + y * np.sin(q2_2)) 

        return [q1_1, q2_1, q1_2, q2_2]
    
    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
            viewer.cam.lookat[:] = [0, 0, 0]
            viewer.cam.distance = 10
            viewer.cam.elevation = -90
            viewer.cam.azimuth = 90


            start = time.time()


            while viewer.is_running() and time.time() - start < 99999:
                step_start = time.time()
    
                if self.x_goal is not None and self.y_goal is not None:
                    q1, q2, _, _ = self.inverse_kinematics(self.x_goal, self.y_goal)
                    self.data.ctrl = self.pd_control([q1, q2]) 
                else:
                    self.data.ctrl = 0

                ### MARKER ###

                viewer.user_scn.ngeom = 0

                if self.show_marker and self.x_goal is not None and self.y_goal is not None:
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[0],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.05, 0, 0],
                        pos=np.array([self.x_goal, self.y_goal, 0.0]),
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0.5, 0.5, 0, 1], dtype=np.float32)
                    )
                    viewer.user_scn.ngeom = 1

                ##############
                
                if not self.paused:
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
    
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def pd_control(self, x_ref):
        control = self.kp*(x_ref - self.data.qpos) -self.kd*self.data.qvel
        return control

def main():
    rclpy.init()

    node = SimulateNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    spin_thread.start()

    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)



if __name__ == '__main__':
    main()