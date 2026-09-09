#!/usr/bin/env python
import os

os.environ.pop('WAYLAND_DISPLAY', None)
os.environ.pop('XDG_SESSION_TYPE', None)
os.environ['GLFW_PLATFORM'] = 'x11'

import rclpy    
import mujoco
import mujoco.viewer
import time
import numpy as np
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

class SimulateNode(Node):
    def __init__(self):
        super().__init__('simulate_node')
        self.xml_path = os.path.join('/home/luka/projekti/robotic_manipulation_ws/src/two_link_manipulator', 'models', 'simple_two_link_manip', 'two_link_manipulator.xml')
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.paused = False
        self.show_marker = False
        mujoco.mj_forward(self.model, self.data)

        self.srv = self.create_service()

        self.kp = 100
        self.kd = 500

    def key_callback(self, keycode):
            if chr(keycode) == ' ':
                self.paused = not self.paused
            elif chr(keycode) == '6':
                self.show_marker = not self.show_marker;
    
    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
            viewer.cam.lookat[:] = [0, 0, 0]
            viewer.cam.distance = 10
            viewer.cam.elevation = -90
            viewer.cam.azimuth = 90


            start = time.time()


            while viewer.is_running() and time.time() - start < 99999:
                step_start = time.time()
    
                self.data.ctrl = self.pd_control([np.pi/2, 0]) 

                ### MARKER ###

                viewer.user_scn.ngeom = 0

                if self.show_marker:
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[0],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.05, 0, 0],
                        pos=np.array([1.0, 1.0, 0.0]),
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
    node.run()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()