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
        super().__init__('robot_sim')
        self.xml_path = os.path.join('/home/luka/projekti/robotic_manipulation_ws/src/industrial_robot', 'model', 'industrial_robot.xml')
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.paused = False
        self.show_marker = False
        self.x_goal = None
        self.y_goal = None
        self.z_goal = None
        self.solution = 1
        self.kp = 100
        self.kd = 200
        mujoco.mj_forward(self.model, self.data)


    def key_callback(self, keycode):
            if chr(keycode) == ' ':
                self.paused = not self.paused
            elif chr(keycode) == '6':
                self.show_marker = not self.show_marker
            elif chr(keycode) == '7':
                self.solution = 1
            elif chr(keycode) == '8':
                self.solution = 2
            elif chr(keycode) == '9':
                self.solution = 3
            elif chr(keycode) == '-':
                self.solution = 4

    
    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
            viewer.cam.lookat[:] = [0, 0, 0]
            viewer.cam.distance = 20
            viewer.cam.elevation = -45
            viewer.cam.azimuth = -45


            start = time.time()

            self.data.qpos[5] = 1.57/2.0
            self.data.qpos[4] = 1.57/4.0
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)


            while viewer.is_running() and time.time() - start < 99999:
                step_start = time.time()
    
                ### MARKER ###

                viewer.user_scn.ngeom = 0

                if self.show_marker and self.x_goal is not None and self.y_goal is not None and self.z_goal is not None:
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[0],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.05, 0, 0],
                        pos=np.array([self.x_goal, self.y_goal, self.z_goal]),
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
        control = self.kp*(x_ref - self.data.qpos) - self.kd*self.data.qvel
        return control

def main():
    rclpy.init()

    node = SimulateNode()

    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()