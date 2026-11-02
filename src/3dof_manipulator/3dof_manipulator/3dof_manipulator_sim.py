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
        self.xml_path = os.path.join('/home/luka/projekti/robotic_manipulation_ws/src/3dof_manipulator', 'model', '3dof_manipulator.xml')
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

    def inverse_kinematics(self, p):
        L2 = 1.0
        L3 = 1.0
        x, y, z = p
        q1 = np.atan2(y, x)
        q1_ = q1 + np.pi
        q3_1 = np.arccos(((x*np.cos(q1) + y*np.sin(q1))**2 + z**2 - L3**2 - L2**2)/(2*L2*L3))
        q3_2 = np.arccos(((x*np.cos(q1_) + y*np.sin(q1_))**2 + z**2 - L3**2 - L2**2)/(2*L2*L3))
        q3_3 = -np.arccos(((x*np.cos(q1) + y*np.sin(q1))**2 + z**2 - L3**2 - L2**2)/(2*L2*L3))
        q3_4 = -np.arccos(((x*np.cos(q1_) + y*np.sin(q1_))**2 + z**2 - L3**2 - L2**2)/(2*L2*L3))

        k1_1 = L3*np.cos(q3_1) + L2
        k2_1 = L3*np.sin(q3_1)

        k1_2 = L3*np.cos(q3_3) + L2
        k2_2 = L3*np.sin(q3_3)

        k1_3 = L3*np.cos(q3_2) + L2
        k2_3 = L3*np.sin(q3_2)

        k1_4 = L3*np.cos(q3_4) + L2
        k2_4 = L3*np.sin(q3_4)

        sol1 = np.array([q1, np.atan2( -k2_1*(x*np.cos(q1) + y*np.sin(q1)) - k1_1*z, k1_1*(x*np.cos(q1) + y*np.sin(q1)) - k2_1*z), q3_1])
        sol2 = np.array([q1, np.atan2( -k2_2*(x*np.cos(q1) + y*np.sin(q1)) - k1_2*z, k1_2*(x*np.cos(q1) + y*np.sin(q1)) - k2_2*z), q3_3])

        sol3 = np.array([q1_, np.atan2( -k2_3*(x*np.cos(q1_) + y*np.sin(q1_)) - k1_3*z, k1_3*(x*np.cos(q1_) + y*np.sin(q1_)) - k2_3*z), q3_2])
        sol4 = np.array([q1_, np.atan2( -k2_4*(x*np.cos(q1_) + y*np.sin(q1_)) - k1_4*z, k1_4*(x*np.cos(q1_) + y*np.sin(q1_)) - k2_4*z), q3_4])

        return (sol1, sol2, sol3, sol4)


    
    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
            viewer.cam.lookat[:] = [0, 0, 0]
            viewer.cam.distance = 10
            viewer.cam.elevation = -45
            viewer.cam.azimuth = 45


            start = time.time()

            target = np.array([-0.3, -0.6, 1.2]) - np.array([0.0, 0.0, 1.0])

            sol1, sol2, sol3, sol4 = self.inverse_kinematics(target)
            print(f'{sol1}\n{sol2}\n{sol3}\n{sol4}')
            self.x_goal, self.y_goal, self.z_goal = target + np.array([0.0, 0.0, 1.0])

            self.data.qpos[:] = sol3
            self.data.qvel[:] = 0
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

                if self.solution == 1:
                    self.data.qpos[:] = sol1
                    self.data.qvel[:] = 0
                    mujoco.mj_forward(self.model, self.data)
                elif self.solution == 2:
                    self.data.qpos[:] = sol2
                    self.data.qvel[:] = 0
                    mujoco.mj_forward(self.model, self.data)
                elif self.solution == 3:
                    self.data.qpos[:] = sol3
                    self.data.qvel[:] = 0
                    mujoco.mj_forward(self.model, self.data)
                elif self.solution == 4:
                    self.data.qpos[:] = sol4
                    self.data.qvel[:] = 0
                    mujoco.mj_forward(self.model, self.data)

                ##############

                if not self.paused:
                    #self.data.ctrl = self.pd_control(sol1)
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