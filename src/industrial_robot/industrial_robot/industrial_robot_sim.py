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
        self.show_solution = 1
        self.x_goal = None
        self.y_goal = None
        self.z_goal = None
        self.L1 = 4.0
        self.L3 = 2.0
        self.L4 = 3.0
        self.d6 = 2.0

        mujoco.mj_forward(self.model, self.data)


    def key_callback(self, keycode):
            if chr(keycode) == ' ':
                self.paused = not self.paused
            elif chr(keycode) == '6':
                self.show_marker = not self.show_marker
            elif chr(keycode) == '7':
                self.show_solution = 1
            elif chr(keycode) == '8':
                self.show_solution = 2
            elif chr(keycode) == '9':
                self.show_solution = 3
            elif chr(keycode) == '-':
                self.show_solution = 4

    def inverse_kinematics(self, T_goal):
        R_goal = T_goal[:3, :3]
        x, y, z = T_goal[:3, 3]

        ### SOLUTION 1 (q1 bez '+ pi', q3 plus)###
        q1_1 = np.atan2(y, x)
        q3_1 = np.arccos(((z - self.L1)**2 + (x*np.cos(q1_1) + y*np.sin(q1_1))**2 - self.L4**2 - self.L3**2)/(2*self.L3*self.L4))

        k1_1 = self.L3 + self.L4*np.cos(q3_1)
        k2_1 = self.L4*np.sin(q3_1)

        q2_1 = np.atan2(k1_1*(self.L1 - z) - k2_1*(x*np.cos(q1_1) + y*np.sin(q1_1)), k1_1*(x*np.cos(q1_1) + y*np.sin(q1_1)) + k2_1*(self.L1 - z))

        sol1 = np.array([q1_1, q2_1, q3_1])
        ##################

        ### SOLUTION 2 (q1 sa '+ pi', q3 plus)###
        q1_2 = np.atan2(y, x) + np.pi
        q3_2 = np.arccos(((z - self.L1)**2 + (x*np.cos(q1_2) + y*np.sin(q1_2))**2 - self.L4**2 - self.L3**2)/(2*self.L3*self.L4))

        k1_2 = self.L3 + self.L4*np.cos(q3_2)
        k2_2 = self.L4*np.sin(q3_2)

        q2_2 = np.atan2(k1_2*(self.L1 - z) - k2_2*(x*np.cos(q1_2) + y*np.sin(q1_2)), k1_2*(x*np.cos(q1_2) + y*np.sin(q1_2)) + k2_2*(self.L1 - z))

        sol2 = np.array([q1_2, q2_2, q3_2])
        ##################

        ### SOLUTION 3 (q1 bez '+ pi', q3 minus)###
        q1_3 = np.atan2(y, x)
        q3_3 = -np.arccos(((z - self.L1)**2 + (x*np.cos(q1_3) + y*np.sin(q1_3))**2 - self.L4**2 - self.L3**2)/(2*self.L3*self.L4))

        k1_3 = self.L3 + self.L4*np.cos(q3_3)
        k2_3 = self.L4*np.sin(q3_3)

        q2_3 = np.atan2(k1_3*(self.L1 - z) - k2_3*(x*np.cos(q1_3) + y*np.sin(q1_3)), k1_3*(x*np.cos(q1_3) + y*np.sin(q1_3)) + k2_3*(self.L1 - z))
        sol3 = np.array([q1_3, q2_3, q3_3])
        ##################
        
        ### SOLUTION 4 (q1 sa '+ pi', q3 minus)###
        q1_4 = np.atan2(y, x) + np.pi
        q3_4 = -np.arccos(((z - self.L1)**2 + (x*np.cos(q1_4) + y*np.sin(q1_4))**2 - self.L4**2 - self.L3**2)/(2*self.L3*self.L4))

        k1_4 = self.L3 + self.L4*np.cos(q3_4)
        k2_4 = self.L4*np.sin(q3_4)

        q2_4 = np.atan2(k1_4*(self.L1 - z) - k2_4*(x*np.cos(q1_4) + y*np.sin(q1_4)), k1_4*(x*np.cos(q1_4) + y*np.sin(q1_4)) + k2_4*(self.L1 - z))

        sol4 = np.array([q1_4, q2_4, q3_4])
        ##################

        self.get_logger().info(f'{sol1}\n{sol2}\n{sol3}\n{sol4}')

        return sol1, sol2, sol3, sol4

    
    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
            viewer.cam.lookat[:] = [0, 0, 0]
            viewer.cam.distance = 26
            viewer.cam.elevation = -45
            viewer.cam.azimuth = 135


            start = time.time()


            while viewer.is_running() and time.time() - start < 99999:
                step_start = time.time()

                T = np.array([[1, 0, 0, 2.5],
                              [0, 1, 2, 3.5],
                              [0, 0, 1, 3.0],
                              [0, 0, 0, 1]])

                sol1, sol2, sol3, sol4 = self.inverse_kinematics(T)
                self.x_goal, self.y_goal, self.z_goal = T[:3, 3]

                self.data.qpos[:3] = sol1
                self.data.qpos[3:] = 0.0
                self.data.qvel[:] = 0.0
                mujoco.mj_forward(self.model, self.data)
    
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

                if self.show_solution == 1:
                    self.data.qpos[:3] = sol1
                    self.data.qpos[3:] = 0.0
                    self.data.qvel[:] = 0
                    mujoco.mj_forward(self.model, self.data)
                elif self.show_solution == 2:
                    self.data.qpos[:3] = sol2
                    self.data.qpos[3:] = 0.0
                    self.data.qvel[:] = 0
                    mujoco.mj_forward(self.model, self.data)
                elif self.show_solution == 3:
                    self.data.qpos[:3] = sol3
                    self.data.qpos[3:] = 0.0
                    self.data.qvel[:] = 0
                    mujoco.mj_forward(self.model, self.data)
                elif self.show_solution == 4:
                    self.data.qpos[:3] = sol4
                    self.data.qpos[3:] = 0.0
                    self.data.qvel[:] = 0
                    mujoco.mj_forward(self.model, self.data)

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