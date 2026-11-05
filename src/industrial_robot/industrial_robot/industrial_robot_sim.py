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
        self.L1 = 4.0
        self.L3 = 2.0
        self.L4 = 3.0
        self.d6 = 2.0

        mujoco.mj_forward(self.model, self.data)


    def add_frame_marker(self, scn, T, length=1.0, radius=0.05):
        R = T[:3, :3]
        p = T[:3, 3]
        colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]

        for i in range(3):
            if scn.ngeom >= scn.maxgeom:
                return
            axis = R[:, i]
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=np.zeros(3),
                pos=np.zeros(3),
                mat=np.eye(3).flatten(),
                rgba=np.array(colors[i], dtype=np.float32),
            )
            mujoco.mjv_connector(
                scn.geoms[scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                width=radius,
                from_=p,
                to=p + length * axis,
            )
            scn.ngeom += 1


    def key_callback(self, keycode):
            if chr(keycode) == ' ':
                self.paused = not self.paused
            elif chr(keycode) == '-':
                self.show_marker = not self.show_marker
            elif chr(keycode) == '1':
                self.show_solution = 1
            elif chr(keycode) == '2':
                self.show_solution = 2
            elif chr(keycode) == '3':
                self.show_solution = 3
            elif chr(keycode) == '4':
                self.show_solution = 4
            elif chr(keycode) == '5':
                self.show_solution = 5
            elif chr(keycode) == '6':
                self.show_solution = 6
            elif chr(keycode) == '7':
                self.show_solution = 7
            elif chr(keycode) == '8':
                self.show_solution = 8


    def inverse_kinematics(self, T_goal):
        R_goal = T_goal[:3, :3]
        p_goal = T_goal[:3, 3]

        p_zgl = p_goal - self.d6*R_goal[:, 2]

        x, y, z = p_zgl

        solutions = []

        for q1_offset in (0.0, np.pi):
            q1 = np.atan2(y, x) + q1_offset
            for sign_q3 in (-1.0, 1.0):
                a = (x*np.cos(q1) + y*np.sin(q1))
                q3 = sign_q3*np.arccos(((z - self.L1)**2 + a**2 - self.L4**2 - self.L3**2)/(2*self.L3*self.L4))
                k1 = self.L3 + self.L4*np.cos(q3)
                k2 = self.L4*np.sin(q3)
                q2 = np.atan2(k1*(self.L1 - z) - k2*a, k1*a + k2*(self.L1 - z))

                R_0_3 = np.array([[np.sin(q2 + q3)*np.cos(q1), np.cos(q2+ q3)*np.cos(q1), -np.sin(q1)],
                                  [np.sin(q2 + q3)*np.sin(q1), np.cos(q2 + q3)*np.sin(q1), np.cos(q1)],
                                  [np.cos(q2 + q3),           -np.sin(q2 + q3),            0         ]])
                R_3_6 = (R_0_3.T) @ R_goal                

                for sign_q5 in (-1.0, 1.0):
                    q5 = sign_q5*np.arccos(R_3_6[1, 2])
                    if abs(q5) < 1e-4:
                        q6 = 0.0
                        q4 = np.atan2(-R_3_6[2, 1], R_3_6[0, 1])
                    elif abs(abs(q5) - np.pi) < 1e-4:
                        q6 = 0.0
                        q4 = np.atan2(R_3_6[2, 1], R_3_6[0, 1])
                    else:
                        q4 = np.atan2(R_3_6[0, 2]/np.sin(q5), R_3_6[2, 2]/np.sin(q5))
                        q6 = np.atan2(R_3_6[1, 1]/np.sin(q5), -R_3_6[1, 0]/np.sin(q5))

                    solutions.append(np.array([q1, q2, q3, q4, q5, q6]))

        return solutions


    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
            viewer.cam.lookat[:] = [0, 0, 0]
            viewer.cam.distance = 26
            viewer.cam.elevation = -45
            viewer.cam.azimuth = 135

            T = np.array([[0.766, 0.0, 0.643, 2.5],
                          [0.321, 0.866, -0.383, 3.5],
                          [-0.557, 0.5, 0.663, 4.0],
                          [0.0, 0.0, 0.0, 1.0]])
                          
            sol1, sol2, sol3, sol4, sol5, sol6, sol7, sol8 = self.inverse_kinematics(T)

            self.data.qpos[:] = sol1
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

            start = time.time()

            while viewer.is_running() and time.time() - start < 99999:
                step_start = time.time()
    
                ### MARKER ###

                viewer.user_scn.ngeom = 0

                if self.show_marker and T is not None:
                    self.add_frame_marker(viewer.user_scn, T)

                if self.show_solution == 1:
                    self.data.qpos[:] = sol1
                    self.data.qvel[:] = 0
                    mujoco.mj_forward(self.model, self.data)
                elif self.show_solution == 2:
                    self.data.qpos[:] = sol2
                    self.data.qvel[:] = 0
                    mujoco.mj_forward(self.model, self.data)
                elif self.show_solution == 3:
                    self.data.qpos[:] = sol3
                    self.data.qvel[:] = 0
                    mujoco.mj_forward(self.model, self.data)
                elif self.show_solution == 4:
                    self.data.qpos[:] = sol4
                    self.data.qvel[:] = 0
                    mujoco.mj_forward(self.model, self.data)
                elif self.show_solution == 5:
                    self.data.qpos[:] = sol5
                    self.data.qvel[:] = 0
                    mujoco.mj_forward(self.model, self.data)
                elif self.show_solution == 6:
                    self.data.qpos[:] = sol6
                    self.data.qvel[:] = 0
                    mujoco.mj_forward(self.model, self.data)
                elif self.show_solution == 7:
                    self.data.qpos[:] = sol7
                    self.data.qvel[:] = 0
                    mujoco.mj_forward(self.model, self.data)
                elif self.show_solution == 8:
                    self.data.qpos[:] = sol8
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