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
        package_share = get_package_share_directory('kuka_iiwa')
        self.xml_path = os.path.join(package_share, 'models', 'lbr_iiwa14', 'scene_with_pd_control.xml')
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.paused = False
        self.data.qpos[:] = self.model.key_qpos[0]
        self.data.ctrl[:] = self.model.key_ctrl[0]
        mujoco.mj_forward(self.model, self.data)

    def key_callback(self, keycode):
        if chr(keycode) == ' ':
            self.paused = not self.paused

    def run_simulation(self):
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
            start = time.time()
            while viewer.is_running() and time.time() - start < 99999:
                step_start = time.time()

                pos_0, R_0 = self.forward_kinematics(self.data.qpos)

                pos_final = np.array([0.6, 0.4, 0.7])

                alfa = (step_start - start)/10.0

                pos_t = pos_0 + alfa*(pos_final - pos_0)

                if alfa < 0.99:
                    q_ref = self.inverse_kinematics(pos_t)
                    self.data.ctrl = q_ref

                
                if not self.paused:
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()

                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


    def inverse_kinematics(self, target_pos, site_name="attachment_site",
        target_quat=None, max_iters=100, tol=1e-4,
        step_size=0.5, damping=1e-3):
        model = self.model
        data = mujoco.MjData(model)          # <-- posebna kopija, ne self.data!
        data.qpos[:] = self.data.qpos        # kreni od trenutnog stanja robota
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

        qpos = data.qpos.copy()

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        for i in range(max_iters):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)   # ovo sad radi na "senci", ne na pravom robotu

            current_pos = data.site_xpos[site_id]
            pos_err = target_pos - current_pos

            if target_quat is not None:
                current_mat = data.site_xmat[site_id].reshape(3, 3)
                current_quat = np.zeros(4)
                mujoco.mju_mat2Quat(current_quat, current_mat.flatten())
                neg_quat = np.zeros(4)
                mujoco.mju_negQuat(neg_quat, current_quat)
                err_quat = np.zeros(4)
                mujoco.mju_mulQuat(err_quat, target_quat, neg_quat)
                quat_err = np.zeros(3)
                mujoco.mju_quat2Vel(quat_err, err_quat, 1.0)
                err = np.concatenate([pos_err, quat_err])
            else:
                err = pos_err

            if np.linalg.norm(err) < tol:
                break

            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            J = jacp if target_quat is None else np.vstack([jacp, jacr])

            JJt = J @ J.T
            lam = damping * np.eye(JJt.shape[0])
            dq = J.T @ np.linalg.solve(JJt + lam, err)

            qpos[:model.nv] += step_size * dq

        return qpos   # samo vraća uglove, self.data ostaje netaknut

    def forward_kinematics(self, qpos, site_name="attachment_site"):
        data = mujoco.MjData(self.model)

        data.qpos[:] = qpos
        mujoco.mj_forward(self.model, data)

        site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name
        )

        position = data.site_xpos[site_id].copy()
        rotation = data.site_xmat[site_id].reshape(3, 3).copy()

        return position, rotation
         

def main(args=None):
    rclpy.init(args=args)

    node = SimulateNode()
    node.run_simulation()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()