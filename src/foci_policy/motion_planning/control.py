import roboticstoolbox as rtb
from spatialmath import SE3,base
import spatialmath as sm
import qpsolvers as qp
import numpy as np
import sys
import torch
import time

def arrived(panda, cur_joint, target_pose, threshold=0.001):
    n = 7
    panda.q = cur_joint
    Te = panda.fkine(cur_joint)

    Tep = target_pose.as_matrix()
    Tep = sm.SE3(Tep)

    # Transform from the end-effector to desired pose
    eTep = Te.inv() * Tep

    # Spatial error
    e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() ])) # * np.pi / 180

    # Calulate the required end-effector spatial velocity for the robot
    # to approach the goal. Gain is set to 1.0
    # v, arrived = rtb.p_servo(Te, Tep, 5, 0.001)
    v, arrived = rtb.p_servo(Te, Tep, 1, threshold)
    return e, arrived

def simple_calculate_velocity(panda, cur_joint, target_pose, Gain=1, threshold=0.001):
    Te = panda.fkine(cur_joint)
    Tep = target_pose.as_matrix()
    v, arrived = rtb.p_servo(Te, Tep, Gain, threshold)
    if arrived:
        return np.zeros(7,), True
    jacobian = panda.jacobe(cur_joint)
    jacobian_pinv = np.linalg.pinv(jacobian)
    dq_task = jacobian_pinv @ v
    return dq_task.reshape(-1), False

def simple_velocity_based_control(panda, cur_joint, tar_vel, ang_vel, Gain=1, onbase=True, j_vel=None):
    n = 7
    panda.q = cur_joint
    Te = panda.fkine(cur_joint)
    if onbase:
        tar_vel = Te.inv().A[:3,:3] @ tar_vel # convert absolute to relative 
        ang_vel = Te.inv().A[:3,:3] @ ang_vel

    v = np.r_[tar_vel, ang_vel] * Gain

    jacobian = panda.jacobe(cur_joint)
    jacobian_pinv = np.linalg.pinv(jacobian)
    dq_task = jacobian_pinv @ v
    if j_vel is not None:
        k_null = 0.1
        I = np.eye(jacobian.shape[1])  # 7x7 identity matrix
        null_space_projection = I - (jacobian_pinv @ jacobian)
        dq_0 = -j_vel.reshape(7,1) # damping effect
            
        dq_null = null_space_projection @ dq_0
        
        # 4. Combine total velocity
        dq_task = dq_task + k_null * dq_null.reshape(-1)

    return dq_task.reshape(-1)


def calculate_velocity(panda, cur_joint, target_pose, obstacles=None, Lambda=0.1, Gain=1, threshold=0.001, initvals=None):
    # The pose of the Panda's end-effector
    n = 7
    panda.q = cur_joint
    Te = panda.fkine(cur_joint)
    
    # t1 = time.time()
    try:
        Tep = target_pose.as_matrix()
    except:
        Tep = target_pose
    Tep = sm.SE3(Tep)


    # Transform from the end-effector to desired pose
    eTep = Te.inv() * Tep
    # t2 = time.time()
    # Spatial error
    e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy()])) #  * np.pi / 180

    # Calulate the required end-effector spatial velocity for the robot
    # to approach the goal. Gain is set to 1.0
    # v, arrived = rtb.p_servo(Te, Tep, 5, 0.001)
    v, arrived = rtb.p_servo(Te, Tep, Gain, threshold)
    # print('v:', v)
    
    # Gain term (lambda) for control minimisation
    Y = Lambda

    # v += rand(v.shape[0]) * v * 0.5 # * np.array([1,1,1,0,0,0])

    # Quadratic component of objective function
    Q = np.eye(n + 6)

    # Joint velocity component of Q
    Q[:n, :n] *= Y

    # Slack component of Q
    Q[n:, n:] = (1 / e) * np.eye(6)
    # t3 = time.time()
    # The equality contraints
    Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
    # t4 = time.time()

    beq = v.reshape((6,))

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((n + 6, n + 6))
    bin = np.zeros(n + 6)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = 0.05

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = 0.9

    # Form the joint limit velocity damper
    Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)
    
    if obstacles is not None:
        for collision in obstacles:
            # Form the velocity damper inequality contraint for each collision
            # object on the robot to the collision in the scene
            c_Ain, c_bin = panda.link_collision_damper(
                collision,
                panda.q[:n],
                0.3,
                0.05,
                1.0,
                start=panda.link_dict["panda_link1"],
                end=panda.link_dict["panda_hand"],
            )

            # If there are any parts of the robot within the influence distance
            # to the collision in the scene
            if c_Ain is not None and c_bin is not None:
                c_Ain = np.c_[c_Ain[:,:n], np.zeros((c_Ain.shape[0], 6))]

                # Stack the inequality constraints
                Ain = np.r_[Ain, c_Ain]
                bin = np.r_[bin, c_bin]

    # Linear component of objective function: the manipulability Jacobian
    c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6)]

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]
    ub = np.r_[panda.qdlim[:n], 10 * np.ones(6)]
    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='daqp', initvals=initvals)
    # t5 = time.time()
    # Apply the joint velocities to the Panda
    joint_velocity = qd[:n]

    # print(f"Timing: fkine+inv+error={t2-t1:.6f}s, p_servo={t3-t2:.6f}s, set_constraints={t4-t3:.6f}s, qp_solve={t5-t4:.6f}s")

    return joint_velocity, arrived



def velocity_based_control(panda, cur_joint, tar_vel, ang_vel, Lambda=0.1, Gain=1, obstacles=None, onbase=True):
    # The pose of the Panda's end-effector
    n = 7
    panda.q = cur_joint
    Te = panda.fkine(cur_joint)
    if onbase:
        tar_vel = Te.inv().A[:3,:3] @ tar_vel # convert absolute to relative 
        ang_vel = Te.inv().A[:3,:3] @ ang_vel

    # Spatial error
    e = np.sum(np.abs(np.r_[tar_vel, ang_vel]))

    # Calulate the required end-effector spatial velocity for the robot
    # to approach the goal. Gain is set to 1.0
    v = np.r_[tar_vel, ang_vel] * Gain

    # Gain term (lambda) for control minimisation
    Y = Lambda

    # Quadratic component of objective function
    Q = np.eye(n + 6)

    # Joint velocity component of Q
    Q[:n, :n] *= Y

    # Slack component of Q
    Q[n:, n:] = (1 / e) * np.eye(6)

    # The equality contraints
    Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
    beq = v.reshape((6,))

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((n + 6, n + 6))
    bin = np.zeros(n + 6)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = 0.05

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = 0.9

    # Form the joint limit velocity damper
    Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)

    if obstacles is not None:
        for collision in obstacles:
            # Form the velocity damper inequality contraint for each collision
            # object on the robot to the collision in the scene
            c_Ain, c_bin = panda.link_collision_damper(
                collision,
                panda.q[:n],
                0.3,
                0.05,
                1.0,
                start=panda.link_dict["panda_link1"],
                end=panda.link_dict["panda_hand"],
            )
            

            # If there are any parts of the robot within the influence distance
            # to the collision in the scene
            if c_Ain is not None and c_bin is not None:
                c_Ain = np.c_[c_Ain[:,:n], np.zeros((c_Ain.shape[0], 6))]

                # Stack the inequality constraints
                Ain = np.r_[Ain, c_Ain]
                bin = np.r_[bin, c_bin]

    # Linear component of objective function: the manipulability Jacobian
    c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6)]

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]
    ub = np.r_[panda.qdlim[:n], 10 * np.ones(6)]
    # print(c_Ain)
    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='daqp')

    # Apply the joint velocities to the Panda
    joint_velocity = qd[:n]

    return joint_velocity

