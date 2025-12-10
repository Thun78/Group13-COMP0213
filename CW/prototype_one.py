import pybullet as p
import pybullet_data
import pandas as pd
import time
import numpy as np
import random
from abc import ABC, abstractmethod
import os

# ---- Abstract Class ----
class SceneObject(ABC):
    def __init__(self, urdf_file, position, orientation=(0, 0, 0)):
        self.urdf_file = urdf_file
        self.position = position
        self.orientation = p.getQuaternionFromEuler(orientation)
        self.id = None
    @abstractmethod
    def load(self):
        pass

class Grippers(ABC):
    def __init__(self, urdf_file, position, orientation=(0, 0, 0)):
        self.urdf_file = urdf_file
        self.position = position
        self.orientation = p.getQuaternionFromEuler(orientation)
        self.id = None

    def move_up_classify(self, obj_id, new_z=1, duration=1.5, no_contact_timeout=0.5,
                contact_ratio_threshold=0.99):
        sim_hz = 50
        total_steps = max(1, int(duration * sim_hz))
        timeout_steps = max(1, int(no_contact_timeout * sim_hz))

        contact_steps = 0
        total_contact_checks = 0
        consecutive_no_contact_steps = 0

        start_z = self.position[2]
        dz = (new_z - start_z) / float(total_steps)

        for step_idx in range(total_steps):
            self.position[2] += dz
            p.changeConstraint(self.constraint_id,
                            jointChildPivot=self.position,
                            maxForce=100)

            # check contact
            contacts = p.getContactPoints(bodyA=self.id, bodyB=obj_id) or []
            has_contact = len(contacts) > 0

            if has_contact:
                contact_steps += 1
                consecutive_no_contact_steps = 0
            else:
                consecutive_no_contact_steps += 1
            total_contact_checks += 1

            # --- EARLY ABORT RULE ---
            if consecutive_no_contact_steps >= timeout_steps:
                print(f"[move_up] Lost contact for {no_contact_timeout} seconds (consecutive_no_contact_steps={consecutive_no_contact_steps}) → aborting and returning False")
                return False

            # step simulation
            p.stepSimulation()
            time.sleep(1./240.)

        # finish lift, compute ratio for final decision
        contact_ratio = contact_steps / float(max(1, total_contact_checks))
        print(f"[move_up] contact_ratio={contact_ratio:.2f}")
        return contact_ratio >= contact_ratio_threshold

    def move_pickup(self, obj_position, stop_dist):
        dx = obj_position[0] - self.position[0]
        dy = obj_position[1] - self.position[1]
        dz = obj_position[2] - self.position[2]
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)

        steps = int(dist/ 0.01)

        for step in range(steps):
            alpha = (step + 1) / steps
            new_x = self.position[0] + dx * alpha
            new_y = self.position[1] + dy * alpha
            new_z = self.position[2] + dz * alpha

            p.changeConstraint(self.constraint_id, 
                           jointChildPivot=[new_x, new_y, new_z],
                           maxForce=100)
            p.stepSimulation()
            time.sleep(1./240.)

            # check distance from new end-effector to object
            cur_dx = obj_position[0] - new_x
            cur_dy = obj_position[1] - new_y
            cur_dz = obj_position[2] - new_z
            cur_dist = np.sqrt(cur_dx*cur_dx + cur_dy*cur_dy + cur_dz*cur_dz)
            if cur_dist < stop_dist:
                break

        self.position = [new_x, new_y, new_z]
        
    @abstractmethod
    def load(self):
        pass
    @abstractmethod
    def open_gripper(self):
        pass
    @abstractmethod
    def close_gripper(self):
        pass

# ---- Class ----
class Cube(SceneObject):
    def __init__(self, position, orientation=(0, 0, 0)):
        super().__init__("cube_small.urdf", position, orientation)

    def load(self, scaling=1):
        self.id = p.loadURDF(self.urdf_file, self.position, self.orientation, globalScaling=scaling)
        return self.id
    
class Cylinder(SceneObject):

    def __init__(self, position, orientation=(0, 0, 0)):
        current_dir1 = os.path.dirname(os.path.abspath(__file__))  
        urdf_path1 = os.path.join(current_dir1, "cylinder.urdf")
        super().__init__(urdf_path1, position, orientation)

    def load(self, scaling=1):
        self.id = p.loadURDF(self.urdf_file, self.position, self.orientation, globalScaling=scaling)
        return self.id
    
class ThreeFingersGripper(Grippers):
    GRASP_JOINTS = [1, 4, 7]
    PRESHAPE_JOINTS = [2, 5, 8]
    UPPER_JOINTS = [3, 6, 9]

    def __init__(self, position, orientation=(0, 0, 0)):
        current_dir = os.path.dirname(os.path.abspath(__file__))  
        urdf_path = os.path.join(current_dir, "grippers", "threeFingers", "sdh", "sdh.urdf")
        super().__init__(urdf_path, position, orientation)
        self.open = False
        self.num_joints = 0

    def load(self):
        self.id = p.loadURDF(self.urdf_file, self.position, self.orientation)
        self.num_joints = p.getNumJoints(self.id)
        return self.id
    
    def attach_fixed(self, offset=[0,0,0]):
        self.constraint_id = p.createConstraint(
            parentBodyUniqueId = self.id,
            parentLinkIndex = -1,
            childBodyUniqueId = -1,
            childLinkIndex = -1,
            jointType = p.JOINT_FIXED,
            jointAxis = [0, 0, 0],
            parentFramePosition = offset,
            childFramePosition = self.position,
            parentFrameOrientation = [0,0,0,1],
            childFrameOrientation = self.orientation
        )

    def get_joint_positions(self):
        return [p.getJointState(self.id, i)[0] for i in range(self.num_joints)]
    
    def _apply_joint_command(self, joint, target, max_velo=3):
        p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                targetPosition=target, maxVelocity=max_velo, force=20)

    def open_gripper(self):
        closed, iteration = True, 0
        while closed and not self.open:
            joints = self.get_joint_positions()
            closed = False
            for k in range(self.num_joints):
                if k in [2, 5, 8] and joints[k] >= 0.9:
                    self._apply_joint_command(k, joints[k] - 0.05)
                    closed = True
                elif k in [3, 6, 9] and joints[k] <= 0.9:
                    self._apply_joint_command(k, joints[k] - 0.05)
                    closed = True
                elif k in [1, 4, 7] and joints[k] <= 0.9:
                    self._apply_joint_command(k, joints[k] - 0.05)
                    closed = True
            iteration += 1
            if iteration > 1000:
                break
            p.stepSimulation()
        self.open = True

    def pregrasp(self):
        for i in [2, 5, 8]:
            self._apply_joint_command(i, 0.4)
        self.open = False

    def close_gripper(self):
        for j in [1, 4]:
            self._apply_joint_command(j, 0.05, 5)
        self._apply_joint_command(7, 0.05, 5)

class TwoFingersGripper(Grippers):
    def __init__(self, position, orientation=(0, 0, 0)):
        super().__init__("pr2_gripper.urdf", position, orientation)

    def load(self):
        self.id = p.loadURDF(self.urdf_file, self.position, self.orientation)
        return self.id
    
    def attach_fixed(self, offset):
        self.constraint_id = p.createConstraint(
            parentBodyUniqueId = self.id,
            parentLinkIndex = -1,
            childBodyUniqueId = -1,
            childLinkIndex = -1,
            jointType = p.JOINT_FIXED,
            jointAxis = [0, 0, 0],
            parentFramePosition = offset,
            childFramePosition = self.position,
            parentFrameOrientation = [0,0,0,1],
            childFrameOrientation = self.orientation
        )

    def open_gripper(self):
        for joint in [0, 2]:
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                    targetPosition=0.9, maxVelocity=10, force=40)

    def close_gripper(self):
        for joint in [0, 2]:
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                    targetPosition=0, maxVelocity=2.7, force=200)

class Sample():
    current_sample = 0

    def __init__(self, obj_center, radius, samples=5):
        self.samples = samples
        self.obj_center = obj_center
        self.radius = radius
        self.sample_name = None
        Sample.current_sample += 1
    
    def sample_grasp_pose(self, gripper_type, obj_type):
        positions = []
        orientations = []

        radius_noise_std = 0.01
        r_noisy = self.radius + np.random.normal(0.0, radius_noise_std)

        threefinger_offset_EULER = [0, np.pi/2, 0]   # adjust sign if needed
        threefinger_offset__QUAT  = p.getQuaternionFromEuler(threefinger_offset_EULER)

        for _ in range(self.samples):
            # Sample on sphere surface around the object
            theta = random.uniform(0, 2 * np.pi)  # Azimuth
            phi = random.uniform(0, np.pi/2)        # Polar
                
            x = r_noisy * np.sin(phi) * np.cos(theta) + self.obj_center[0]
            y = r_noisy * np.sin(phi) * np.sin(theta) + self.obj_center[1]
            z = r_noisy * np.cos(phi) + self.obj_center[2]
            position = [x, y, z]
            positions.append(position)

            dx = self.obj_center[0] - x
            dy = self.obj_center[1] - y
            dz = self.obj_center[2] - z
            dist_xy = np.sqrt(dx*dx + dy*dy)

            yaw = np.atan2(dy, dx)
            pitch = -np.atan2(dz, dist_xy)

    
            roll = random.uniform(-np.pi/2, np.pi/2) 

            orientation = [roll, pitch, yaw]

            if gripper_type == "three-finger":
                q_sample = p.getQuaternionFromEuler(orientation)

                _, q_hand = p.multiplyTransforms(
                    [0, 0, 0], q_sample,
                    [0, 0, 0], threefinger_offset__QUAT
                )

                rpy_hand = p.getEulerFromQuaternion(q_hand)
                orientations.append(list(rpy_hand))
            else:
                orientations.append(orientation)

        samples_po_orein = pd.DataFrame({
            "positions": positions,
            "orientations": orientations})
        
        return samples_po_orein  # 6D pose

# ---- Functions ----
def relative_position(gripper_id, obj_id):
    g_pos, g_orn = p.getBasePositionAndOrientation(gripper_id)
    o_pos, o_orn = p.getBasePositionAndOrientation(obj_id)

    # Convert to relative
    inv_o_pos, inv_o_orn = p.invertTransform(o_pos, o_orn)
    rel_pos, rel_orn = p.multiplyTransforms(inv_o_pos, inv_o_orn, g_pos, g_orn)

    # Convert to Euler
    rel_roll, rel_pitch, rel_yaw = p.getEulerFromQuaternion(rel_orn)
    return rel_pos, (rel_roll, rel_pitch, rel_yaw)

def setup_environment():
    cid = p.connect(p.GUI)
    print("Connection ID:", cid)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -10)
    p.setRealTimeSimulation(0)

    plane_id = p.loadURDF("plane.urdf")

    p.resetDebugVisualizerCamera(
        cameraDistance=0.5,
        cameraYaw=40,
        cameraPitch=-30,
        cameraTargetPosition=[0.6, 0.3, 0.2]
    )
    return plane_id

def generate_dataset(gripper_type, chosen_object, sample_num=10):
    graspable = []
    rel_x = []
    rel_y = []
    rel_z = []
    rel_roll = []
    rel_pitch = []
    rel_yaw = []

    if gripper_type == "two-finger" and chosen_object == "cube":
        print("two-finger cube")
        object_pos = [0.6, 0.3, 0.025]
        scaling = 1
        obj1 = Cube(object_pos)
        obj1_id = obj1.load(scaling)
    elif gripper_type == "three-finger" and chosen_object == "cube":
        print("three-finger cube")
        object_pos = [0.6, 0.3, 0.05]
        scaling = 2
        obj1 = Cube(object_pos)
        obj1_id = obj1.load(scaling)
    elif chosen_object == "cylinder" and gripper_type == "two-finger":
        print(f"{gripper_type} cylinder")
        object_pos = [0.6, 0.3, 0.05]
        scaling = 0.5
        obj1 = Cylinder(object_pos)
        obj1_id = obj1.load(scaling)
        p.changeDynamics(
        obj1_id,
        -1,                    # base link
        lateralFriction=1,
        spinningFriction=0.05,
        rollingFriction=0.05
        )
    elif chosen_object == "cylinder" and gripper_type == "three-finger":
        print(f"{gripper_type} cylinder")
        object_pos = [0.6, 0.3, 0.1]
        obj1 = Cylinder(object_pos)
        obj1_id = obj1.load()
        p.changeDynamics(
        obj1_id,
        -1,                    # base link
        lateralFriction=1,
        spinningFriction=0.05,
        rollingFriction=0.05
        )
    
    sampler = Sample(obj_center=object_pos, radius=1, samples=sample_num)
    df_samples = sampler.sample_grasp_pose(gripper_type, obj_type=chosen_object)

    for i in range(sample_num):
        pos = df_samples["positions"][i]
        orn = df_samples["orientations"][i]

        # Create gripper instance
        if gripper_type == "three-finger":
            gripper = ThreeFingersGripper(position=pos, orientation=orn)
        else:
            gripper = TwoFingersGripper(position=pos, orientation=orn)
        current_gid = gripper.load()
        gripper.attach_fixed(offset=[0,0,0])

        # record relative pose
        rel_pos, rel_orn = relative_position(current_gid, obj1_id)
        rel_x.append(rel_pos[0])
        rel_y.append(rel_pos[1])
        rel_z.append(rel_pos[2])
        rel_roll.append(rel_orn[0])
        rel_pitch.append(rel_orn[1])
        rel_yaw.append(rel_orn[2])

        gripper.open_gripper()
        if gripper_type == "three-finger" and chosen_object == "cube":
            print("three-finger cube grasp")
            gripper.pregrasp()
            distance = 0.152
        elif gripper_type == "two-finger" and chosen_object == "cube":
            print("two-finger cube grasp")
            distance = 0.31
        elif gripper_type == "three-finger" and chosen_object == "cylinder":
            print("three-finger cylinder grasp")
            gripper.pregrasp()
            distance = 0.161
        elif gripper_type == "two-finger" and chosen_object == "cylinder":
            print("two-finger cylinder grasp")
            distance = 0.31
        
        # wait for fingers to open
        for _ in range(40):
            p.stepSimulation()
            time.sleep(1./240.)

        # attempt grasp
        gripper.move_pickup(obj_position=object_pos, stop_dist= distance)
        gripper.close_gripper()
        for _ in range(70):
            p.stepSimulation()
            time.sleep(1./240.)
        result = gripper.move_up_classify(obj1_id)

        print(result)
        graspable.append(result)
        p.removeBody(current_gid)

        # reset object position before next grasp attempt
        p.resetBasePositionAndOrientation(obj1_id, object_pos, [0,0,0,1])
        for _ in range(40):
            p.stepSimulation()
            time.sleep(1./240.)

    p.removeBody(obj1_id)
    df_real = pd.DataFrame({
        "rel_x": rel_x,
        "rel_y": rel_y,
        "rel_z": rel_z,
        "rel_roll": rel_roll,
        "rel_pitch": rel_pitch,
        "rel_yaw": rel_yaw,
        "success": graspable})
    print(df_real) 
    return df_real

# ---- MAIN ----
def main(sample_num = 10):
    setup_environment()

    three_finger_cube = generate_dataset(gripper_type="three-finger", chosen_object="cube", sample_num=sample_num)
    three_finger_cube.to_csv("three_finger_cube.csv", index=False)

    two_finger_cube = generate_dataset(gripper_type="two-finger", chosen_object="cube", sample_num=sample_num)
    two_finger_cube.to_csv("two_finger_cube.csv", index=False)

    two_finger_cylinder = generate_dataset(gripper_type="two-finger", chosen_object="cylinder", sample_num=sample_num)
    two_finger_cylinder.to_csv("two_finger_cylinder.csv", index=False)

    three_finger_cylinder = generate_dataset(gripper_type="three-finger", chosen_object="cylinder", sample_num=sample_num)
    three_finger_cylinder.to_csv("three_finger_cylinder.csv", index=False)

if __name__ == "__main__":
    main()