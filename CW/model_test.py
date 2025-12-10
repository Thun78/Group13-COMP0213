from __future__ import annotations

import os
import time
import math
import random
import json
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pybullet as p
import pybullet_data
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

sns.set(context="talk", style="whitegrid")



# Environment Manager
class EnvironmentManager:
    # Initialize the environment manager with GUI option.
    def __init__(self, gui: bool = True) -> None:
        self.gui = gui
        self.connection_id = None
        self.plane_id = None

    def setup(self) -> int:
        self.connection_id = p.connect(p.GUI if self.gui else p.DIRECT)
        print("Connection ID:", self.connection_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation() 
        p.setGravity(0, 0, -10)  # Set gravity for the simulation (x=0, y=0, z=-10 m/s^2).
        p.setRealTimeSimulation(0)  # Disable real-time simulation
        self.plane_id = p.loadURDF("plane.urdf")
        p.resetDebugVisualizerCamera(
            cameraDistance=0.5,
            cameraYaw=40,
            cameraPitch=-30,
            cameraTargetPosition=[0.6, 0.3, 0.2],
        )
        return self.plane_id

    def step(self, steps: int = 1, hz: float = 240.0) -> None:
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(1.0 / hz)


# Abstract Scene Object
class BaseSceneObject(ABC):
    def __init__(self, urdf_file: str, position: List[float], orientation_euler: Tuple[float, float, float] = (0, 0, 0)) -> None:
        self.urdf_file = urdf_file
        self.position = position
        self.orientation_quat = p.getQuaternionFromEuler(orientation_euler)
        self.id = None

    @abstractmethod
    def load(self, scaling: float = 1.0) -> int:
        # Abstract method to load the object into the simulation.
        raise NotImplementedError



class Cube(BaseSceneObject):
    def __init__(self, position: List[float], orientation_euler: Tuple[float, float, float] = (0, 0, 0)) -> None:
        # Initialize Cube with a fixed URDF file "cube_small.urdf"
        super().__init__("cube_small.urdf", position, orientation_euler)

    def load(self, scaling: float = 1.0) -> int:
        # Load the cube into the PyBullet simulation.
        self.id = p.loadURDF(self.urdf_file, self.position, self.orientation_quat, globalScaling=scaling)
        return self.id


class Cylinder(BaseSceneObject):
    def __init__(self, position: List[float], orientation_euler: Tuple[float, float, float] = (0, 0, 0)) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir, "cylinder.urdf")
        super().__init__(urdf_path, position, orientation_euler)

    def load(self, scaling: float = 1.0) -> int:
        self.id = p.loadURDF(self.urdf_file, self.position, self.orientation_quat, globalScaling=scaling)
        return self.id


# Abstract Gripper
class BaseGripper(ABC):
    def __init__(self, urdf_file: str, position: List[float], orientation_euler: Tuple[float, float, float] = (0, 0, 0)) -> None:
        self.urdf_file = urdf_file
        self.position = position
        self.orientation_quat = p.getQuaternionFromEuler(orientation_euler)
        self.id = None
        self.constraint_id = None

    @abstractmethod
    def load(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def attach_fixed(self, offset: List[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def open_gripper(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close_gripper(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def move_pickup(self, target_position: List[float], stop_dist: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def move_up_classify(
        self,
        obj_id: int,
        new_z: float = 1.0,
        duration: float = 2.5,
        no_contact_timeout: float = 0.7,
        contact_ratio_threshold: float = 0.7,
        debug_contacts: bool = False,
    ) -> bool:
        raise NotImplementedError


class TwoFingersGripper(BaseGripper):
    def __init__(self, position: List[float], orientation_euler: Tuple[float, float, float] = (0, 0, 0)) -> None:
        super().__init__("pr2_gripper.urdf", position, orientation_euler)

    def load(self) -> int:
        self.id = p.loadURDF(self.urdf_file, self.position, self.orientation_quat)
        return self.id

    def attach_fixed(self, offset: List[float]) -> None:
        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=offset,
            childFramePosition=self.position,
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=self.orientation_quat,
        )

    def open_gripper(self) -> None:
        for joint in [0, 2]:
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL, targetPosition=0.9, maxVelocity=10, force=40)

    def close_gripper(self) -> None:
        for joint in [0, 2]:
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL, targetPosition=0.0, maxVelocity=2.7, force=200)

    def move_pickup(self, target_position: List[float], stop_dist: float) -> None:
        dx = target_position[0] - self.position[0]
        dy = target_position[1] - self.position[1]
        dz = target_position[2] - self.position[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        steps = max(1, int(dist / 0.01))

        new_x, new_y, new_z = self.position
        for step in range(steps):
            alpha = (step + 1) / steps
            new_x = self.position[0] + dx * alpha
            new_y = self.position[1] + dy * alpha
            new_z = self.position[2] + dz * alpha
            p.changeConstraint(self.constraint_id, jointChildPivot=[new_x, new_y, new_z], maxForce=300)
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

            cur_dx = target_position[0] - new_x
            cur_dy = target_position[1] - new_y
            cur_dz = target_position[2] - new_z
            cur_dist = math.sqrt(cur_dx * cur_dx + cur_dy * cur_dy + cur_dz * cur_dz)
            if cur_dist < stop_dist:
                break

        self.position = [new_x, new_y, new_z]

    def move_up_classify(self, obj_id: int, new_z: float = 1.0, duration: float = 2.5, no_contact_timeout: float = 0.7, contact_ratio_threshold: float = 0.7, debug_contacts: bool = False) -> bool:
        sim_hz = 50
        total_steps = max(1, int(duration * sim_hz))
        timeout_steps = max(1, int(no_contact_timeout * sim_hz))

        contact_steps = 0
        total_contact_checks = 0
        consecutive_no_contact_steps = 0

        start_z = self.position[2]
        dz = (new_z - start_z) / float(total_steps)

        for _ in range(total_steps):
            self.position[2] += dz
            p.changeConstraint(self.constraint_id, jointChildPivot=self.position, maxForce=300)
            contacts = p.getContactPoints(bodyA=self.id, bodyB=obj_id) or []
            if debug_contacts:
                print(f"[TwoFinger Lift] contacts={len(contacts)}")

            has_contact = len(contacts) > 0
            if has_contact:
                contact_steps += 1
                consecutive_no_contact_steps = 0
            else:
                consecutive_no_contact_steps += 1
            total_contact_checks += 1

            if consecutive_no_contact_steps >= timeout_steps:
                return False

            p.stepSimulation()
            time.sleep(1.0 / 240.0)

        contact_ratio = contact_steps / float(max(1, total_contact_checks))
        return contact_ratio >= contact_ratio_threshold


class ThreeFingersGripper(BaseGripper):
    GRASP_JOINTS = [1, 4, 7]
    PRESHAPE_JOINTS = [2, 5, 8]
    UPPER_JOINTS = [3, 6, 9]

    def __init__(self, position: List[float], orientation_euler: Tuple[float, float, float] = (0, 0, 0)) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir, "grippers", "threeFingers", "sdh", "sdh.urdf")
        super().__init__(urdf_path, position, orientation_euler)
        self.open = False
        self.num_joints = 0

    def load(self) -> int:
        self.id = p.loadURDF(self.urdf_file, self.position, self.orientation_quat)
        self.num_joints = p.getNumJoints(self.id)
        return self.id

    def attach_fixed(self, offset: List[float]) -> None:
        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=offset,
            childFramePosition=self.position,
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=self.orientation_quat,
        )

    def get_joint_positions(self) -> List[float]:
        return [p.getJointState(self.id, i)[0] for i in range(self.num_joints)]

    def _apply_joint_command(self, joint: int, target: float, max_velo: float = 3.0, force: float = 60.0) -> None:
        p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL, targetPosition=target, maxVelocity=max_velo, force=force)

    def open_gripper(self) -> None:
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

    def pregrasp(self) -> None:
        for i in [2, 5, 8]:
            self._apply_joint_command(i, 0.4)
        self.open = False

    def close_gripper(self) -> None:
        for j in [1, 4]:
            self._apply_joint_command(j, 0.05, 5, force=100.0)
        self._apply_joint_command(7, 0.05, 5, force=100.0)

    def move_pickup(self, target_position: List[float], stop_dist: float) -> None:
        dx = target_position[0] - self.position[0]
        dy = target_position[1] - self.position[1]
        dz = target_position[2] - self.position[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        steps = max(1, int(dist / 0.01))

        new_x, new_y, new_z = self.position
        for step in range(steps):
            alpha = (step + 1) / steps
            new_x = self.position[0] + dx * alpha
            new_y = self.position[1] + dy * alpha
            new_z = self.position[2] + dz * alpha
            p.changeConstraint(self.constraint_id, jointChildPivot=[new_x, new_y, new_z], maxForce=300)
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

            cur_dx = target_position[0] - new_x
            cur_dy = target_position[1] - new_y
            cur_dz = target_position[2] - new_z
            cur_dist = math.sqrt(cur_dx * cur_dx + cur_dy * cur_dy + cur_dz * cur_dz)
            if cur_dist < stop_dist:
                break

        self.position = [new_x, new_y, new_z]

    def move_up_classify(self, obj_id: int, new_z: float = 1.0, duration: float = 2.8, no_contact_timeout: float = 0.8, contact_ratio_threshold: float = 0.65, debug_contacts: bool = False) -> bool:
        sim_hz = 50
        total_steps = max(1, int(duration * sim_hz))
        timeout_steps = max(1, int(no_contact_timeout * sim_hz))

        contact_steps = 0
        total_contact_checks = 0
        consecutive_no_contact_steps = 0

        start_z = self.position[2]
        dz = (new_z - start_z) / float(total_steps)

        for _ in range(total_steps):
            self.position[2] += dz
            p.changeConstraint(self.constraint_id, jointChildPivot=self.position, maxForce=350)
            contacts = p.getContactPoints(bodyA=self.id, bodyB=obj_id) or []
            if debug_contacts:
                print(f"[ThreeFinger Lift] contacts={len(contacts)}")

            has_contact = len(contacts) > 0
            if has_contact:
                contact_steps += 1
                consecutive_no_contact_steps = 0
            else:
                consecutive_no_contact_steps += 1
            total_contact_checks += 1

            if consecutive_no_contact_steps >= timeout_steps:
                return False

            p.stepSimulation()
            time.sleep(1.0 / 240.0)

        contact_ratio = contact_steps / float(max(1, total_contact_checks))
        return contact_ratio >= contact_ratio_threshold


# -----------------------
# Utility
# -----------------------
def compute_relative_pose(gripper_id: int, obj_id: int) -> Tuple[List[float], Tuple[float, float, float]]:
    g_pos, g_orn = p.getBasePositionAndOrientation(gripper_id)
    o_pos, o_orn = p.getBasePositionAndOrientation(obj_id)
    inv_o_pos, inv_o_orn = p.invertTransform(o_pos, o_orn)
    rel_pos, rel_orn = p.multiplyTransforms(inv_o_pos, inv_o_orn, g_pos, g_orn)
    rel_roll, rel_pitch, rel_yaw = p.getEulerFromQuaternion(rel_orn)
    return list(rel_pos), (rel_roll, rel_pitch, rel_yaw)

# Sampler
class BaseSampler(ABC):
    @abstractmethod
    def sample(self, samples: int, orientation_offset_euler: Tuple[float, float, float] | None = None) -> Tuple[List[List[float]], List[List[float]]]:
        raise NotImplementedError


class SphericalSampler(BaseSampler):
    def __init__(self, obj_center: List[float], radius: float) -> None:
        self.obj_center = obj_center
        self.radius = radius

    def sample(self, samples: int, orientation_offset_euler: Tuple[float, float, float] | None = None) -> Tuple[List[List[float]], List[List[float]]]:
        positions: List[List[float]] = []
        orientations: List[List[float]] = []
        q_offset = p.getQuaternionFromEuler(orientation_offset_euler) if orientation_offset_euler is not None else None

        for _ in range(samples):
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi / 2)
            x = self.radius * math.sin(phi) * math.cos(theta) + self.obj_center[0]
            y = self.radius * math.sin(phi) * math.sin(theta) + self.obj_center[1]
            z = self.radius * math.cos(phi) + self.obj_center[2]
            positions.append([x, y, z])

            dx = self.obj_center[0] - x
            dy = self.obj_center[1] - y
            dz = self.obj_center[2] - z
            dist_xy = math.sqrt(dx * dx + dy * dy)
            yaw = math.atan2(dy, dx)
            pitch = -math.atan2(dz, dist_xy)
            roll = random.uniform(-math.pi, math.pi)

            if q_offset is None:
                orientations.append([roll, pitch, yaw])
            else:
                q_sample = p.getQuaternionFromEuler([roll, pitch, yaw])
                _, q_hand = p.multiplyTransforms([0, 0, 0], q_sample, [0, 0, 0], q_offset)
                rpy_hand = p.getEulerFromQuaternion(q_hand)
                orientations.append(list(rpy_hand))
        return positions, orientations


# Evaluator
class BaseEvaluator(ABC):
    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    @abstractmethod
    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict:
        raise NotImplementedError


class PredictionEvaluator(BaseEvaluator):
    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "report": classification_report(y_true, y_pred, output_dict=True),
        }

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix (Test)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "test_confusion_matrix.png"))
        plt.close()

        with open(os.path.join(self.out_dir, "test_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def save_results_csv(self, y_true: List[int], y_pred: List[int]) -> str:
        df = pd.DataFrame({"pred": y_pred, "actual": y_true})
        path = os.path.join(self.out_dir, "test_results.csv")
        df.to_csv(path, index=False)
        return path

    def save_summary_md(self, metrics: Dict, samples: int) -> str:
        path = os.path.join(self.out_dir, "summary.md")
        with open(path, "w") as f:
            f.write("# Grasp Classifier Test Summary\n\n")
            f.write(f"- Samples: {samples}\n")
            f.write(f"- Accuracy: {metrics['accuracy']:.3f}\n")
            f.write(f"- F1: {metrics['f1']:.3f}\n")
            f.write(f"- Precision: {metrics['precision']:.3f}\n")
            f.write(f"- Recall: {metrics['recall']:.3f}\n")
        return path

# Orchestrator
class EvaluationOrchestrator:
    def __init__(
        self,
        model_path: str,
        out_dir: str,
        samples: int,
        gui: bool = True,
        radius: float = 0.7,  # reduced default radius for stability
        gripper_type: str = "three-finger",
        object_type: str = "cube",
        orient_offset_flip: bool = False,
        contact_thresh: float = 0.65,
        friction_lateral: float = 1.3,
        debug_contacts: bool = False,
    ) -> None:
        self.model_path = model_path
        self.out_dir = out_dir
        self.samples = samples
        self.gui = gui
        self.radius = radius
        self.gripper_type = gripper_type
        self.object_type = object_type
        self.orient_offset_flip = orient_offset_flip
        self.contact_thresh = contact_thresh
        self.friction_lateral = friction_lateral
        self.debug_contacts = debug_contacts

        self.env = EnvironmentManager(gui=self.gui)
        self.evaluator = PredictionEvaluator(out_dir=self.out_dir)
        self.model = None

        # Scene defaults per config
        if self.gripper_type == "three-finger" and self.object_type == "cube":
            self.obj_position = [0.6, 0.3, 0.05]
            self.scaling = 2.0
            self.stop_distance = 0.13
            self.use_pregrasp = True
            self.add_friction = True
        elif self.gripper_type == "two-finger" and self.object_type == "cube":
            self.obj_position = [0.6, 0.3, 0.025]
            self.scaling = 1.0
            self.stop_distance = 0.31
            self.use_pregrasp = False
            self.add_friction = False
        elif self.gripper_type == "two-finger" and self.object_type == "cylinder":
            self.obj_position = [0.6, 0.3, 0.05]
            self.scaling = 0.5
            self.stop_distance = 0.31
            self.use_pregrasp = False
            self.add_friction = True
        elif self.gripper_type == "three-finger" and self.object_type == "cylinder":
            self.obj_position = [0.6, 0.3, 0.1]
            self.scaling = 1.0
            self.stop_distance = 0.15
            self.use_pregrasp = True
            self.add_friction = True
        else:
            self.obj_position = [0.6, 0.3, 0.025]
            self.scaling = 1.0
            self.stop_distance = 0.31
            self.use_pregrasp = False
            self.add_friction = False

        self.object: BaseSceneObject = Cube(self.obj_position) if self.object_type == "cube" else Cylinder(self.obj_position)
        self.sampler: BaseSampler = SphericalSampler(obj_center=self.obj_position, radius=self.radius)

    def run(self) -> Dict:
        os.makedirs(self.out_dir, exist_ok=True)

        self.model = load(self.model_path)
        print(f"Loaded model: {self.model_path}")

        self.env.setup()
        obj_id = self.object.load(self.scaling)

        if self.add_friction:
            p.changeDynamics(obj_id, -1, lateralFriction=self.friction_lateral, spinningFriction=0.05, rollingFriction=0.05)

        # Orientation offset for three-finger
        base_offset = (0.0, math.pi / 2.0, 0.0)
        if self.orient_offset_flip:
            base_offset = (0.0, -math.pi / 2.0, 0.0)
        orientation_offset = base_offset if self.gripper_type == "three-finger" else None

        positions, orientations = self.sampler.sample(self.samples, orientation_offset_euler=orientation_offset)

        y_true: List[int] = []
        y_pred: List[int] = []

        for i in range(self.samples):
            pos = positions[i]
            orn = orientations[i]

            if self.gripper_type == "three-finger":
                gripper: BaseGripper = ThreeFingersGripper(position=pos, orientation_euler=tuple(orn))
            else:
                gripper = TwoFingersGripper(position=pos, orientation_euler=tuple(orn))

            gid = gripper.load()
            gripper.attach_fixed(offset=[0, 0, 0])

            gripper.open_gripper()
            if self.use_pregrasp and isinstance(gripper, ThreeFingersGripper):
                gripper.pregrasp()

            self.env.step(steps=40, hz=240.0)

            rel_pos, rel_orn = compute_relative_pose(gid, obj_id)
            feature = np.array([[rel_pos[0], rel_pos[1], rel_pos[2], rel_orn[0], rel_orn[1], rel_orn[2]]], dtype=np.float32)

            pred = int(self.model.predict(feature)[0])

            gripper.move_pickup(target_position=self.obj_position, stop_dist=self.stop_distance)
            gripper.close_gripper()
            self.env.step(steps=200, hz=240.0)

            success = gripper.move_up_classify(obj_id, contact_ratio_threshold=self.contact_thresh, debug_contacts=self.debug_contacts)
            actual = int(success)

            print(f"[{self.gripper_type} {self.object_type} {i+1}/{self.samples}] predicted={pred} actual={actual}")

            y_pred.append(pred)
            y_true.append(actual)

            p.removeBody(gid)
            p.resetBasePositionAndOrientation(obj_id, self.obj_position, [0, 0, 0, 1])
            self.env.step(steps=40, hz=240.0)

        metrics = self.evaluator.evaluate(y_true, y_pred)
        print(f"Test accuracy={metrics['accuracy']:.3f}, f1={metrics['f1']:.3f}, precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}")

        csv_path = self.evaluator.save_results_csv(y_true, y_pred)
        md_path = self.evaluator.save_summary_md(metrics, samples=self.samples)
        print(f"Saved test results to {csv_path}")
        print(f"Saved summary to {md_path}")

        p.removeBody(obj_id)

        return metrics


def main(model="models/best_model.pkl", out="eval", samples=10, gui="store_true", radius=0.7, 
         gripper="three-finger", object="cube", flip_offset="store_true", contact_thresh=0.65,
         friction=1.3, debug_contacts="store_true") -> Dict:
    orchestrator = EvaluationOrchestrator(
        model_path=model,
        out_dir=out,
        samples=samples,
        gui=gui,
        radius=radius,
        gripper_type=gripper,
        object_type=object,
        orient_offset_flip=flip_offset,
        contact_thresh=contact_thresh,
        friction_lateral=friction,
        debug_contacts=debug_contacts,
    )
    orchestrator.run()

if __name__ == "__main__":
    main()
