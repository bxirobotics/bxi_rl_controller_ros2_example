"""MuJoCo-backed self-collision checks for commanded Elf3 poses."""

from pathlib import Path

import mujoco

from .elf3 import JOINT_NAMES, validate_joint_vector


class CollisionGuard:
    """Check simplified contacts and non-adjacent visual-mesh intersections."""

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.reference_qpos = self.data.qpos.copy()
        self.qpos_addresses = {}
        self.joint_body_ids = {}
        for name in JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise ValueError("MuJoCo model is missing joint: " + name)
            self.qpos_addresses[name] = int(
                self.model.jnt_qposadr[joint_id]
            )
            self.joint_body_ids[name] = int(
                self.model.jnt_bodyid[joint_id]
            )
        self.visual_geoms = [
            geom_id
            for geom_id in range(self.model.ngeom)
            if self.model.geom_group[geom_id] == 2
        ]
        self.visual_pairs_by_joint = {
            name: self._visual_cross_pairs(self.joint_body_ids[name])
            for name in JOINT_NAMES
        }
        self.all_visual_pairs = tuple(
            {
                (geom_1, geom_2): (geom_1, geom_2, body_1, body_2)
                for pairs in self.visual_pairs_by_joint.values()
                for geom_1, geom_2, body_1, body_2 in pairs
            }.values()
        )

    def _is_descendant(self, root_body, body):
        current = body
        while current != 0 and current != root_body:
            current = int(self.model.body_parentid[current])
        return current == root_body

    def _body_tree_distance(self, body_1, body_2):
        ancestors = {}
        current = body_1
        distance = 0
        while True:
            ancestors[current] = distance
            if current == 0:
                break
            current = int(self.model.body_parentid[current])
            distance += 1
        current = body_2
        distance = 0
        while current not in ancestors:
            if current == 0:
                return 1000
            current = int(self.model.body_parentid[current])
            distance += 1
        return distance + ancestors[current]

    def _visual_cross_pairs(self, moving_root_body):
        pairs = []
        for offset, geom_1 in enumerate(self.visual_geoms):
            body_1 = int(self.model.geom_bodyid[geom_1])
            moving_1 = self._is_descendant(moving_root_body, body_1)
            for geom_2 in self.visual_geoms[offset + 1:]:
                body_2 = int(self.model.geom_bodyid[geom_2])
                moving_2 = self._is_descendant(moving_root_body, body_2)
                if moving_1 == moving_2:
                    continue
                # Adjacent housings (including the zero-length wrist/ankle
                # intermediate links) intentionally overlap in the CAD.
                if self._body_tree_distance(body_1, body_2) <= 2:
                    continue
                pairs.append((geom_1, geom_2, body_1, body_2))
        return pairs

    def _set_pose(self, joint_positions):
        positions = validate_joint_vector("collision pose", joint_positions)
        self.data.qpos[:] = self.reference_qpos
        for index, name in enumerate(JOINT_NAMES):
            self.data.qpos[self.qpos_addresses[name]] = positions[index]
        mujoco.mj_forward(self.model, self.data)

    def collisions(self, joint_positions):
        """Return penetrating non-floor contacts from model collision geoms."""
        self._set_pose(joint_positions)
        result = []
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if contact.dist > 0.0:
                continue
            geom_1 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
            )
            geom_2 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
            )
            if "floor" in (geom_1, geom_2):
                continue
            result.append((geom_1, geom_2, float(contact.dist)))
        return result

    def visual_mesh_collisions(self, joint_positions, active_joint_name):
        """Check one moving link subtree against non-adjacent visual meshes."""
        self._set_pose(joint_positions)
        result = []
        for geom_1, geom_2, body_1, body_2 in self.visual_pairs_by_joint[
            active_joint_name
        ]:
            distance = mujoco.mj_geomDistance(
                self.model, self.data, geom_1, geom_2, 0.0, None
            )
            if distance >= -1.0e-5:
                continue
            result.append(
                (
                    mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, body_1
                    ),
                    mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, body_2
                    ),
                    float(distance),
                )
            )
            break
        return result

    def visual_mesh_collisions_any(self, joint_positions):
        """Check every relevant non-adjacent visual-mesh pair once."""
        self._set_pose(joint_positions)
        for geom_1, geom_2, body_1, body_2 in self.all_visual_pairs:
            distance = mujoco.mj_geomDistance(
                self.model, self.data, geom_1, geom_2, 0.0, None
            )
            if distance >= -1.0e-5:
                continue
            return [
                (
                    mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, body_1
                    ),
                    mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, body_2
                    ),
                    float(distance),
                )
            ]
        return []
