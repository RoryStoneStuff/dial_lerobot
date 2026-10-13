"""Flat-floor distances derived from the robot's actual foot collision capsules."""
import mujoco
import numpy as np


def foot_capsules(model):
    """Return left/right geometry indices; fail explicitly for unsupported models."""
    ids = []
    for side in ('left', 'right'):
        foot = [i for i in range(model.ngeom)
                if (model.geom(i).name or '').startswith(side + '_foot')
                and (model.geom(i).name or '').endswith('_collision')]
        if not foot or any(model.geom_type[i] != mujoco.mjtGeom.mjGEOM_CAPSULE for i in foot):
            raise ValueError('Foot distance requires named foot collision capsules')
        ids.append(foot)
    if len(ids[0]) != len(ids[1]):
        raise ValueError('Expected matching numbers of left/right foot capsules')
    ground = model.geom('ground').id
    if model.geom_type[ground] != mujoco.mjtGeom.mjGEOM_PLANE or not np.allclose(model.geom_pos[ground], 0) or not np.allclose(model.geom_quat[ground], [1, 0, 0, 0]):
        raise ValueError('Foot distance currently supports the world z=0 ground plane')
    return np.asarray(ids)


def capsule_bottoms(model, data, ids, xp=np):
    """Signed distance from each capsule's lowest surface to the flat floor.

    MuJoCo capsules extend along their local z axis. This remains correct when
    the foot rolls/pitches; a foot-frame marker alone does not capture this.
    """
    rotation = xp.asarray(data.geom_xmat).reshape((-1, 3, 3))[ids]
    size = xp.asarray(model.geom_size)[ids]
    return (xp.asarray(data.geom_xpos)[ids, 2]
            - xp.abs(rotation[..., 2, 2]) * size[..., 1] - size[..., 0])
