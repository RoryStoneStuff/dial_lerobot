"""Reconstruct foot geometry/velocities from saved states, without running MPC."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from dial_lerobot.assets.foot_geometry import foot_capsules, capsule_bottoms


def analyze(folder):
    folder=Path(folder)
    meta=json.loads((folder/'metadata.json').read_text())
    c=meta['config']['cost']
    m=mujoco.MjModel.from_xml_path(str(folder/'model.xml'))
    d=mujoco.MjData(m); ids=foot_capsules(m)
    summary={s['seed']:s for s in json.loads((folder/'results.json').read_text())['episodes']}
    output=[]
    for file in sorted(folder.glob('seed_*.npz')):
        a=dict(np.load(file)); t=a['time']; gaps=[]; velocities=[]; flat=[]
        if not len(t):
            print(f'{file}: empty prefix; see results.json for the failure',flush=True)
            continue
        for q,v,u in zip(a['qpos'],a['qvel'],a['ctrl']):
            d.qpos[:]=q; d.qvel[:]=v; d.ctrl[:]=u
            mujoco.mj_forward(m,d)
            heights=capsule_bottoms(m,d,ids); gaps.append(heights.min(axis=1));flat.append(heights.max(axis=1)-heights.min(axis=1))
            footv=[]
            for side in ('left','right'):
                vel=np.zeros(6)
                mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_SITE,m.site('foot_'+side).id,vel,0)
                footv.append(vel[3:])
            velocities.append(footv)
        gaps=np.asarray(gaps);velocities=np.asarray(velocities);flat=np.asarray(flat)
        phase=(np.maximum(t[:,None]-c.get('gait_start_delay',0),0)*c['cadence']+[0,.5])%1
        stance=(phase<c['duty'])|(t[:,None]<c.get('gait_start_delay',0))
        if c['swing_height']==c['hip_amplitude']==c['knee_amplitude']==0: stance[:]=True
        proximity=(gaps<=.002)&(gaps>=-.01)
        mask=stance&proximity
        speed=np.linalg.norm(velocities[:,:,:2],axis=-1)
        foot_xy = a['feet'][:, :, :2]
        foot_step_delta = np.diff(foot_xy, axis=0)
        foot_dx = foot_step_delta[:, :, 0]
        foot_travel = np.linalg.norm(foot_step_delta, axis=-1)
        seed=int(file.stem.split('_')[-1]); s=summary[seed]
        metrics={'case':folder.name,'seed':seed,'sim_seconds':s['sim_seconds'], 'fell':s['fell'],
                 'distance_x_m':s['distance_x_m'], 'world_mean_vx':s['distance_x_m']/s['sim_seconds'],
                 'wall_seconds':s['wall_seconds'],
                 'integrated_cost':s['total_running_cost']*meta['config']['robot']['timestep'],
                 'stance_near_ground_fraction':float(proximity[stance].mean()),
                 'stance_slip_rms_m_s':float(np.sqrt(np.mean(speed[mask]**2))) if mask.any() else None,
                 'swing_above_15mm_fraction':float((gaps[~stance]>.015).mean()) if (~stance).any() else None,
                 'max_clearance_left_right_m':gaps.max(axis=0).tolist(),
                 'stance_mean_sole_height_spread_m':float(flat[stance].mean()),
                 'foot_total_travel_m':float(foot_travel.sum()),
                 'foot_forward_travel_m':float(np.maximum(foot_dx, 0).sum()),
                 'foot_backward_travel_m':float(np.maximum(-foot_dx, 0).sum()),
                 'foot_net_displacement_m':float(np.linalg.norm(foot_xy[-1] - foot_xy[0], axis=-1).mean()),
                 'final_xy_m':a['qpos'][-1,:2].tolist()}
        output.append(metrics)
        print(json.dumps(metrics),flush=True)
        np.savez_compressed(folder/f'contact_{seed}.npz',time=t,gaps=gaps,foot_velocity=velocities,stance=stance,proximity=proximity)
    (folder/'contact_analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folders',nargs='+');a=p.parse_args()
    for f in a.folders: analyze(f)
