"""Reproducible, fall-terminated stance ablations (run from the repository root)."""
import argparse
from dataclasses import replace
from pathlib import Path
import subprocess
import jax
from dial_lerobot.config import Config
from dial_lerobot.evaluator import Evaluator, write_json
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--case', default='legacy')
p.add_argument('--seconds', type=float, default=3)
p.add_argument('--seeds', type=int, nargs='+', default=[45,46])
p.add_argument('--dt', type=float, choices=[.002,.005,.01], default=.005)
p.add_argument('--samples', type=int, default=128)
p.add_argument('--contact', type=float, default=None)
p.add_argument('--slip', type=float, default=None)
p.add_argument('--delay', type=float, default=None)
p.add_argument('--duty', type=float, default=None)
p.add_argument('--stand', action='store_true')
a=p.parse_args()
cfg=Config()
cost_changes={'target_vx':.3}
for arg,field in [('contact','contact_weight'),('slip','stance_slip_weight'),('delay','gait_start_delay'),('duty','duty')]:
 if getattr(a,arg) is not None: cost_changes[field]=getattr(a,arg)
if a.stand: cost_changes.update(target_vx=0.,swing_height=0.,hip_amplitude=0.,knee_amplitude=0.)
cfg=replace(cfg, robot=replace(cfg.robot,timestep=a.dt),
 mpc=replace(cfg.mpc,plan_horizon=.3,num_knots=5,num_samples=a.samples,noise_level=.03,temperature=.05,beta_horizon=.9,control_decimation=round(.01/a.dt)),
 cost=replace(cfg.cost,**cost_changes),
 evaluation=replace(cfg.evaluation,episode_steps=round(a.seconds/.01),height_jitter=0.,num_episodes=1))
e=Evaluator(cfg)
folder=Path('results/stance_contact_audit')/a.case
if folder.exists() and any(folder.iterdir()):
 raise ValueError(f'Experiment folder already contains results: {folder}')
folder.mkdir(parents=True,exist_ok=True)
write_json(folder/'experiment.json', {'args':vars(a),'config':cfg.to_dict(), 'git_head':subprocess.check_output(['git','-c','safe.directory='+str(Path.cwd()),'rev-parse','HEAD'],text=True).strip(),'git_diff':subprocess.check_output(['git','-c','safe.directory='+str(Path.cwd()),'diff'],text=True)})
print('Devices:',jax.devices(), 'Case:',a.case, 'Compiling...', flush=True)
w=e.warmup()
print('Warmup seconds:',w,flush=True)
from dial_lerobot.assets.model_builder import model_xml
(folder/'model.xml').write_text(model_xml(e.theta,cfg.robot))
write_json(folder/'metadata.json', e.metadata())
summaries=[]
for seed in a.seeds:
 s,arrays=e.episode(seed,progress=True)
 np.savez_compressed(folder/f'seed_{seed}.npz',**arrays)
 summaries.append(s)
 write_json(folder/'results.json',{'warmup_seconds':w,'episodes':summaries})
 print(s,flush=True)
