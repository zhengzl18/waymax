import os
import numpy as np
import mediapy
from matplotlib import pyplot as plt
from tqdm import tqdm
import dataclasses
import jax
# jax.config.update('jax_enable_x64', True)

from waymax import config as _config, dynamics
from waymax import dataloader
from waymax import datatypes
from waymax import visualization
from waymax.env import PlanningAgentEnvironment, MultiAgentEnvironment
from waymax.datatypes.observation import observation_from_state
from waymax.visualization import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
mediapy.set_show_save_dir('/home/zhengzhilong')

config = dataclasses.replace(_config.WOD_1_2_0_TRAINING, repeat=1, max_num_objects=32, include_sdc_paths=False, num_paths=1, num_points_per_path=5, batch_dims=(5,))
data_iter = dataloader.simulator_state_generator(config=config)
scenario = next(data_iter)

# scenario_ids = []
# for scenario in data_iter:
#     scenario_ids.append(scenario.scenaro_id)
# scenario_ids = np.concatenate(scenario_ids)
# print('scenario_ids:', scenario_ids)

# for i in range(scenario.shape[0]):
#     img = visualization.plot_simulator_state(scenario, use_log_traj=True, highlight_obj=_config.ObjectType.SDC, batch_idx=i)
#     mediapy.show_image(img, title=f'simulator_state_{i}')

# imgs = []

# state = scenario
# for _ in range(scenario.remaining_timesteps):
#     state = datatypes.update_state_by_log(state, num_steps=1)
#     img = visualization.plot_simulator_state(state, use_log_traj=True)
#     imgs.append(img)

# mediapy.show_video(imgs, fps=10, title="simulator_state")
dynamics_model = dynamics.StateDynamics()

# Expect users to control all valid object in the scene.
env = PlanningAgentEnvironment(
    dynamics_model=dynamics_model,
    config=dataclasses.replace(
        _config.EnvironmentConfig(),
        max_num_objects=32,
        controlled_object=_config.ObjectType.SDC,
    ),
)
# env = MultiAgentEnvironment(
#     dynamics_model=dynamics_model,
#     config=dataclasses.replace(
#         _config.EnvironmentConfig(),
#         max_num_objects=32,
#         controlled_object=_config.ObjectType.SDC,
#     ),
# )

state = env.reset(scenario)
obs = observation_from_state(state, obs_num_steps=10, coordinate_frame=_config.CoordinateFrame.SDC)
print('Initial state:', state)