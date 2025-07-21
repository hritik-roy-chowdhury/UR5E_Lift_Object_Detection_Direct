# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-UR5E-Lift-Object-Detection-Direct-Direct-v0",
    entry_point=f"{__name__}.ur5e_lift_object_detection_direct_env:UR5ELiftObjectDetectionDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5e_lift_object_detection_direct_env_cfg:UR5ELiftObjectDetectionDirectEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)