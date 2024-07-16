#!/usr/bin/env python
import sys
import os
# print(sys.path)
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
import logging
import pathlib
from onpolicy.config import get_config
from onpolicy.envs.overcooked.overcooked_environment import OvercookedEnvironment, MapSetting
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
import argparse
from gym_cooking.utils.replay import Replay
os.environ["WANDB_MODE"] = "offline"
from gym_cooking.play_test import MAP_SETTINGS
from agent.mind.agent import AgentSetting
from agent.gameplay import GamePlay
from copy import deepcopy
from agent.onpolicy.runner.shared.overcooked_runner import OvercookedRunner
def make_train_env(arglist, all_args, run_dir):
    # def get_env_fn(rank):
    #     def init_env():
    #         map_set = MapSetting(**MAP_SETTINGS[arglist.map])
    #         # agent_set = AgentSetting(arglist.agent, speed=2.5 if arglist.map != 'quick' else 3.5)
    #         env = OvercookedEnvironment(map_set)
    #         # env.seed(arglist.seed + rank * 1000)
    #         return env
    #
    #     return init_env
    # return DummyVecEnv([get_env_fn(0)])
    map_set = MapSetting(**MAP_SETTINGS[arglist.map])
    agent_set = AgentSetting(arglist.agent, speed=2.5 if arglist.map != 'quick' else 3.5)
    env = OvercookedEnvironment(map_set)
    replay = Replay()
    env.reset()
    game = GamePlay(env, replay, agent_set)
    # env.seed(arglist.seed + rank * 1000)

    replay['set_map'] = deepcopy(map_set)
    replay['set_agent'] = deepcopy(agent_set)
    replay['order_rand'] = deepcopy(env.order_scheduler.rand_recipe_list)
    replay['chg_rand'] = deepcopy(env.chg_rand_list)

    return game, env, replay


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=1, help="number of players")

    all_args = parser.parse_known_args(args)[0]
    #(all_args)
    return all_args

def parse_arguments():
    #parser = get_config()
    parser = argparse.ArgumentParser(
        "Overcooked argument parser")
    parser.add_argument(
        "--map", type=str,
        choices=['ring', 'bottleneck', 'partition', 'quick'], default='partition'
    )
    parser.add_argument(
        "--agent", type=str,
        choices=['HLA', 'SMOA', 'FMOA', 'NEA'], default='FMOA'
    )

    return parser.parse_args()
parser = get_config()
all_args = parse_args([], parser)
arg_list = parse_arguments()


    # run dir
run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
if not run_dir.exists():
    os.makedirs(str(run_dir))

    # wandb
if all_args.use_wandb:
    run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
else:
    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
                              str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
        all_args.user_name))
# games = []
# envss = []
# replays = []
# for _ in range(2):
game, envs, replay = make_train_env(arg_list, all_args, run_dir)
# games.append(game)
# envss.append(envs)
#     replays.append(replay)

config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": None,
        "num_agents": 1,
        "device": 'cuda',
        "run_dir": run_dir
    }
runner = OvercookedRunner(config)
runner1 = OvercookedRunner(config)

