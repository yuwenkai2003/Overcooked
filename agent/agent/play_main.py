import os
import sys
# print(sys.path)
from agent.mind.agent import AgentSetting
from agent.gameplay import GamePlay
from agent.onpolicy.config import get_config
from agent.onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
from agent.onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from agent.onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
from gym_cooking.utils.gui import *
from gym_cooking.utils.replay import Replay
from onpolicy.envs.overcooked.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.play_test import MAP_SETTINGS
from copy import deepcopy

import argparse
from datetime import datetime
from pathlib import Path
num_agents = 1
policy = None
trainer = None
buffer = None

def parse_arguments():
    #parser = get_config()
    parser = argparse.ArgumentParser(
        "Overcooked argument parser")
    parser.add_argument(
        "--map", type=str,
        choices=['ring', 'bottleneck', 'partition', 'quick'], default='ring'
    )
    parser.add_argument(
        "--agent", type=str,
        choices=['HLA', 'SMOA', 'FMOA', 'NEA'], default='HLA'
    )

    return parser.parse_args()


def init_env_replay(arglist):
    map_set = MapSetting(**MAP_SETTINGS[arglist.map])
    agent_set = AgentSetting(arglist.agent, speed=2.5 if arglist.map != 'quick' else 3.5)
    
    replay = Replay()

    env = OvercookedEnvironment(map_set)
    env.reset()

    game = GamePlay(env, replay, agent_set)

    replay['set_map'] = deepcopy(map_set)
    replay['set_agent'] = deepcopy(agent_set)
    replay['order_rand'] = deepcopy(env.order_scheduler.rand_recipe_list)
    replay['chg_rand'] = deepcopy(env.chg_rand_list)

    return game, env, replay


if __name__ == '__main__':
    arglist = parse_arguments()

    # initialize replay
    for _ in range(2):
        game, env, replay = init_env_replay(arglist)

        # play
        # print(type(game.env.trainer))
        ok = game.on_execute()
    print(replay['order_result'])
    repdir = Path(__file__).resolve().parent / 'replay'
    # replay.save(repdir / f'{arglist.map}-{arglist.agent}-{datetime.now().strftime("%Y%m%d_%H%M%S")}.rep')
    output_dir = repdir / f'{arglist.map}-{arglist.agent}-{datetime.now().strftime("%Y%m%d_%H%M%S")}.rep'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Save replay
    replay.save(output_dir)
    # record
    # if ok is True:
    #     popup_box("Game End!")
    # else:
    #     popup_box("Game Failed!")
