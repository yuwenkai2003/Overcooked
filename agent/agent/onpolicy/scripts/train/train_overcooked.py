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
                        default=2, help="number of players")

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

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    arg_list = parse_arguments()
    print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
    all_args.use_recurrent_policy = False
    all_args.use_naive_recurrent_policy = False

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

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

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    from agent.play import game, envs, replay
    # config = {
    #     "all_args": all_args,
    #     "envs": envs,
    #     "eval_envs": eval_envs,
    #     "num_agents": num_agents,
    #     "device": device,
    #     "run_dir": run_dir
    # }
    # env ini
    num_agents = all_args.num_agents

    # run experiments
    # from onpolicy.runner.shared.overcooked_runner import OvercookedRunner as Runner
    from agent.play import runner, runner1
    # runner = Runner(config)
    # runner.run()
    # for k in range(2):
    #     games[k].on_execute()
        # runner.compute()
        # runner.train()
        # # post process
        # envss[k].close()
    import threading
    # thread_0 = threading.Thread(target=games[0].on_execute, daemon=True)
    # thread_0.start()
    # thread_0.join()
    # # thread_1 = threading.Thread(target=games[1].on_execute, daemon=True)
    # # thread_1.start()
    # # thread_1.join()
    # games[0].on_execute()
    # games[1].on_execute()
    # from agent.executor.low import set_step
    # runner.restore('')
    # policy_actor_state_dict = torch.load(str(8089) + '/actor.pt')
    # runner.policy.actor.load_state_dict(policy_actor_state_dict)
    # policy_critic_state_dict = torch.load(str(8089) + '/critic.pt')
    # runner.policy.critic.load_state_dict(policy_critic_state_dict)
    # policy_actor_state_dict1 = torch.load(str(10089) + '/actor.pt')
    # runner1.policy.actor.load_state_dict(policy_actor_state_dict1)
    # policy_critic_state_dict1 = torch.load(str(10089) + '/critic.pt')
    # runner1.policy.critic.load_state_dict(policy_critic_state_dict1)
    for _ in range(100):
        # set_step()
        print(_)
        game, env, replay = make_train_env(arg_list, all_args, run_dir)
        runner.buffer.step = 0
        runner1.buffer.step = 0
        game.on_execute()
        runner.compute()
        runner1.compute()
        runner.train()
        runner1.train()
        runner.save(episode=_+12000)
        runner1.save(episode=_+13000)
    #game.on_execute()
    # # for _ in range(runner.buffer.step):
    # #     print(runner.buffer.actions[_], runner.buffer.rewards[_])
    # runner.compute()
    # runner.train()
    if all_args.use_wandb:
        run.finish()


if __name__ == "__main__":
    main(sys.argv[1:])