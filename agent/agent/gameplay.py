# modules for game
from gym_cooking.misc.game.game import Game
from gym_cooking.misc.game.utils import *
from gym_cooking.utils.gui import popup_text
from gym_cooking.utils.replay import Replay
from agent.executor.low import EnvState
from agent.mind.agent import get_agent, AgentSetting
from agent.onpolicy.utils.shared_buffer import SharedReplayBuffer
# helpers
import pygame
import threading
import queue
import time
from copy import deepcopy as dcopy

# import vosk
# from vosk import Model, KaldiRecognizer
# import pyaudio
# import os

import numpy as np
# def speak(text: str):
#     def _speak(text: str):
#         from win32com.client import Dispatch
#         speaker = Dispatch("SAPI.SpVoice")
#         speaker.Rate = 5
#         speaker.Speak(text)
#         pass
#
#     threading.Thread(target=_speak, args=(text,)).start()

class GamePlay(Game):
    #execute_lock = threading.Lock()
    def __init__(self, env, replay: Replay, agent_set: AgentSetting):
        Game.__init__(self, env, play=True)
        # print("sl")
        self.replay = replay
        self.agent_set = agent_set

        # fps of human and ai
        self.fps = 10
        self.fps_ai = agent_set.speed

        self.idx_human = 0
        self.ai = get_agent(self.agent_set, self.replay)
        self.ai1 = get_agent(self.agent_set, self.replay)
        # concurrent control variables
        self._q_control = queue.Queue()  # receive
        self._q_env = queue.Queue()
        self._q_ai = queue.Queue()
        self._success = False

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._q_control.put(('Quit', {}))

        elif event.type == pygame.KEYDOWN:
            if event.key in KeyToTuple.keys():
                # Control
                action_dict = {agent.name: (0, 0) for agent in self.sim_agents}
                action = KeyToTuple[event.key]
                action_dict[self.current_agent.name] = action
                self._q_env.put(
                    ('Action', {"agent": "human", "action": action}))
                self._q_ai.put(
                    ('Action', {"agent": "human", "action": action}))

            if pygame.key.name(event.key) == "space":
                self._q_env.put(('Pause', {}))

                s = popup_text("Say to AI:")

                if s is not None:
                    self._q_env.put(('ChatIn', {"chat": s, "mode": "text"}))
                    self._q_ai.put(('Chat', dict(chat=s)))

                self._q_env.put(('Continue', {}))

    def _get_obs_from_env(self, env: EnvState):
        return np.concatenate([np.random.rand(1) for _ in range(0, 4)])

    def _get_move(self, actions):
        available_moves = [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
        return available_moves[actions]

    def _run_env(self):
        # self.warmup()
        seconds_per_step = 1 / self.fps
        idx_human = self.idx_human
        paused = 0
        chat_in, chat_out = "", ""
        last_t = time.time()
        action_dict = {agent.name: None for agent in self.sim_agents}

        self.on_render(paused=paused)
        info = self.env.get_ai_info()
        e = EnvState(world=info['world'],
                     agents=info['sim_agents'],
                     agent_idx=0,
                     order=info['order_scheduler'],
                     event_history=info['event_history'],
                     time=info['current_time'],
                     chg_grid=info['chg_grid'],
                     result=info['result0'])
        e1 = EnvState(world=info['world'],
                      agents=info['sim_agents'],
                      agent_idx=1,
                      order=info['order_scheduler'],
                      event_history=info['event_history'],
                      time=info['current_time'],
                      chg_grid=info['chg_grid'],
                      result=info['result1'])
        self._q_ai.put_nowait(('Env', {"EnvState": e, "EnvState1": e1}))
        # self._q_ai.put_nowait(('Env', {"EnvState": e1}))
        # print(e.self_pos)
        # print(e.order)
        while True:
            while not self._q_env.empty():
                event = self._q_env.get_nowait()
                event_type, args = event
                if event_type == 'Action':
                    #if args['agent'] == "human":
                    action_dict[self.sim_agents[0].name] = args['action']
                        # action_dict[self.sim_agents[0].name] = args['action']
                    #elif idx_human == 0:
                        # action_dict[self.sim_agents[1].name] = args['action']
                    action_dict[self.sim_agents[1].name] = args['action1']
                elif event_type == 'Pause':
                    paused += 1
                elif event_type == 'Continue':
                    paused -= 1
                elif event_type == 'ChatIn':
                    chat_in = f"User Input: [{args['mode']}]\n\n" + \
                        args['chat']
                    chat_out = ""
                elif event_type == 'ChatOut':
                    chat_out = "AI Output:\n\n" + args['chat']
            if not paused:
                # values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(self.env.t)
                ad = {k: v if v is not None else (
                    0, 0) for k, v in action_dict.items()}
                # print(actions)
               # print(ad)
                # ad['agent-1'] = self._get_move(actions.item())
                # ad = {'agent-1': (0, 1), 'agent-2': (0, 0)}
                self.replay.log(
                    'env.step', {'action_dict': ad, 'passed_time': seconds_per_step})
                obs, rewards, done, infos = self.env.step(ad, passed_time=0.1)
                # print(infos)
                #obs, rewards, done, infos = self.env.step(ad, passed_time=seconds_per_step)
                # done = np.all([done])
                # data = obs, rewards, [done], infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                # self.insert(data)
                # if self.env.t % 10 == 0:
                #     self.compute()
                #     # train_infos = self.train()
                if done:
                    self._success = True
                    self._q_control.put(('Quit', {}))
                    # self.compute()
                    # train_infos = self.train()
                    return
                info = self.env.get_ai_info()
                e = EnvState(world=info['world'],
                             agents=info['sim_agents'],
                             agent_idx=0,
                             order=info['order_scheduler'],
                             event_history=info['event_history'],
                             time=info['current_time'],
                             chg_grid=info['chg_grid'],
                             result=info['result0'])
                e1 = EnvState(world=info['world'],
                              agents=info['sim_agents'],
                              agent_idx=1,
                              order=info['order_scheduler'],
                              event_history=info['event_history'],
                              time=info['current_time'],
                              chg_grid=info['chg_grid'],
                              result=info['result1'])
                # print(e.self_pos)
                # print(e.order)
                if action_dict[self.sim_agents[1 - self.idx_human].name] is not None:
                    # print(250)
                    self._q_ai.put(('Env', {"EnvState": dcopy(e), "EnvState1": dcopy(e1)}))
                    # self._q_ai.put(('Env', {"EnvState": dcopy(e1)}))
                action_dict = {agent.name: None for agent in self.sim_agents}

            sleep_time = max(seconds_per_step - (time.time() - last_t), 0)
            last_t = time.time()
            time.sleep(sleep_time)

            chat = chat_in + '\n\n' + chat_out

            if not paused:
                self.replay.log('on_render', {'paused': paused, 'chat': chat})
            self.on_render(paused=paused, chat=chat)

    def _run_ai(self):
        time_per_step = 1 / self.fps_ai
        time_last = time.time()
        human_act = True
        env = None
        env1 = None
        env_update = False
        chat = ''
        chat1 = ''
        # self.warmup()
        while True:
            event = self._q_ai.get()
            while True:
                event_type, args = event
                if event_type == 'Env':
                    env = args['EnvState']
                    env1 = args['EnvState1']
                    env_update = True
                elif event_type == 'Chat':
                    chat = args['chat']
                    chat1 = args['chat1']
                elif event_type == "Action":
                    human_act = True
                elif event_type == "Quit":
                    return
                if not self._q_ai.empty():
                    event = self._q_ai.get()
                else:
                    break

            if chat != '':
                self.ai.high_level_infer(env, chat)
                chat = ''

            if chat1 != '':
                self.ai1.high_level_infer(env1, chat1)
                chat1 = ''

            if env_update:
                #print(1)
                # print(type(env))
                # from agent.play import runner
                # # values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = runner.collect(0)
                # obs = self._get_obs_from_env(env)
                # share_obs = obs
                # step = runner.buffer.step
                # runner.buffer.obs[step] = obs.copy()
                # runner.buffer.share_obs[step] = share_obs.copy()
                # values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = runner.collect(step)
                # dones = np.array([[False]])
                # rnn_states[(dones == True)] = np.zeros(((dones == True).sum(), runner.recurrent_N, runner.hidden_size),
                #                                        dtype=np.float32)
                # rnn_states_critic[(dones == True)] = np.zeros(
                #     ((dones == True).sum(), *runner.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
                # masks = np.ones((runner.n_rollout_threads, runner.num_agents, 1), dtype=np.float32)
                # masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
                # action = self._get_move(actions.item())
                # runner.buffer.rnn_states[step + 1] = rnn_states.copy()
                # runner.buffer.rnn_states_critic[step + 1] = rnn_states_critic.copy()
                # runner.buffer.actions[step] = actions.copy()
                # runner.buffer.action_log_probs[step] = action_log_probs.copy()
                # runner.buffer.value_preds[step] = values.copy()
                # runner.buffer.masks[step + 1] = masks.copy()
                # #state, move, msg = self._task[0](env)
                # step += 1
                # runner.buffer.step = step
                move, chat_ret = self.ai(env)
                move1, chat_ret1 = self.ai1(env1)
                # values, move, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(self.env.t)
                # print(move, chat_ret)
                # sleep
                sleep_time = max(time_per_step - (time.time() - time_last), 0)
                time.sleep(sleep_time)
                time_last = time.time()

                if chat_ret:
                    self._q_env.put(('ChatOut', {"chat": chat_ret, "chat1": chat_ret1}))
                self._q_env.put(('Action', {"agent": "ai", "action": move, "agent1": "ai", "action1": move1}))
                human_act = False
                env_update = False

    # def _run_listen(self):
    #     ena_listen = False
    #     self._q_env.put(('Setting', {"ena_listen": ena_listen}))

    #     dir_path = os.path.dirname(os.path.realpath(__file__))
    #     model_path = dir_path + r"/vosk"
    #     model = Model(model_path)
    #     recognizer = KaldiRecognizer(model, 16000)

    #     mic = pyaudio.PyAudio()
    #     stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    #     stream.start_stream()

    #     while True:
    #         while not self._q_listen.empty():
    #             event = self._q_listen.get_nowait()
    #             event_type, args = event
    #             if event_type == 'Quit':
    #                 return
    #             elif event_type == 'ListenSwitch':
    #                 ena_listen = not ena_listen
    #                 self._q_env.put(('Setting', {"ena_listen": ena_listen}))

    #         if not ena_listen:
    #             time.sleep(0.1)
    #             continue

    #         data = stream.read(10000)

    #         if recognizer.AcceptWaveform(data):
    #             text = recognizer.Result()
    #             s = text[14:-3]
    #             if s is not None and len(s) >= 4:
    #                 self._q_env.put(('ChatIn', {"chat": s, "mode": "speech"}))
    #                 self._q_ai.put(('Chat', dict(chat=s)))

    def _run_human(self):
        while True:
            for event in pygame.event.get():
                self.on_event(event)
            if not self._q_control.empty():
                event, args = self._q_control.get_nowait()
                if event == 'Quit':
                    self._q_ai.put(('Quit', {}))
                    return

    def on_execute(self):
        # with GamePlay.execute_lock:
            if self.on_init() == False:
                exit()

            thread_env = threading.Thread(target=self._run_env, daemon=True)
            thread_ai = threading.Thread(target=self._run_ai, daemon=True)
            # thread_listen = threading.Thread(target=self._run_listen, daemon=True)
            thread_env.start()
            thread_ai.start()
            # thread_listen.start()
            # thread_env.join()
            # thread_ai.join()
            self._run_human()

            # clean up
            self.on_cleanup()

            # save history
            if hasattr(self.ai, "_lock"):
                self.ai._lock.acquire()
            if hasattr(self.ai, "_int_hist"):
                self.replay['int_hist'] = self.ai._int_hist
            if hasattr(self.ai, "_llm_hist"):
                self.replay['llm_hist'] = self.ai._llm_hist
            if hasattr(self.ai, "_mov_hist"):
                self.replay['mov_hist'] = self.ai._mov_hist
            # log recipy infos
            self.replay['order_result'] = dict(
                success=self.env.order_scheduler.successful_orders,
                fail=self.env.order_scheduler.failed_orders,
                reward=self.env.order_scheduler.reward
            )
            if hasattr(self.ai, "_lock"):
                self.ai._lock.release()
            # thread_env.join()
            # thread_ai.join()
            return self._success
