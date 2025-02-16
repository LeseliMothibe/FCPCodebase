from scripts.gyms.Basic_Run import Basic_Run, Train as BaseTrain
from stable_baselines3 import PPO
import numpy as np

from agent.Base_Agent import Base_Agent as Agent
from behaviors.custom.Step.Step import Step
from world.commons.Draw import Draw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from time import sleep
import os, gym
import numpy as np

import time
import matplotlib.pyplot as plt
import csv
import cloudpickle
import dill

class Basic_Stop(Basic_Run):
    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw, env_id, running_model_path):
        super().__init__(ip, server_p, monitor_p, r_type, enable_draw, env_id)
        self.running_model = PPO.load(running_model_path, device='cpu')
        self.max_running_steps = 100  # Steps to simulate running before stopping phase

        self.prev_speed = 0.0
        self.stop_flag = 0
        self.stopping_start_pos = self.player.world.robot.cheat_abs_pos[0]
        self.stopping_start_pos_y = self.player.world.robot.cheat_abs_pos[1]

        self.initial_speed = 0.09
        self.last_joint_speeds = self.player.world.robot.joints_speed[2:22].copy()


    def reset(self):
        # Standard reset (beam and stabilize)
        obs = super().reset()
        
        # Run the pre-trained running model to initialize motion
        self.step_counter = 0  # Reset step counter for stopping phase

        self.stopping_start_pos = self.player.world.robot.cheat_abs_pos[0]
        self.stopping_start_pos_y = self.player.world.robot.cheat_abs_pos[1]

        ''''

        if self.episode_number == 300:

            print("********************************************************************************")
            print("********************************************************************************")
            print("*************************************M O D E L 2******************************************")
            print("********************************************************************************")
            print("********************************************************************************")

            self.running_model = PPO.load('./scripts/gyms/logs/Basic_Run_R1_254/best_model', device='cpu') # medium run == 47 minutes training

        elif self.episode_number == 600:

            print("********************************************************************************")
            print("********************************************************************************")
            print("*************************************M O D E L 3******************************************")
            print("********************************************************************************")
            print("********************************************************************************")

            self.running_model = PPO.load('./scripts/gyms/logs/Basic_Run_R1_255/best_model', device='cpu') # fastish run == 45 hour traning  

        elif self.episode_number == 900:

            print("********************************************************************************")
            print("********************************************************************************")
            print("*************************************M O D E L 4******************************************")
            print("********************************************************************************")
            print("********************************************************************************")

            self.running_model = PPO.load('./scripts/gyms/logs/Basic_Run_R1_257/best_model', device='cpu') # fastisher run == 1750000  traning  

        elif self.episode_number == 1200:

            print("********************************************************************************")
            print("********************************************************************************")
            print("*************************************M O D E L 5******************************************")
            print("********************************************************************************")
            print("********************************************************************************")

            self.running_model = PPO.load('./scripts/gyms/logs/Basic_Run_R1_258/best_model', device='cpu') # fast run == 1 hour traning  

        '''


        for _ in range(self.max_running_steps):
            action, _ = self.running_model.predict(obs, deterministic=True)
            obs, _, done, _ = super().step(action)
            if done:
                break
        return obs

    def step(self, action):

        obs, _, terminal, info = super().step(action)
        r = self.player.world.robot

         # 1. Torso Upright Reward (Maximize)
        torso_height = r.cheat_abs_pos[2]
        torso_reward = 2.0 * torso_height  # [0.3-0.8m] → 0.6 to 1.6 reward
        
        # 2. Foot Contact Balance Reward (Maximize)
        lf_force = r.frp.get('lf', (0,)*6)[5]  # Vertical force on left foot
        rf_force = r.frp.get('rf', (0,)*6)[5]  # Vertical force on right foot
        contact_balance = 1.0 - abs(lf_force - rf_force)/(lf_force + rf_force + 1e-6)
        foot_reward = 1.5 * contact_balance  # [0-1] → 0 to 1.5 reward
        
        # 3. CoM over Support Polygon Reward (Maximize)
        com_x, com_y = r.cheat_abs_pos[0], r.cheat_abs_pos[1]
        lf_x, lf_y = r.frp.get('lf', (0,)*6)[0:2]
        rf_x, rf_y = r.frp.get('rf', (0,)*6)[0:2]
        support_center_x = (lf_x + rf_x)/2
        support_center_y = (lf_y + rf_y)/2
        com_deviation = np.sqrt((com_x - support_center_x)**2 + (com_y - support_center_y)**2)
        com_reward = 1.8 * (1.0 - np.tanh(5.0 * com_deviation))  # 0-1.8 reward
        
        # 4. Smooth Deceleration Reward (Maximize)
        current_speed = abs(r.cheat_abs_pos[0] - self.lastx)
        speed_reduction = self.initial_speed - current_speed
        decel_reward = 0.6 * np.clip(speed_reduction, 0, None)  # Reward for slowing down
        
        # 5. Joint Smoothness Reward (Maximize)
        joint_accel = np.abs(r.joints_speed[2:22] - self.last_joint_speeds)
        smoothness_reward = 0.4 * (1.0 - np.tanh(0.5 * np.sum(joint_accel)))
        self.last_joint_speeds = r.joints_speed[2:22].copy()
        
        # 6. Full Stop Bonus (One-Time Reward)
        if current_speed < 0.01:  # Nearly stopped
            full_stop_bonus = 5.0
        else:
            full_stop_bonus = 0.0
        
        # 7. Survival Bonus (Per-Step Reward)
        survival_reward = 0.1  # Small reward for not falling

        # 8. Deviation Penalty (Per-Step Reward)
        self.deviation_from_heading = abs(self.stopping_start_pos_y - r.cheat_abs_pos[1])
        deviation_penalty = 0.05 * self.deviation_from_heading

        
        # Total Reward
        reward = (
            torso_reward + 
            foot_reward + 
            com_reward + 
            decel_reward + 
            smoothness_reward + 
            full_stop_bonus + 
            survival_reward -
            deviation_penalty
        )
        

        """"
        
        # Calculate stopping rewards
        current_speed = abs(r.cheat_abs_pos[0] - self.lastx)
        torso_penalty = 0.01 * (abs(r.imu_torso_roll) + abs(r.imu_torso_pitch))
        falling_penalty = 2.0 if r.cheat_abs_pos[2] < 0.3 else 0.0
        slow_speeding_reward = 1.5 if current_speed < self.prev_speed else 0.0

        # Joint velocity penalty (discourage rapid joint movements)
        joint_speed_penalty = 0.001 * np.sum(np.abs(r.joints_speed[2:22]))  # Exclude head and toes

        # Foot contact bonus (encourage stable foot placement)
        lf_fz = r.frp.get('lf', (0,)*6)[5]  # Left foot force (z-axis)
        rf_fz = r.frp.get('rf', (0,)*6)[5]  # Right foot force (z-axis)
        contact_bonus = 0.1 * (lf_fz + rf_fz) / 200  # Normalize force values

        # Balance reward (encourage even weight distribution)
        balance_reward = -0.1 * abs(lf_fz - rf_fz) / 200  # Penalize uneven weight distribution
        
        
        # CoM stability reward (encourage CoM over support polygon)
        com_x = r.cheat_abs_pos[0]  # Approximate CoM x position
        com_y = r.cheat_abs_pos[1]  # Approximate CoM y position
        lf_x, lf_y = r.frp.get('lf', (0,)*6)[0:2]  # Left foot position (x, y)
        rf_x, rf_y = r.frp.get('rf', (0,)*6)[0:2]  # Right foot position (x, y)
        
        # Calculate distance of CoM to the center of the support polygon
        support_center_x = (lf_x + rf_x) / 2
        support_center_y = (lf_y + rf_y) / 2
        com_deviation = np.sqrt((com_x - support_center_x)**2 + (com_y - support_center_y)**2)
        com_stability_reward = -0.1 * com_deviation  # Penalize CoM deviation
        
        
        reward = -current_speed - torso_penalty - falling_penalty + contact_bonus + slow_speeding_reward - joint_speed_penalty + balance_reward + contact_bonus + com_stability_reward
        #if the robot has stopped, log this info

        """
        if current_speed < 0.2 and self.stop_flag != 1 and self.env_id == 0:
                
                self.stop_flag = 1
                distance = abs(self.stopping_start_pos - r.cheat_abs_pos[0])
                self.log_to_csv(self.stopping_time_csv, self.episode_number, self.step_counter)
                self.log_to_csv(self.stopping_dist_csv, self.episode_number, distance)                

        if self.step_counter == 299 or r.cheat_abs_pos[2] < 0.3:
            if self.env_id == 0:
                
                self.log_to_csv(self.falls_csv_stopping, self.episode_number, self.fall_count)
                self.log_to_csv(self.stopping_deviation_csv, self.episode_number, self.deviation_from_heading)



        self.prev_speed = current_speed
        
        
        # Terminal conditions: Fallen or timeout (adjust max stopping steps as needed)
        terminal = terminal or self.step_counter >= 300
        
        return obs, reward, terminal, info

class Train(BaseTrain):
    def __init__(self, script) -> None:
        super().__init__(script)

    def train(self, args):
        n_envs = min(16, os.cpu_count())
        n_steps_per_env = 1024  # RolloutBuffer is of size (n_steps_per_env * n_envs)
        minibatch_size = 64  
        total_steps = 30000000  # Adjust training steps
        #total_steps = 7000  # Adjust training steps
        #running_model_path = "./scripts/gyms/logs/Basic_Run_R1_246/best_model"  # Path to fast trained running model
        running_model_path = "./scripts/gyms/logs/Basic_Run_R1_252/best_model"  # Path to slow trained running model == 30 minutes training

        folder_name = f'Basic_Stop_R{self.robot_type}'
        model_path = f'./scripts/gyms/logs/{folder_name}/'


        def init_env(i):
            def thunk():
                return Basic_Stop(
                    self.ip, 
                    self.server_p + i, 
                    self.monitor_p_1000 + i, 
                    self.robot_type, 
                    False,
                    env_id=i,
                    running_model_path=running_model_path
                )
            return thunk

        servers = Server(self.server_p, self.monitor_p_1000, n_envs+1)
        env = SubprocVecEnv([init_env(i) for i in range(n_envs)])
        eval_env = SubprocVecEnv( [init_env(n_envs)] )

        
        model = PPO("MlpPolicy", env, verbose=1, n_steps=1024, batch_size=64, learning_rate=3e-4, device="cpu")
        #model.learn(total_timesteps=total_steps)
        model_path = self.learn_model( model, total_steps, model_path, eval_env=eval_env, eval_freq=n_steps_per_env*20, save_freq=n_steps_per_env*200, backup_env_file=__file__ )


        
        env.close()
        servers.kill()

        def test(self, args):

            # Uses different server and monitor ports
            server = Server( self.server_p-1, self.monitor_p, 1 )
            env = Basic_Stop( self.ip, self.server_p-1, self.monitor_p, self.robot_type, True, 0 )
            model = PPO.load( args["model_file"], env=env )

            try:
                self.export_model( args["model_file"], args["model_file"]+".pkl", False )  # Export to pkl to create custom behavior
                self.test_model( model, env, log_path=args["folder_dir"], model_path=args["folder_dir"] )
            except KeyboardInterrupt:
                print()

            env.close()
            server.kill()





