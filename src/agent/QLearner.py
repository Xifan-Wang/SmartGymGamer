import gymnasium as gym
import numpy as np
import time
from src.utils.agent import q_update_bellmann, epsilon_greedy

class QLearnig():
    def __init__(self, env:gym.Env, random_seed:int = 43):
        self.env = env
        self.random_seed = random_seed
        self.observations = env.observation_space.n
        self.actions = env.action_space.n
    
    def init_q_table(self):
        self.q_table = np.zeros((self.observations, self.actions))
        return True
    
    def load_q_table(self, path:str):
        self.q_table = np.load(path)
        return True

    def learn(self, episodes: int, epsilon, alpha, gamma):
        for episode in range(episodes):
            print(f"episode {episode}")
            start_time = time.time()
            observation, _ = self.env.reset()
            terminated = False
            steps = 0
            while not terminated and steps < 50:
                action = epsilon_greedy(
                    self.q_table,self.env,observation,epsilon
                )
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                self.q_table[observation, action] = q_update_bellmann(
                    self.q_table, observation, action, reward, next_observation, alpha, gamma
                    )
                observation = next_observation
                steps += 1
            end_time = time.time()
            duration = end_time - start_time
            print(f"duration: {duration}")
        print("training completed")

        np.save("src/agent/q_table", self.q_table)
        return None
    
    def decide(self):
        

    
    # def output_model(self, path:str):
    #     np.save(path, self.q_table)
    #     return None
    
    # def output_rewards(self, path, )




