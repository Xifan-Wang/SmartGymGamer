import gymnasium as gym
import numpy as np

env = gym.make("Taxi-v3", render_mode="human")
observation, info = env.reset(seed = 43)

class QLearnig():
    def __init__(self, env:gym.Env, random_seed:int = 43):
        self.env = env
        self.random_seed = random_seed
    
    def init_q_table(self):
        n_observations = self.env.observation_space.n
        n_actions = self.env.action_space.n
        self.q_table = np.zeros((n_observations, n_actions))
        return True

    def learning(self, timesteps: int, epsilon, alpha, gamma):
        observation, _ = self.env.reset(seed=self.random_seed)
        for i in range(timesteps):
            rand_num = np.random.uniform(0, 1)
            if rand_num < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(self.q_table[observation])
            new_observation, reward, terminated, truncated, info = env.step(action)
            self.q_table[observation, action] += (
                (1-alpha)*self.q_table[observation, action] + alpha * (reward + gamma * np.max(self.q_table[new_observation]) - np.max(self.q_table[observation]))
            )
            observation = new_observation
            if terminated or truncated:
                observation, _ = self.env.reset()
        print("training completed")
