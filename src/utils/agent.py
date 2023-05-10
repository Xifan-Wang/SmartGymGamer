import numpy as np
def epsilon_greedy(q_table, env, observation, epsilon):
    random_number = np.random.uniform(0, 1)
    if random_number < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[observation])
    return action

def q_update_bellmann(q_table: np.array, observation: int, action:int, reward: int, next_observation:int, alpha:float, gamma: float):
    value = q_table[observation, action] + alpha * (reward + gamma * np.max(q_table[next_observation]) - q_table[observation, action])
    return value