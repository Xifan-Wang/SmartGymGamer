from src.env.env import Env
from src.agent.QLearner import QLearnig, QLearningTest
from src.utils.agent import epsilon_greedy
import numpy as np

def taxi_q_learning():
    taxi_env = Env(env_name="Taxi-v3").create_env()
    agent = QLearnig(env=taxi_env)
    agent.init_q_table()
    agent.learn(
        episodes=2000,
        epsilon=0.1,
        alpha=0.2,
        gamma=0.99
    )
    agent.plot_rewards()
    taxi_env.close()
    return agent

def taxi_test(agent: QLearnig, episodes:int):
    taxi_test_env = Env(env_name="Taxi-v3", render_mode="human").create_env()
    test = QLearningTest(taxi_test_env, agent)
    test.run_test(episodes)
    taxi_test_env.close()



if __name__ == "__main__":
    agent = taxi_q_learning()
    taxi_test(agent, 10)

    