from src.env.env import Env
from src.agent.QLearner import QLearnig
from src.utils.agent import epsilon_greedy

def taxi_q_learning():
    taxi_env = Env(env_name="Taxi-v3").create_env()
    agent = QLearnig(
        env=taxi_env,
        random_seed=43
    )
    agent.init_q_table()
    agent.learn(
        episodes=10000,
        epsilon=0.1,
        alpha=0.2,
        gamma=0.99
    )
    taxi_env.close()
    taxi_env_test = Env(env_name="Taxi-v3", render_mode="human").create_env()
    for i in range(5):
        observation, _ = taxi_env_test.reset()
        terminated = False
        rewards = 0
        while not terminated and rewards > -50:
            action = epsilon_greedy(agent.q_table, taxi_env_test, observation, 0)
            next_observation, reward, terminated, _, _ = taxi_env_test.step(action)
            rewards += reward
            observation = next_observation
        print(f"reward is {rewards}")

    taxi_env_test.close()


if __name__ == "__main__":
    taxi_q_learning()

    