from Blackjack import BlackjackEnv
from blackjack_dqn import BlackjackDqn

n_train_episodes = 200000
n_test_episodes = 10000
n_batch_size = 100
t_train_interval = 10

environment_configurations = [{"one_card_dealer": True},
                              {},
                              {"card_values": [2] * 52},
                              {"card_values": [3,  1,  3,  9,  6,  0,  7, -2,  2,  6,  8,  1,  3,
                                               4, -1,  4,  3,  9, -1,  4,  0,  4,  7, -2, -1,  5,
                                               2,  6, -3, -1,  2,  2, -1,  7,  1,  0,  7,  8,  4,
                                               5,  3, -1,  0,  3, -1,  3,  0,  6, -2,  4, -3,  4]}]

achieved_win_rates = []
for env_config in environment_configurations:
    print("Config: {}".format(env_config))
    env = BlackjackEnv(env_config)
    agent = BlackjackDqn(env)
    agent.train(n_train_episodes, n_batch_size, t_train_interval)
    avg_reward = agent.test(n_test_episodes)
    print('Average reward: {}'.format(avg_reward))
    achieved_win_rates.append(avg_reward)

print("DQN achieved the following win rates: {}".format(achieved_win_rates))

# OUTPUT:
# DQN achieved the following win rates:
# [0.9977, 0.4078, 1.0, 0.4448]
