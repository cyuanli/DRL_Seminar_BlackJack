import gym
import Blackjack
import numpy as np

# Train params
discount_factor = 1.0
epsilon = 0.1
min_step_size = 0.01
n_no_updates = 1000

# Play params
n_games = 1000000

env = gym.make('Blackjack-v1')

Q = {}
N = {}


def get_state(state):
    hand = np.copy(state[0])
    hand = np.sum(np.reshape(hand, (4, 13)), axis=0)
    card_points = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])
    points = np.sum(np.multiply(hand, card_points))
    aces = hand[0]
    dealer = state[1] % 13
    if dealer >= 9:
        dealer = 9

    s = (points, aces, dealer)
    if s not in Q:
        Q[s] = {0: 0.0, 1: 0.0}
        N[s] = {0: 0, 1: 0}
    return s


def get_action(s, explore=False):
    if explore and np.random.rand() < epsilon:
        action = np.random.randint(2)
    else:
        action = 0 if Q[s][0] >= Q[s][1] else 1
    N[s][action] += 1
    return action


def get_temporal_difference(s, action, reward, s_1):
    max_Q = max(Q[s_1].values()) if s_1 else 0
    return reward + discount_factor * max_Q - Q[s][action]


def train():
    no_update_count = 0
    max_no_update = 0
    while no_update_count < n_no_updates:
        state = env.reset()
        terminal = False
        while not terminal:
            s = get_state(state)
            action = get_action(s, explore=True)
            state, reward, terminal, info = env.step(action)
            # print(info)
            s_1 = get_state(state) if not terminal else None
            step_size = 1 / N[s][action] * \
                get_temporal_difference(s, action, reward, s_1)
            Q[s][action] += step_size
            if abs(step_size) < min_step_size:
                no_update_count += 1
            else:
                if no_update_count > max_no_update:
                    max_no_update = no_update_count
                    print('Number of states visited: {}\nNumber of consecutive small update steps: {}'.format(
                        len(Q), no_update_count))
                no_update_count = 0


def play():
    sum_reward = 0
    for _ in range(n_games):
        state = env.reset()
        terminal = False
        while not terminal:
            s = get_state(state)
            action = get_action(s, explore=False)
            state, reward, terminal, info = env.step(action)
            sum_reward += reward
    print('Avg. reward per episode: {}'.format(sum_reward / n_games))


def main():
    train()
    play()


if __name__ == '__main__':
    main()
