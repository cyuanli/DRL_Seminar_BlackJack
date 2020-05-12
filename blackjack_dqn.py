import argparse
import Blackjack
import gym
import numpy as np
from tqdm import trange
from Agents.dqn import DqnAgent


class BlackjackDqn(DqnAgent):
    def __init__(self, env):
        DqnAgent.__init__(self, env, state_dim=104, action_space=2)

    def _transform_state(self, state):
        one_hot = [False for _ in range(52)]
        one_hot[state[1]] = True
        return np.concatenate([state[0], one_hot])

    def train(self, n_episodes, n_batch_size, t_train_interval):
        for t in trange(n_episodes, desc='Training', ncols=100):
            state = self._transform_state(self.env.reset())
            terminal = False

            episode = []
            while not terminal:
                action = self.get_action(state)
                state_next, reward, terminal, _ = self.env.step(action)
                state_next = self._transform_state(state_next)
                episode.append(
                    np.array([state, action, reward, state_next, terminal]))
                state = state_next
            self.replay_buffer.add(episode)

            if t % t_train_interval == 0 and len(self.replay_buffer) >= n_batch_size:
                batch = self.replay_buffer.sample(n_batch_size)
                self.q_network.train(batch, self.gamma)

    def test(self, n_episodes):
        rewards = []
        for _ in trange(n_episodes, desc='Testing', ncols=100):
            state = self._transform_state(self.env.reset())
            terminal = False
            cummulative_reward = 0

            while not terminal:
                action = self.get_action(state, explore=False)
                state, reward, terminal, _ = self.env.step(action)
                state = self._transform_state(state)
                cummulative_reward += reward

            rewards.append(cummulative_reward)
        return np.mean(rewards)


def main(n_train_episodes, n_batch_size, t_train_interval, n_test_episodes):
    agent = BlackjackDqn(gym.make('Blackjack-v1'))
    agent.train(n_train_episodes, n_batch_size, t_train_interval)
    print('Average reward: {}'.format(agent.test(n_test_episodes)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='DQN for coding challenge of DRL seminar')
    parser.add_argument('--n_train', dest='n_train_episodes', default=20000,
                        type=int, help='number of episodes for training')
    parser.add_argument('--batch_size', dest='n_batch_size', default=100,
                        type=int, help='size of training batches')
    parser.add_argument('--train_interval', dest='t_train_interval', default=10,
                        type=int, help='interval between trainings in episodes')
    parser.add_argument('--n_test', dest='n_test_episodes', default=2000,
                        type=int, help='number of episodes for testing')
    args = parser.parse_args()
    main(args.n_train_episodes, args.n_batch_size,
         args.t_train_interval, args.n_test_episodes)
