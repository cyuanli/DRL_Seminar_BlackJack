import argparse
import Blackjack
import gym
import numpy as np
import random
import tensorflow as tf
from tqdm import trange


# tf.keras.mixed_precision.experimental.set_policy('mixed_float16')


class QNetworkModel(tf.keras.Model):
    def __init__(self):
        super(QNetworkModel, self).__init__()
        self.feature_1 = tf.keras.layers.Dense(11, activation='linear')
        self.feature_2 = tf.keras.layers.Dense(1, activation='relu')
        self.q_value_1 = tf.keras.layers.Dense(2, activation='relu')
        self.q_value_2 = tf.keras.layers.Dense(2, activation='linear')

    def call(self, state):
        x1 = self.feature_1(state[:, :52])
        x1 = self.feature_2(x1)
        x2 = self.feature_1(state[:, 52:])
        x2 = self.feature_2(x2)
        x = tf.concat([x1, x2], 1)
        x = self.q_value_1(x)
        x = self.q_value_2(x)
        return x


class QNetwork():
    def __init__(self, state_dim, action_space):
        self.model = QNetworkModel()
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def __call__(self, state):
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        return self.model(state)

    def train(self, experiences, gamma):
        states = np.stack(experiences[:, 0])
        actions = experiences[:, 1]
        rewards = experiences[:, 2]
        states_next = np.stack(experiences[:, 3])
        terminals = experiences[:, 4]

        q_next = self.__call__(states_next)
        q_expected = rewards + (1 - terminals) * \
            gamma * np.max(q_next, axis=1)
        with tf.GradientTape() as tape:
            q = self.__call__(states)
            q_train = np.copy(q)
            for i, action in enumerate(actions):
                q_train[i][action] = q_expected[i]

            loss = self.loss(q, q_train)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables))


class ReplayBuffer():
    def __init__(self, size=5000):
        self.buffer = []
        self.size = size

    def __len__(self):
        return len(self.buffer)

    def add(self, experiences):
        new_size = len(self.buffer) + len(experiences)
        if new_size > self.size:
            del self.buffer[0:new_size-self.size]
        self.buffer.extend(experiences)
        assert len(self.buffer) <= self.size

    def sample(self, n):
        return np.array(random.sample(self.buffer, n))


class BlackjackEnv():
    def __init__(self):
        self.env = gym.make('Blackjack-v1')
        self.state_dim = 104
        self.action_space = 2

    def _transform_state(self, state):
        one_hot = [False for _ in range(52)]
        one_hot[state[1]] = True
        return np.concatenate([state[0], one_hot])

    def reset(self):
        return self._transform_state(self.env.reset())

    def step(self, action):
        state, reward, terminal, _ = self.env.step(action)
        return self._transform_state(state), reward, terminal


class Agent():
    def __init__(self, env):
        self.epsilon = 0.2
        self.gamma = 1.

        self.env = env
        self.q_network = QNetwork(env.state_dim, env.action_space)
        self.replay_buffer = ReplayBuffer()

    def get_action(self, state, explore=True):
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.env.action_space)
        q_values = self.q_network(state)
        return np.argmax(q_values[0])

    def train(self, n_episodes, n_batch_size, t_train_interval):
        for t in trange(n_episodes, desc='Training', ncols=100):
            state = self.env.reset()
            terminal = False

            episode = []
            while not terminal:
                action = self.get_action(state)
                state_next, reward, terminal = self.env.step(action)
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
            state = self.env.reset()
            terminal = False
            cummulative_reward = 0

            while not terminal:
                action = self.get_action(state, explore=False)
                state, reward, terminal = self.env.step(action)
                cummulative_reward += reward

            rewards.append(cummulative_reward)
        print('Average reward: {}'.format(np.mean(rewards)))


def main(n_train_episodes, n_batch_size, t_train_interval, n_test_episodes):
    blackjack_env = BlackjackEnv()
    agent = Agent(blackjack_env)
    agent.train(n_train_episodes, n_batch_size, t_train_interval)
    agent.test(n_test_episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='DQN for coding challenge of DRL seminar')
    parser.add_argument('--n_train', dest='n_train_episodes', default=20000,
                        type=int, help='number of episodes for training')
    parser.add_argument('--batch_size', dest='n_batch_size', default=200,
                        type=int, help='size of training batches')
    parser.add_argument('--train_interval', dest='t_train_interval', default=10,
                        type=int, help='interval between trainings in episodes')
    parser.add_argument('--n_test', dest='n_test_episodes', default=10000,
                        type=int, help='number of episodes for testing')
    args = parser.parse_args()
    main(args.n_train_episodes, args.n_batch_size,
         args.t_train_interval, args.n_test_episodes)
