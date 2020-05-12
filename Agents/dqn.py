import numpy as np
import random
import tensorflow as tf
from tqdm import trange


# tf.keras.mixed_precision.experimental.set_policy('mixed_float16')


class QNetwork():
    def __init__(self, state_dim):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(state_dim),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2, activation='linear')
        ])
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.2)

    def predict(self, state):
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        return self.model(state)

    def train(self, experiences, gamma):
        states = np.stack(experiences[:, 0])
        actions = experiences[:, 1]
        rewards = experiences[:, 2]
        states_next = np.stack(experiences[:, 3])
        terminals = experiences[:, 4]

        q_next = self.predict(states_next)
        q_expected = rewards + (1 - terminals) * \
            gamma * np.max(q_next, axis=1)
        with tf.GradientTape() as tape:
            q = self.predict(states)
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


class DqnAgent():
    def __init__(self, env, state_dim, action_space):
        self.state_dim = state_dim
        self.action_space = action_space
        self.epsilon = 0.2
        self.gamma = 1.

        self.env = env
        self.q_network = QNetwork(state_dim)
        self.replay_buffer = ReplayBuffer()

    def get_action(self, state, explore=True):
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)
        q_values = self.q_network.predict(state)
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
        return np.mean(rewards)
