
import gym
import Blackjack
import numpy as np
import tensorflow as tf


class QNetwork(object):
    def __init__(self):
        # Params
        self.n_train_episodes = 20000
        self.n_train_iterations = 20
        self.n_batch_episodes = 100
        self.n_epochs = 20
        self.epsilon = 0.1
        self.gamma = 1.0

        self.n_play_episodes = 10000

        self.env = gym.make('Blackjack-v1')

        self._model = None
        self._init_model()

    def _init_model(self):
        self._model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu',
                                  input_shape=(104,), dtype=tf.dtypes.float16),
            tf.keras.layers.Dense(10, activation='relu',
                                  dtype=tf.dtypes.float16),
            tf.keras.layers.Dense(3, activation='relu',
                                  dtype=tf.dtypes.float16),
            tf.keras.layers.Dense(2, dtype=tf.dtypes.float16)
        ])
        opt = tf.keras.optimizers.Adamax(learning_rate=0.001)
        self._model.compile(optimizer=opt, loss='mse')

    def transform_state(self, state):
        dealer = [False for _ in range(52)]
        dealer[state[1]] = True
        return np.concatenate([state[0], dealer])

    def get_action(self, state, explore=True):
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(2)
        state = np.array([state])
        q_values = self._model.predict(state)
        return np.argmax(q_values[0])

    def act_and_observe(self, action):
        state, reward, terminal, info = self.env.step(action)
        # print(info)
        return self.transform_state(state), reward, terminal

    def train(self):
        replay_buffer = []
        for t in range(self.n_train_episodes):
            state = self.transform_state(self.env.reset())
            terminal = False

            while not terminal:
                action = self.get_action(state)
                state_next, reward, terminal = self.act_and_observe(action)
                replay_buffer.append(
                    (state, action, reward, state_next, terminal))
                state = state_next

            if t % self.n_batch_episodes == 0 and len(replay_buffer) > 100:
                loss_sum = 0
                for _ in range(self.n_train_iterations):
                    states = []
                    actions = []
                    rewards = []
                    states_next = []
                    terminals = []
                    for idx in np.random.randint(0, len(replay_buffer), 100):
                        experience = replay_buffer[idx]
                        states.append(experience[0])
                        actions.append(experience[1])
                        rewards.append(experience[2])
                        states_next.append(experience[3])
                        terminals.append(experience[4])
                    states = np.array(states)
                    states_next = np.array(states_next)
                    q_next = self._model.predict(states_next)
                    q_next = np.max(q_next, axis=1)
                    q_expected = np.array(rewards) + \
                        (1 - np.array(terminals)) * self.gamma * q_next
                    q = self._model.predict(states)
                    for i, action in enumerate(actions):
                        q[i][action] = q_expected[i]
                    history = self._model.fit(
                        states, q, epochs=self.n_epochs, verbose=0)
                    loss_sum += history.history['loss'][-1]
                print('{}/{}: {}'.format(t, self.n_train_episodes,
                                         loss_sum / self.n_train_iterations))
        print('Finished training ...')

    def play(self):
        cummulative_reward = 0
        for t in range(self.n_play_episodes):
            print('\rPlaying {}/{}'.format(t, self.n_play_episodes), end='')
            state = self.transform_state(self.env.reset())
            terminal = False

            while not terminal:
                action = self.get_action(state, explore=False)
                state, reward, terminal = self.act_and_observe(action)
                cummulative_reward += reward
        print('\nAvg. reward: {}'.format(
            cummulative_reward / self.n_play_episodes))


def main():
    agent = QNetwork()
    agent.train()
    agent.play()


if __name__ == '__main__':
    main()
