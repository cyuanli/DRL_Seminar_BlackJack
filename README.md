# Deep Reinforcement Learning Seminar '20 Coding Challenge

This repository contains a Blackjack environment as coding challenge for the deep reinforcement learning seminar 2020.

The challenge is setup in 4 difficulty levels:

1. Get familiar with the frameworks and learn an easy version of Blackjack, where the dealer only has one card which is revield to you. That is, your algorithm should be able to learn to collect cards until the value of the dealers card is exceeded, without going above 21.
2. Let your agent learn Blackjack (Dealer hits on 16, Dealer wins draws).
3. Investigate, whether your agent is also able to learn Blackjack if the carde values are changed.
4. We will provide a new version of the game shortly before the deadline to question your agents capabilities further.

## Setup

Requirements: Conda (https://docs.conda.io/en/latest/)

Clone and enter the Git repository:

```bash
git clone https://gitlab.ethz.ch/disco/teaching/deep-reinforcement-learning-seminar.git
cd deep-reinforcement-learning-seminar
```

Install and activate the conda environment

```bash
conda create -f conda_env.yml
conda activate drl-seminar
```

Run the random agent to see whether everything is working

```bash
python random_agent.py
```

The repository also provides a code snippet to train a PPO agent from RLlib (https://ray.readthedocs.io/en/latest/rllib.html) for 100000 steps on the easy Blackjack task, where the dealer only uses one card.
