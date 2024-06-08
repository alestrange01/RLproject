import numpy as np
import random

class Environment:
    def __init__(self):
        self.grid_size = (6, 6)
        self.BS_coverage = {
            0: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
            1: [(3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (4, 3), (5, 0), (5, 1), (5, 2), (5, 3)],
            2: [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3)],
            3: [(2, 4), (2, 5), (3, 3), (3, 4), (3, 5), (4, 4), (4, 5), (5, 4), (5, 5)]
        }
        self.BS_states = [0, 0, 0, 0]
        self.UE_position = (0, 0)
        self.end_position = (5, 3)
        self.time_step = 0
        self.is_end = False
        self.UE_path = self.gen_UE_path()

    def gen_UE_path(self):
        path = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), 
                (2, 3), (3, 3), (4, 3), (5, 3)]
        return path

    def reset(self):
        self.UE_position = (0, 0)
        self.BS_state = [0 for i in range(len(self.BS_coverage))]
        self.time_step = 0
        self.isEnd = False

    def step(self, action):
        for i, state in enumerate(action):
            self.BS_state[i] = state
        
        self.time_step += 1
        if self.time_step < len(self.UE_path):
            self.UE_position = self.UE_path[self.time_step]
        else:
            self.isEnd = True
        
        reward = self.get_reward()
        return reward

    def get_reward(self):
        covered = any(self.UE_position in self.BS_coverage[i] and self.BS_state[i] == 1 for i in range(len(self.BS_coverage)))
        active_cost = sum(self.BS_state.values())
        reward = covered - active_cost
        return reward

class Agent:
    def __init__(self, env):
        self.env = env
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1
        self.Q = {}

    def initialize_Q(self):
        for x in range(self.env.grid_size[0]):
            for y in range(self.env.grid_size[1]):
                for state in range(2**len(self.env.BS_coverage)):
                    self.Q[((x, y), state)] = [0] * 2**len(self.env.BS_coverage)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return [random.choice([0, 1]) for _ in range(len(self.env.BS_coverage))]
        return self.Q[state].index(max(self.Q[state]))

    def update_Q(self, state, action, reward, next_state):
        best_next_action = max(self.Q[next_state])
        self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * best_next_action - self.Q[state][action])

    def train(self, episodes):
        self.initialize_Q()
        for _ in range(episodes):
            self.env.reset()
            while not self.env.isEnd:
                state = (self.env.UE_position, sum([self.env.BS_state[i] * 2**i for i in range(len(self.env.BS_coverage))]))
                action = self.choose_action(state)
                reward = self.env.step(action)
                next_state = (self.env.UE_position, sum([self.env.BS_state[i] * 2**i for i in range(len(self.env.BS_coverage))]))
                self.update_Q(state, action, reward, next_state)

if __name__ == "__main__":
    env = Environment()
    agent = Agent(env)
    agent.train(10000)
