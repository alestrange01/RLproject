import numpy as np
import random
import matplotlib.pyplot as plt

class Environment:
    def __init__(self):
        #Inizializzo la griglia 6x6
        self.grid_size = (6, 6)
        #Definisco la copertura di ciascuna BS
        self.BS_coverage = {
            0: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
            1: [(3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (4, 3), (5, 0), (5, 1), (5, 2), (5, 3)],
            2: [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3)],
            3: [(2, 4), (2, 5), (3, 3), (3, 4), (3, 5), (4, 4), (4, 5), (5, 4), (5, 5)]
        }
        #Definisco lo stato iniziale della BS, la posizione iniziale e finale dell'UE con il relativo path
        self.BS_state = [0, 0, 0, 0]
        self.UE_position = (0, 0)
        self.end_position = (5, 3)
        self.time_step = 0
        self.isEnd = False
        self.UE_path = self.gen_UE_path()
        self.covered_time = 0
        self.active_cost = 0

    def gen_UE_path(self):
        # Path dell'UE
        path = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), 
                (2, 3), (3, 3), (4, 3), (5, 3)]
        return path

    def reset(self):
        # Reset environment
        self.UE_position = (0, 0)
        self.BS_state = [0, 0, 0, 0]
        self.time_step = 0
        self.isEnd = False
        self.covered_time = 0
        self.active_cost = 0

    def step(self, action):
        #Aggiorno lo stato delle BS
        for i, state in enumerate(action):
            self.BS_state[i] = state
        
        #Vado avanti di un time step e aggiorno la posizione dell'UE
        self.time_step += 1
        if self.time_step < len(self.UE_path):
            self.UE_position = self.UE_path[self.time_step]
        else:
            self.isEnd = True
        
        #Calcolo la reward
        reward = self.get_reward()
        return reward

    def get_reward(self):
        #Vedo se l'UE è coperto da una sola BS
        covering_BS_count = sum(self.UE_position in self.BS_coverage[i] and self.BS_state[i] == 1 for i in range(len(self.BS_coverage)))
        active_BS_count = sum(self.BS_state)
        
        if covering_BS_count == 1 and active_BS_count == 1:
            reward = 1.5
            self.covered_time += 1
        elif covering_BS_count == 1 and active_BS_count > 1:
            reward = - 1 * (covering_BS_count)
        else:
            reward = -1.5

        #Calcolo il costo attivo
        active_BS_count = sum(self.BS_state)
        self.active_cost += active_BS_count

        return reward

class Agent:
    def __init__(self, env):
        self.env = env
        self.alpha = 0.1  #Tasso di apprendimento
        self.gamma = 0.99  #Fattore di sconto
        self.epsilon = 1.0  #Tasso di esplorazione iniziale
        self.epsilon_min = 0.025  #Tasso minimo di esplorazione
        self.epsilon_decay = 0.999  #Decadimento dell'epsilon per favorire sfruttamento
        self.Q = {}  #Tabella Q

    def initialize_Q(self):
        #Inizializzo la tabella Q
        for x in range(self.env.grid_size[0]):
            for y in range(self.env.grid_size[1]):
                for state in range(2**len(self.env.BS_coverage)): #Ogni possibile stato lo inizializzo a 0
                    self.Q[((x, y), state)] = [0] * 2**len(self.env.BS_coverage)

    def choose_action(self, state):
        #Scelgo l'azione da fare in base all'epsilon
        if random.random() < self.epsilon:
            return [random.choice([0, 1]) for _ in range(len(self.env.BS_coverage))]
        else:
            best_action_index = np.argmax(self.Q[state])
            return [(best_action_index >> i) & 1 for i in range(len(self.env.BS_coverage))]

    def update_Q(self, state, action, reward, next_state):
        #Aggiorno la tabella Q
        best_next_action = max(self.Q[next_state])
        action_index = sum([action[i] << i for i in range(len(action))])
        self.Q[state][action_index] = self.Q[state][action_index] + self.alpha * (reward + self.gamma * best_next_action - self.Q[state][action_index])

    def train(self, episodes):
        #Svolgo i miei episodi
        self.initialize_Q()
        rewards_per_episode = []
        for episode in range(episodes):
            #Ogni episodio resetto l'ambiente
            self.env.reset()
            cumulative_reward = 0
            while not self.env.isEnd:
                #Finchè non finisce l'episodio scelgo l'azione e aggiorno la tabella Q
                state = (self.env.UE_position, sum([self.env.BS_state[i] * 2**i for i in range(len(self.env.BS_coverage))]))
                action = self.choose_action(state)
                reward = self.env.step(action)
                next_state = (self.env.UE_position, sum([self.env.BS_state[i] * 2**i for i in range(len(self.env.BS_coverage))]))
                self.update_Q(state, action, reward, next_state)
                cumulative_reward += reward

            rewards_per_episode.append(cumulative_reward)
            #Ridurre l'epsilon per favorire l'utilizzo della tabella Q
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            #Stampo i risultati
            print(f"Episode {episode + 1}/{episodes}")
            print(f"Cumulative Reward: {cumulative_reward}")
            print(f"Covered Time: {self.env.covered_time}")
            print(f"Active Cost: {self.env.active_cost}")
            print(f"Epsilon: {self.epsilon}")
            print("------------")

        # Plot cumulative reward per episode
        plt.plot(range(episodes), rewards_per_episode)
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Reward per Episode')
        plt.show()

if __name__ == "__main__":
    env = Environment()
    agent = Agent(env)
    agent.train(10000)
