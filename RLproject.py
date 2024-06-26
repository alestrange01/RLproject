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
        #Path dell'UE
        path = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (4, 3), (5, 3)]
        return path

    def reset(self):
        #Reset environment
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
        self.UE_position = self.UE_path[self.time_step]
        if self.UE_position == self.end_position:
            self.isEnd = True
        #Calcolo la reward
        reward = self.get_reward()
        return reward

    def get_reward(self):
        #Vedo se l'UE è coperto da una sola BS
        covering_BS_count = sum(self.UE_position in self.BS_coverage[i] and self.BS_state[i] == 1 for i in range(len(self.BS_coverage)))
        active_BS_count = sum(self.BS_state)
        if covering_BS_count == 1:
            self.covered_time += 1
            if active_BS_count == 1:
                reward = 5
            else:
                reward = -1.5 * (active_BS_count - 1) #Penalizzo il costo attivo
        else:
            reward = -2 * (active_BS_count)
        #Calcolo il costo attivo
        self.active_cost += active_BS_count
        return reward

class Agent:
    def __init__(self, env):
        self.env = env
        self.alpha = 0.1 #Tasso di apprendimento
        self.gamma = 0.99 #Fattore di sconto
        self.epsilon = 1.0 #Tasso di esplorazione iniziale
        self.epsilon_min = 0.025 #Tasso minimo di esplorazione
        self.epsilon_decay = 0.999 #Decadimento dell'epsilon
        self.Q = {} #Tabella Q

    def initialize_Q(self):
        #Inizializzo la tabella Q
        #for x in range(self.env.grid_size[0]):
        #    for y in range(self.env.grid_size[1]):
        #        #for state in range(2**len(self.env.BS_coverage)):
        #        #    self.Q[(x, y, state)] = [0] * 2**len(self.env.BS_coverage)
        #        self.Q[(x, y)] = [0] * 2**len(self.env.BS_coverage)
        #Versione ottimizzata: inizializzo la tabella Q solo per le posizioni dell'UE
        #for x, y in self.env.UE_path:
        #    if(x, y) == self.env.end_position:
        #        continue
        #    if (x, y) == (0, 0):
        #        self.Q[(x, y), 0] = [0] * 2**len(self.env.BS_coverage)
        #        continue
        #    for state in range(2**len(self.env.BS_coverage)):
        #        self.Q[(x, y), state] = [0] * 2**len(self.env.BS_coverage)

        #Versione ottimizzata v2: inizializzo la tabella Q solo per le posizioni dell'UE
        for position in self.env.UE_path:
            if position == self.env.end_position:
                continue
            self.Q[position] = [0] * 2**len(self.env.BS_coverage)

    def choose_action(self, state):
        #Scelgo l'azione da fare in base all'epsilon
        if random.random() < self.epsilon:
            return [random.choice([0, 1]) for _ in range(len(self.env.BS_coverage))]
        else:
            #Scelgo l'azione migliore
            #best_action_index = np.argmax(self.Q[state])
            #return [best_action_index >> i & 1 for i in range(len(self.env.BS_coverage))]

            #Scelgo casualmente l'azione migliore nel caso ci siano più azioni migliori
            best_action_indexes = [i for i, action in enumerate(self.Q[state]) if action == max(self.Q[state])]
            return [(random.choice(best_action_indexes)) >> i & 1 for i in range(len(self.env.BS_coverage))]

    def update_Q(self, state, action, reward, next_state):
        #Aggiorno la tabella Q
        if next_state == self.env.end_position:
            best_next_action = 0
        else:
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
                #Finché non finisce l'episodio scelgo l'azione e aggiorno la tabella Q
                state = self.env.UE_position
                action = self.choose_action(state)
                reward = self.env.step(action)
                next_state = self.env.UE_position
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
        #Plot cumulative reward per episode
        plt.plot(range(episodes), rewards_per_episode)
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Reward per Episode')
        plt.show()

    def eval(self):
        #Valuto l'agente
        self.env.reset()
        self.epsilon = 0 #Disabilito l'esplorazione
        cumulative_reward = 0
        actions = []
        while not self.env.isEnd:
            state = self.env.UE_position
            action = self.choose_action(state)
            reward = self.env.step(action)
            cumulative_reward += reward
            actions.append(action)
            print(f"State: {state}")
            print(f"Action: {action}")
            print(f"Reward: {reward}")
            print("BS State: ", self.env.BS_state)
            print("Covered Time: ", self.env.covered_time)
            print("------------")
        print(f"Cumulative Reward: {cumulative_reward}")
        print(f"Covered Time: {self.env.covered_time}")
        print(f"Active Cost: {self.env.active_cost}")
        print("Actions: ", actions)
        #print("Q Table: ", self.Q)

if __name__ == "__main__":
    random.seed(0)
    env = Environment()
    agent = Agent(env)
    agent.train(5000)
    agent.eval()