import gym
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
import collections
from matplotlib import pyplot as plt
from scipy.stats import norm


class BlackBox:
    def __init__(self):
        self._init_model()

    def _init_model(self):
        model = Sequential([
            Dense(2, input_dim=6, activation='relu'),
            Dense(3, activation='softmax'),
            ])
        self.model = model
        self.shape = [2,3]
        return

    def flatten(self, weights):
        w = []
        for l in weights:
            if isinstance(l, collections.Iterable):
                w = w + self.flatten(l)
            else:
                w = w + [l]
        return w

    def unflatten(self, flat_weights, shape=[2,3]):
        w = []
        i = 0
        for l, size in enumerate(shape):
            layer = self.model.layers[l].get_weights()
            params = layer[0]
            bias = layer[1]
            new_layer = []
            new_params = []
            new_bias = []
            for param in params:
                new_params.append(flat_weights[i:i+size])
                i += size
            for b in bias:
                new_bias.append(flat_weights[i])
                i += 1
            w.append(np.array(new_params))
            w.append(np.array(new_bias))
        return w

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_flat_weights(self):
        return self.flatten(self.get_weights())

    def set_flat_weights(self, flat_weights):
        return self.set_weights(self.unflatten(flat_weights))

    def produce_action(self, state):
        inp = np.array([np.array(state).T])
        action_dist = self.model.predict(inp)
        action = np.random.choice(3,1,p=action_dist[0])[0]
        #  print("state: ", state, "actions: ", action_dist, "take: ", action)
        return action

def append_obs(obs, observation):
    obs.append(observation[0])
    obs.append(observation[1])
    obs.popleft()
    obs.popleft()
    return obs

def run_sim():
    env = gym.make('MountainCar-v0')
    alpha = 0.1
    sigma = 3
    bb = BlackBox()
    w = bb.get_flat_weights()
    pop_size = 200
    fitnesses = []
    for i_episode in range(20):
        observation = env.reset()
        obs = collections.deque([0,0,0,0,0,0])
        # Create a population by randomly mutating your network parameters
        noise = np.random.randn(pop_size, len(w))
        population = w + sigma*noise
        F = []
        for agent in population:
            bb.set_flat_weights(agent)
            fitness = 0
            actions = collections.defaultdict(int)
            for t in range(200):
                #  env.render()
                assert(len(obs)==6)
                obs = append_obs(obs, observation)
                action = bb.produce_action(np.array(list(obs)))
                actions[action] += 1
                observation, reward, done, info = env.step(action)
                # We will sum position and velocity to get reward
                reward = sum(observation)
                # Subtract 1 each time step to reward faster finishes
                fitness += (reward - 1)
            # Reward variance to discourage only using a single move
            fitness = fitness + norm.fit(list(actions.values()))[1]
            print("action distribution: ", actions, "fitness: ", fitness)
            F.append(fitness)
        print(
            'Gen: ', i_episode,
            '| Net_R: %.1f' % sum(F),
            )
        w = w + alpha*(1/(pop_size*sigma))*(noise.T*F).T.sum(axis=0)
        fitnesses.append(sum(F))
    # Cache Model
    bb.set_flat_weights(w)
    bb.model.save("ES-MountainCar.hdf5")
    #  plt.plot(fitnesses)

def test_model():
    env = gym.make('MountainCar-v0')
    bb = BlackBox()
    bb.model.load_weights('ES-MountainCar.hdf5')
    obs = collections.deque([0,0,0,0,0,0])
    for i_episode in range(20):
        observation = env.reset()
        for t in range(200):
            obs = append_obs(obs, observation)
            action = bb.produce_action(np.array(list(obs)))
            print(action)
            observation, reward, done, info = env.step(action)
            env.render()

if __name__ == '__main__':
    #  test_model()
    run_sim()

