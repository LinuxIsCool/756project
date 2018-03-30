import gym
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
import collections
import multiprocessing as mp
from collections import defaultdict, deque

from keras import backend as K
import os
from importlib import reload

from scipy.stats import norm

from matplotlib import pyplot as plt

N_CORE = mp.cpu_count()-1

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

set_keras_backend("theano")

class BlackBox:
    def __init__(self):
        self._init_model()

    def _init_model(self):
        model = Sequential([
            Dense(4, input_dim=2, activation='relu'),
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
        return np.argmax(self.model.predict(inp))

def append_obs(obs, observation):
    obs.append(observation[0])
    obs.append(observation[1])
    obs.popleft()
    obs.popleft()
    return obs

def agent_sim(args):
    obs = deque([0,0, 0,0, 0,0])
    (agent, env) = args
    bb = BlackBox()
    bb.set_flat_weights(agent)
    observation = env.reset()
    obs = append_obs(obs, observation)
    fitness = 0
    action_count = defaultdict(int)
    for t in range(200):
        action = bb.produce_action(np.array(observation))#list(obs)))
        action_count[action] += 1
        observation, reward, done, info = env.step(action)
        obs = append_obs(obs, observation)
        print(obs)
        # We will sum position and velocity to get reward
        r = reward + sum([o*o*o for o in observation])
        # Subtract 1 each time step to reward faster finishes
        fitness += r
    # We reward our model for variance in its action choice. Otherwise, it tends to only select one action.
    print("action_count: ", action_count.values(), "fitness: ", fitness, "variance: ", norm.fit(list(action_count.values()))[1])
    fitness = fitness + norm.fit(list(action_count.values()))[1]
    return fitness

def run_sim():
    env = gym.make('MountainCar-v0')
    alpha = 0.3
    sigma = 3
    bb = BlackBox()
    w = bb.get_flat_weights()
    pop_size = 200
    n_episodes = 20
    pool = mp.Pool(processes=N_CORE)
    fitnesses = []
    for i_episode in range(n_episodes):
        print(i_episode)
        # Create a population by randomly mutating your network parameters
        noise = np.random.randn(pop_size, len(w))
        population = w + sigma*noise
        # Parallel agent evaluation
        F = pool.map_async(agent_sim, [(agent, env) for agent in population]).get()
        # Update weights as a linear combination of random perturbations based on fitnesss
        w = w + alpha*(1/(pop_size*sigma))*(noise.T*F).T.sum(axis=0)
        print(
            'Gen: ', i_episode,
            '| Net_R: %.1f' % sum(F),
            )
        fitnesses.append(sum(F))
    # Cache Model
    bb.set_flat_weights(w)
    bb.model.save("ES-MountainCar.hdf5")
    plt.plot(fitnesses)

def test_model():
    env = gym.make('MountainCar-v0')
    bb = BlackBox()
    bb.model.load_weights('ES-MountainCar.hdf5')
    for i_episode in range(20):
        obs = env.reset()
        for t in range(200):
            action = bb.produce_action(obs)
            print(action)
            obs, reward, done, info = env.step(action)
            env.render()

if __name__ == '__main__':
    run_sim()
    #  test_model()

