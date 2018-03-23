import gym
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
import collections

class BlackBox:
    def __init__(self):
        self._init_model()

    def _init_model(self):
        model = Sequential([
            Dense(2, input_dim=2, activation='relu'),
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

def run_sim():
    env = gym.make('MountainCar-v0')
    alpha = 0.1
    sigma = 3
    bb = BlackBox()
    w = bb.get_flat_weights()
    pop_size = 200

    for i_episode in range(20):
        observation = env.reset()
        # Create a population by randomly mutating your network parameters
        noise = np.random.randn(pop_size, len(w))
        population = w + sigma*noise
        F = []
        for agent in population:
            bb.set_flat_weights(agent)
            fitness = 0
            for t in range(200):
                #  env.render()
                action = bb.produce_action(observation)
                observation, reward, done, info = env.step(action)
                # We will sum position and velocity to get reward
                reward = sum(observation)
                # Subtract 1 each time step to reward faster finishes
                fitness += (reward - 1)
            F.append(fitness)
        print(
            'Gen: ', i_episode,
            '| Net_R: %.1f' % sum(F),
            )
        w = w + alpha*(1/(pop_size*sigma))*(noise.T*F).T.sum(axis=0)

if __name__ == '__main__':
    #  bb = BlackBox()
    #  obs = env.reset()
    #  bb.produce_action(obs)
    run_sim()

