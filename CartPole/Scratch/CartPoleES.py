import gym
import time
from keras.models import Sequential
from keras.layers import Dense, Activation
import collections
from matplotlib import pyplot as plt
import numpy as np

class BlackBox:
    def __init__(self):
        self._init_model()

    def _init_model(self, shape=[2,2], actions=2):
        model = Sequential([
            Dense(2, input_dim=4, activation='relu'),
            Dense(actions, activation='softmax'),
            ])
        self.model = model
        self.shape = shape
        self.actions = actions
        return

    def flatten(self, weights):
        w = []
        for l in weights:
            if isinstance(l, collections.Iterable):
                w = w + self.flatten(l)
            else:
                w = w + [l]
        return w

    def unflatten(self, flat_weights):
        w = []
        i = 0
        for l, size in enumerate(self.shape):
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
        action = np.random.choice(self.actions,1,p=action_dist[0])[0]
        #  print("state: ", state, "actions: ", action_dist, "take: ", action)
        return action

def run_sim():
    env = gym.make('CartPole-v1')
    alpha = 0.1
    sigma = 3
    bb = BlackBox()
    w = bb.get_flat_weights()
    pop_size = 200
    test = collections.deque(maxlen=10)
    fitnesses = []
    for generation in range(40):
        observation = env.reset()
        noise = np.random.randn(pop_size, len(w))
        population = w + sigma*noise
        F = []
        for agent in population:
            observation = env.reset()
            bb.set_flat_weights(agent)
            fitness = 0
            actions = collections.defaultdict(int)
            for t in range(200):
                #  time.sleep(0.02)
                #  env.render()
                action = bb.produce_action(np.array(list(observation)))
                actions[action] += 1
                observation, reward, done, info = env.step(action)
                # We will sum position and velocity to get reward
                fitness += reward
                if done:
                    #  print("Episode finished after {} timesteps".format(t+1))
                    break
            #  print("action distribution: ", dict(actions), "fitness: ", fitness)
            #  print()
            F.append(fitness)
        #  print(
        #      'Gen: ', generation,
        #      '| Net_R: %.1f' % average_reward,
        #      )
        w = w + alpha*(1/(pop_size*sigma))*(noise.T*F).T.sum(axis=0)
        current_fitness = test_convergence(w, test, env)
        fitnesses.append(current_fitness)
        test.append(current_fitness)
        if (sum(test) / len(test)) > 195:
            print("Convergence Reached after {} Generations".format(t+1))

    # Cache Model
    bb.set_flat_weights(w)
    bb.model.save("ES-CartPole-v1.hdf5")
    plt.plot(fitnesses)
    plt.savefig('fitnesses.png')
    plt.show()

def test_convergence(w, test, env, max_steps=200, num_trials=10):
    bb = BlackBox()
    bb.set_flat_weights(w)
    average_fitness = 0
    for i in range(num_trials):
        fitness = 0
        observation = env.reset()
        for t in range(max_steps):
            action = bb.produce_action(np.array(list(observation)))
            observation, reward, done, info = env.step(action)
            fitness += reward
            if done:
                break
        average_fitness += fitness / num_trials
    print("Current fitness: ", average_fitness)
    return average_fitness

def test_model():
    env = gym.make('CartPole-v1')
    bb = BlackBox()
    bb.model.load_weights('ES-CartPole-v1.hdf5')
    for generation in range(20):
        observation = env.reset()
        for t in range(500):
            time.sleep(0.01)
            env.render()
            action = bb.produce_action(np.array(list(observation)))
            #  print(action)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break



if __name__ == '__main__':
    run_sim()
    #  test_model()
