import gym
import time
from keras.models import Sequential
from keras.layers import Dense, Activation
import collections
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt
from timeit import default_timer as timer

# Suppress Warnings
ERROR = 40
gym.logger.set_level(ERROR)

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


class CartPoleES:
    def __init__(self,
            env = 'CartPole-v1',
            pop_size=200,
            alpha=0.1,
            sigma=3,
            max_generations=50,
            convergence_reward=195,
            convergence_trials=100,
            max_steps=200,
            device='cpu',
            num_cores=1,
            verbose=1,
            ):
        self.pop_size=pop_size
        self.alpha=alpha
        self.sigma=sigma
        self.max_generations=max_generations
        self.convergence_reward=convergence_reward
        self.convergence_trials=convergence_trials
        self.max_steps=max_steps
        self.device=device
        self.num_cores=num_cores
        self.env = env
        self.verbose = verbose
        return

    def train(self, num_tests=1):
        history = []
        for test in range(num_tests):
            local_env = gym.make(self.env)
            wall_time_start = timer()
            total_trials = 0
            total_steps = 0
            total_generations = 0
            generation_times = []
            converged = False
            bb = BlackBox()
            weights = bb.get_flat_weights()
            convergence_test = collections.deque(maxlen=int(sqrt(self.convergence_trials)))
            fitness_history = []
            for generation in range(self.max_generations):
                total_generations = generation+1
                generation_time_start = timer()
                mutations = np.random.randn(self.pop_size, len(weights))
                population = weights + self.sigma*mutations
                F = []
                for brain in population:
                    total_trials += 1
                    observation = local_env.reset()
                    bb.set_flat_weights(brain)
                    agent_fitness = 0
                    # An agent gets a single episode to determine its fitness.
                    # In CartPole, fitness is equal to number of steps the the
                    # pole remains balanced
                    for step in range(self.max_steps):
                        total_steps += 1
                        action = bb.produce_action(np.array(list(observation)))
                        observation, reward, done, info = local_env.step(action)
                        agent_fitness += reward
                        # If the agents loses balance of the pole, the episode ends
                        if done:
                            break
                    F.append(agent_fitness)
                weights = weights + self.alpha*(1/(self.pop_size*self.sigma))*(mutations.T*F).T.sum(axis=0)
                generation_time_end = timer()
                current_fitness = self.test_weights(weights)
                gen_time = generation_time_end - generation_time_start
                if self.verbose:
                    print(
                        'Gen: ', generation,
                        '| Generation Time: ' + str(gen_time),
                        '| Fitness: %.1f\n' % current_fitness,
                        )
                fitness_history.append(current_fitness)
                convergence = fitness_history[-int(sqrt(self.convergence_trials)):]
                if (sum(convergence) / len(convergence)) >= self.convergence_reward:
                    converged = True
                    wall_time_end = timer()
                    wall_time = wall_time_end - wall_time_start
                    print("Convergence Reached after {0} Generations, in {1} seconds.".format(generation+1, wall_time))
                    break
                if generation == (self.max_generations - 1):
                    converged = False
                    wall_time_end = timer()
                    wall_time = wall_time_end - wall_time_start
                    print("Convergence Not Reached after {0} Generations, in {1} seconds.".format(generation+1, wall_time))

            self.trained_weights = weights
            hist = {
                    'wall_time': wall_time,
                    'fitness_history': fitness_history, 
                    'total_generations': total_generations,
                    'generation_times': generation_times,
                    'total_trials': total_trials,
                    'total_steps': total_steps,
                    'converged': converged,
                    'test_id': test,
                    }
            history.append(hist)
        return history

    def save_weights(self, filename="ES-CartPole-v1.hdf5", experiment_id=None):
        bb.set_flat_weights(self.trained_weights)
        if experiment_id:
            filename = str(experiment_id) + filename
        bb.model.save(filename)

    def test_weights(self, w):
        bb = BlackBox()
        bb.set_flat_weights(w)
        env = gym.make(self.env)
        num_trials = int(sqrt(self.convergence_trials))
        average_fitness = 0
        for i in range(num_trials):
            fitness = 0
            observation = env.reset()
            for t in range(self.max_steps):
                action = bb.produce_action(np.array(list(observation)))
                observation, reward, done, info = env.step(action)
                fitness += reward
                if done:
                    break
            average_fitness += fitness / num_trials
        return average_fitness

