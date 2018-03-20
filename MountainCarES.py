import gym
import math
env = gym.make('MountainCar-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(10000):
        env.render()
        action = 2#env.action_space.sample()
        if math.ceil(math.sqrt(t)) % 2:
            action = 2
            print("action = ", 2)
        else:
            action = 0
            print("action = ", 0)
        observation, reward, done, info = env.step(action)
        #  print(observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            exit()
            break
