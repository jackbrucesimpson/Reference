import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

# first value is position and second is velocity
print(env.observation_space.high)
print(env.observation_space.low)
# actions you can do
print(env.action_space.n)

# want to have 20 buckets for the q table to break up the states
# into manageable sizes
DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

print(discrete_os_win_size)

# 20 x 20 table contains every combo of position and velocity
# plus 3 for each action you can take
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
print(q_table.shape)

done = False
while not done:
    # 0: left
    # 1: nothing
    # right
    action = 2
    
    # state: position and velocity
    # reward will be -1 until it reaches the flag and then you get reward 0
    # when you hit the reward you backprop the chain of events there
    new_state, reward, done, _ = env.step(action)
    
    print(new_state, reward)

    env.render()

env.close()
