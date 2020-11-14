import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
# discount is weight
# measure of how important we find future actions
# how much we value future reward over current reward
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 2000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# chance of doing random action
epsilon = 1.0  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
# amount to decay each episode
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    # we use this tuple to look up the 3 Q values for the available actions in the q-table
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())
    done = False

    if episode % SHOW_EVERY == 0 or episode > 10000:
        render = True
        print(episode)
    else:
        render = False
    
    while not done:

        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
             action = np.random.randint(0, env.action_space.n)
        
        new_state, reward, done, _ = env.step(action)

        new_discrete_state = get_discrete_state(new_state)
        
        if render:
            env.render()

        if not done:
            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            # equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            
            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q

        # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
        elif new_state[0] >= env.goal_position:
            print(f'Made it on episode {episode}')
            q_table[discrete_state + (action,)] = 0

        discrete_sate = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()
