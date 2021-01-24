from collections import deque
import numpy as np

def dqn_learning(env, agent, n_episodes=2000, max_t=1000, eps_start=1.0,
                 eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning

    Params
    ======
        env: environment
        agent: the dqn agent
        n_episodes (int): number of episodes to train the agent
        max_t (int): number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    # initialise the environment
    brain_name = env.brain_names[0]
    # env_info = env.reset(train_mode=True)[brain_name]
    scores = []                                                 # list containing scores from each episode
    scores_window = deque(maxlen=100)                           # last 100 scores
    eps = eps_start                                             # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]       # reset the environment
        state = env_info.vector_observations[0]                 # get the current state
        score = 0                                               # initialize the score
        for t in range(max_t):
            action = agent.act(state, eps)                      # select action according to
                                                                # epsilon greedy policy
            env_info = env.step(action)[brain_name]             # send the action to the environment
            next_state = env_info.vector_observations[0]        # get the next state
            reward = env_info.rewards[0]                        # get the reward
            done = env_info.local_done[0]                       # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)                             # save most recent score
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)                       # decrease epsilon
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'
                .format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\
                \tAverage Score: {:.2f}'.format(i_episode-100,
                np.mean(scores_window)))
            
    return scores
