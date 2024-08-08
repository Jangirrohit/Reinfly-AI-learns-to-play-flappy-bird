# reinfly: AI learns to play flappy bird


Hello, I'm Rohit Jangir. I have taken this project under Summer of code projects. Follow the Readme.md file for overview of the project.

## Objective

Reinforcement Learning (RL) is gaining popularity across various fields such as finance, gaming, and robotics due to its self-learning capabilities based on a reward system. The objective of this project is to apply RL to the Flappy Bird game using Deep Q-Learning.

##  Envirement setup
### State space
The "FlappyBird-v0" environment, yields simple numerical information about the game's state as observations representing the game's screen.

#### state option
- the last pipe's horizontal position
- the last top pipe's vertical position
- the last bottom pipe's vertical position
- the next pipe's horizontal position
- the next top pipe's vertical position
- the next bottom pipe's vertical position
- the next next pipe's horizontal position
- the next next top pipe's vertical position
- the next next bottom pipe's vertical position
- player's vertical position
- player's vertical velocity
- player's rotation

### Action space
- 0 - do nothing
- 1 - flap

### Rewards 
- +0.1 - every frame it stays alive
- +1.0 - successfully passing a pipe
- -1.0 - dying
- −0.5 - touch the top of the screen
### Installation
To install ```flappy-bird-gymnasium```, simply run the following command:
```pip install flappy-bird-gymnasium```
### Playing
for playing run the following command:
```python agent.py flappybird1```


## Code Explanation

- first I created a neural netwok for deep q learning 
  

``` import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.output(x)

if __name__ == '__main__':
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim)
    state = torch.randn(10, state_dim)
    output = net(state)
    print(output)
```
This code defines a simple feedforward neural network with one hidden layer. It is used to estimate the Q-values for different actions given a state in the Flappy Bird game.

- Next, I created an experience replay file containing the ReplayMemory class, which allows for custom-sized memory and sampling:

```
from collections import deque
import random
class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)

        # Optional seed for reproducibility
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
```

- Next I set the hyperparameters for Deep Q-Learning in the hyperparameters.yml file:
  
```
flappybird1:
    env_id: FlappyBird-v0
    replay_memory_size: 100000
    mini_batch_size: 32
    epsilon_init: 1
    epsilon_decay: 0.99_99_5
    epsilon_min: 0.05
    network_sync_rate: 10
    learning_rate_a: 0.0001
    discount_factor_g: 0.99
    stop_on_reward: 100000
    fc1_nodes: 512
    env_make_params:
    use_lidar: False
```
- Next, I make some neccesary imports 
  
```
import gymnasium as gym
    
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from experience_replay import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import argparse
import itertools

import flappy_bird_gymnasium
import os
```
- Next I make a Agent class 
```
class Agent():
```
- initialize the agent class with hyperameters 
```
def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters (adjustable)
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict

        # Neural Network
        self.loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
        self.optimizer = None                # NN Optimizer. Initialize later.

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')
```
- Next, I make Run method in the Agent class 
  
    The run method is responsible for managing the training loop, including action selection, experience collection, network updates, and logging. It handles both training and evaluation modes, adjusts exploration through epsilon decay, and ensures periodic synchronization of networks and model saving. This method allows the agent to learn to play Flappy Bird through interaction with the environment and iterative policy improvement.
```
def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        # Create instance of the environment.
        # Use "**self.env_make_params" to pass in environment-specific parameters from hyperparameters.yml.
        env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

        # Number of possible actions
        num_actions = env.action_space.n

        # Get observation space size
        num_states = env.observation_space.shape[0] # Expecting type: Box(low, high, (shape0,), float64)

        # List to keep track of rewards collected per episode.
        rewards_per_episode = []

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training:
            # Initialize epsilon
            epsilon = self.epsilon_init

            # Initialize replay memory
            memory = ReplayMemory(self.replay_memory_size)

            # Create the target network and make it identical to the policy network
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Policy network optimizer. "Adam" optimizer can be swapped to something else.
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            # List to keep track of epsilon decay
            epsilon_history = []

            # Track number of steps taken. Used for syncing policy => target network.
            step_count=0

            # Track best reward
            best_reward = -9999999
        else:
            # Load learned policy
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            # switch model to evaluation mode
            policy_dqn.eval()

        # Train INDEFINITELY, manually stop the run when you are satisfied (or unsatisfied) with the results
        for episode in itertools.count():

            state, _ = env.reset()  # Initialize environment. Reset returns (state,info).
            state = torch.tensor(state, dtype=torch.float, device=device) # Convert state to tensor directly on device

            terminated = False      # True when agent reaches goal or fails
            episode_reward = 0.0    # Used to accumulate rewards per episode

            # Perform actions until episode terminates or reaches max rewards
            # (on some envs, it is possible for the agent to train to a point where it NEVER terminates, so stop on reward is necessary)
            while(not terminated and episode_reward < self.stop_on_reward):

                # Select action based on epsilon-greedy
                if is_training and random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    # select best action
                    with torch.no_grad():
                        # state.unsqueeze(dim=0): Pytorch expects a batch layer, so add batch dimension i.e. tensor([1, 2, 3]) unsqueezes to tensor([[1, 2, 3]])
                        # policy_dqn returns tensor([[1], [2], [3]]), so squeeze it to tensor([1, 2, 3]).
                        # argmax finds the index of the largest element.
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Execute action. Truncated and info is not used.
                new_state,reward,terminated,truncated,info = env.step(action.item())

                # Accumulate rewards
                episode_reward += reward

                # Convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    # Save experience into memory
                    memory.append((state, action, new_state, reward, terminated))

                    # Increment step counter
                    step_count+=1

                # Move to the next state
                state = new_state

            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(episode_reward)

            # Save model when new best reward is obtained.
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward


                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(memory)>self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count=0
```
- Next, I write a method for  save graph 
```
def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)
```

- next, i write a method for optimize the network
```
def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # Calculate target Q values (expected returns)
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
            '''
                target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                    .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                        [0]             ==> tensor([3,6])
            '''

        # Calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
                actions.unsqueeze(dim=1)
                .gather(1, actions.unsqueeze(dim=1))  ==>
                    .squeeze()                    ==>
        '''

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases
```

- this is our final main calling function
```
if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
```

## Results and evoluation 
- 06-15 21:16:23: Training starting...
- 06-15 21:16:34: New best reward -6.9 (-100.0%) at - - - episode 0, saving model...
- 06-15 21:16:34: New best reward -5.7 (-17.4%) at episode 1, saving model...
- 06-15 21:16:35: New best reward -3.9 (-31.6%) at episode 30, saving model...
- 06-15 21:16:35: New best reward -2.1 (-46.2%) at episode 32, saving model...
- 06-15 21:16:43: New best reward -1.5 (-28.6%) at episode 958, saving model...
- 06-15 21:16:48: New best reward 3.3 (-320.0%) at episode 1523, saving model...
- 06-15 21:16:52: New best reward 3.9 (+18.2%) at episode 1935, saving model...
- 06-15 21:17:57: New best reward 4.0 (+2.6%) at episode 9189, saving model...
- 06-15 21:18:15: New best reward 4.2 (+5.0%) at episode 11036, saving model...
- 06-15 21:18:27: New best reward 4.8 (+14.3%) at episode 12231, saving model...
- 06-15 21:18:30: New best reward 4.9 (+2.1%) at episode 12484, saving model...
- 06-15 21:18:57: New best reward 6.5 (+32.7%) at episode 15071, saving model...
- 06-15 21:19:06: New best reward 8.4 (+29.2%) at episode 15937, saving model...
- 06-15 21:21:06: New best reward 9.4 (+11.9%) at episode 26098, saving model...
- 06-15 22:36:35: New best reward 10.7 (+13.8%) at episode 240814, saving model...
- 06-15 22:36:36: New best reward 11.0 (+2.8%) at episode 240882, saving model...
- 06-15 22:36:37: New best reward 12.9 (+17.3%) at episode 240931, saving model...
- 06-15 22:39:38: New best reward 15.0 (+16.3%) at episode 247469, saving model...
- 06-15 22:41:46: New best reward 17.9 (+19.3%) at episode 251675, saving model...
- 06-15 22:42:37: New best reward 18.8 (+5.0%) at episode 253192, saving model...
- 06-15 22:42:48: New best reward 22.9 (+21.8%) at episode 253493, saving model...
- 06-15 22:44:35: New best reward 23.5 (+2.6%) at episode 256924, saving model...
- 06-15 22:44:53: New best reward 26.9 (+14.5%) at episode 257482, saving model...
- 06-15 22:44:58: New best reward 31.9 (+18.6%) at episode 257792, saving model...
- 06-15 22:46:36: New best reward 34.6 (+8.5%) at episode 260551, saving model...
- 06-15 22:46:56: New best reward 36.4 (+5.2%) at episode 261078, saving model...
- 06-15 22:53:26: New best reward 40.9 (+12.4%) at episode 271818, saving model...
- 06-15 22:57:06: New best reward 41.0 (+0.2%) at episode 278161, saving model...
- 06-15 23:04:49: New best reward 48.7 (+18.8%) at episode 290927, saving model...
- 06-15 23:14:08: New best reward 60.7 (+24.6%) at episode 304692, saving model...
- 06-16 01:08:16: New best reward 64.4 (+6.1%) at episode 458928, saving model...
- 06-16 02:21:54: New best reward 68.9 (+7.0%) at episode 522084, saving model...
- 06-16 06:01:32: New best reward 80.9 (+17.4%) at episode 656794, saving model...
- 06-16 08:31:42: New best reward 82.9 (+2.5%) at episode 733104, saving model...
- 06-16 13:25:42: New best reward 85.7 (+3.4%) at episode 845852, saving model...
- 06-16 20:21:55: New best reward 106.4 (+24.2%) at episode 941863, saving model...
