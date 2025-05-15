"""
Code tested using
    1. gymnasium 0.27.1
    2. box2d-py  2.3.5
    3. pytorch   2.0.0
    4. Python    3.9.12
1 & 2 can be installed using pip install gymnasium[box2d]
"""
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import math
import random
import argparse
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class value_network(nn.Module):
    '''
    Value Network: Designed to take in state as input and give value as output
    Used as a baseline in Policy Gradient (PG) algorithms
    '''
    def __init__(self,state_dim):
        '''
            state_dim (int): state dimenssion
        '''
        super(value_network, self).__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)

    def forward(self,state):
        '''
        Input: State
        Output: Value of state
        '''
        v = F.tanh(self.l1(state))
        v = F.tanh(self.l2(v))
        return self.l3(v)


class policy_network(nn.Module):
    '''
    Policy Network: Designed for continous action space, where given a 
    state, the network outputs the mean and standard deviation of the action
    '''
    def __init__(self,state_dim,action_dim,log_std = 0.0):
        """
            state_dim (int): state dimenssion
            action_dim (int): action dimenssion
            log_std (float): log of standard deviation (std)
        """
        super(policy_network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.l1 = nn.Linear(state_dim,64)
        self.l2 = nn.Linear(64,64)
        self.mean = nn.Linear(64,action_dim)
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    
    def forward(self,state):
        '''
        Input: State
        Output: Mean, log_std and std of action
        '''
        a = F.tanh(self.l1(state))
        a = F.tanh(self.l2(a))
        a_mean = self.mean(a)
        a_log_std = self.log_std.expand_as(a_mean)
        a_std = torch.exp(a_log_std)        
        return a_mean, a_log_std, a_std

    def select_action(self, state):
        '''
        Input: State
        Output: Sample drawn from a normal disribution with mean and std
        '''
        a_mean, _, a_std = self.forward(state)
        action = torch.normal(a_mean, a_std)
        return action
    
    def get_log_prob(self, state, action):
        '''
        Input: State, Action
        Output: log probabilities
        '''
        mean, log_std, std = self.forward(state)
        var = std.pow(2)
        log_density = -(action - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
        return log_density.sum(1, keepdim=True)


class PGAgent():
    '''
    An agent that performs different variants of the PG algorithm
    '''
    def __init__(self,
     state_dim, 
     action_dim,
     discount=0.99,
     lr=1e-3,
     gpu_index=0,
     seed=0,
     env="LunarLander-v2"
     ):
        """
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            discount (float): discount factor
            lr (float): learning rate
            gpu_index (int): GPU used for training
            seed (int): Seed of simulation
            env (str): Name of environment
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.lr = lr
        self.device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
        self.env_name = env
        self.seed = seed
        self.policy = policy_network(state_dim,action_dim)
        self.value = value_network(state_dim)
        self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_value = torch.optim.Adam(self.value.parameters(), lr=self.lr)

    def sample_traj(self,batch_size=2000,evaluate = False):
        '''
        Input: 
            batch_size: minimum batch size needed for update
            evaluate: flag to be set during evaluation
        Output:
            states, actions, rewards,not_dones, episodic reward     
        '''
        self.policy.to("cpu") #Move network to CPU for sampling
        env = gym.make(args.env,continuous=True)
        states = []
        actions = []
        rewards = []
        n_dones = []
        curr_reward_list = []
        while len(states) < batch_size:
            state, _ = env.reset(seed=self.seed)
            curr_reward = 0
            for t in range(1000):
                state_ten = torch.from_numpy(state).float().unsqueeze(0)
                with torch.no_grad():
                    if evaluate:
                        action = self.policy(state_ten)[0][0].numpy() # Take mean action during evaluation
                    else:
                        action = self.policy.select_action(state_ten)[0].numpy() # Sample from distribution during training
                action = action.astype(np.float64)
                n_state,reward,terminated,truncated,_ = env.step(action) # Execute action in the environment
                done = terminated or truncated
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                n_done = 0 if done else 1
                n_dones.append(n_done)
                state = n_state
                curr_reward += reward
                if done:
                    break
            curr_reward_list.append(curr_reward)
        if evaluate:
            return np.mean(curr_reward_list)
        return states,actions,rewards,n_dones, np.mean(curr_reward_list)
    

    def update(self,states,actions,rewards,n_dones,update_type='Baseline'):
        '''
        Inputs:
            states: list of states
            actions: list of actions
            rewards: list of rewards
            n_dones: list of not dones
            update_type: type of PG algorithm
        Output: 
            None
        '''
        self.policy.to(self.device) #Move policy to GPU
        if update_type == "Baseline":
            self.value.to(self.device)  #Move value to GPU
        states_ten = torch.from_numpy(np.stack(states)).to(self.device)   #Convert to tensor and move to GPU
        action_ten = torch.from_numpy(np.stack(actions)).to(self.device)  #Convert to tensor and move to GPU
        rewards_ten = torch.from_numpy(np.stack(rewards)).to(self.device) #Convert to tensor and move to GPU
        n_dones_ten = torch.from_numpy(np.stack(n_dones)).to(self.device) #Convert to tensor and move to GPU

        if update_type == "Rt":
            '''
            TODO: Peform PG using the cumulative discounted reward of the entire trajectory
            1. Compute the discounted reward of each trajectory (rt)
            2. Compute log probabilities using states_ten and action_ten
            3. Compute policy loss and update the policy
            '''
            # Do steps 1-3
            n_traj = (n_dones_ten==0).sum() + int(n_dones_ten[-1]!=0)
            rt = torch.zeros(n_traj).to(self.device)
            index = torch.zeros(n_traj+1).int()
            c = 1
            rt[-c] = rewards_ten[-1].item()
            index[-c] = len(n_dones_ten)
            for i in range(n_dones_ten.shape[0]-2, -1, -1):
                if n_dones_ten[i] == 0:
                    c += 1
                    index[-c] = i + 1
                rt[-c] = rewards_ten[i] + self.discount*rt[-c]
            rt = (rt - rt.mean()) / rt.std()
            loss = 0
            for i in range(c):
                log_prob = self.policy.get_log_prob(states_ten[index[i]: index[i+1]], action_ten[index[i]: index[i+1]])
                loss += -1/c * torch.sum(log_prob * rt[i])
            self.optimizer_policy.zero_grad()
            loss.backward()
            self.optimizer_policy.step()

        if update_type == 'Gt':
            '''
            TODO: Peform PG using reward_to_go
            1. Compute reward_to_go (gt) using rewards_ten and n_dones_ten
            2. gt should be of the same length as rewards_ten
            3. Compute log probabilities using states_ten and action_ten
            4. Compute policy loss and update the policy
            '''
            gt = torch.zeros(rewards_ten.shape[0],1).to(self.device)

            n_traj = (n_dones_ten==0).sum() + int(n_dones_ten[-1]!=0)
            gt = torch.zeros(n_dones_ten.shape[0], 1).to(self.device)
            c = 0
            index = torch.zeros(n_traj+1).int()
            for i in range(n_dones_ten.shape[0]-1, -1, -1):
                if n_dones_ten[i] == 0:
                    c += 1
                    index[-c] = i + 1
                    g = rewards_ten[i]
                g = rewards_ten[i] + self.discount*g
                gt[i] = g

            gt = (gt - gt.mean()) / gt.std() #Helps with learning stablity

            loss = 0
            for i in range(c):
                log_prob = self.policy.get_log_prob(states_ten[index[i]: index[i+1]], action_ten[index[i]: index[i+1]])
                loss += -1/c * torch.sum(log_prob * gt[index[i]: index[i+1]])
            self.optimizer_policy.zero_grad()
            loss.backward()
            self.optimizer_policy.step()

        if update_type == 'Baseline':
            '''
            TODO: Peform PG using reward_to_go and baseline
            1. Compute values of states, this will be used as the baseline 
            2. Compute reward_to_go (gt) using rewards_ten and n_dones_ten
            3. gt should be of the same length as rewards_ten
            4. Compute advantages 
            5. Update the value network to predict gt for each state (L2 norm)
            6. Compute log probabilities using states_ten and action_ten
            7. Compute policy loss (using advantages) and update the policy
            '''
            with torch.no_grad():
                values_adv = self.value(states_ten)
            gt = torch.zeros(rewards_ten.shape[0],1).to(self.device)

            # Compute reward_to_go (gt) and advantages 
            n_traj = (n_dones_ten==0).sum() + int(n_dones_ten[-1]!=0)
            c = 0
            index = torch.zeros(n_traj+1).int()
            for i in range(n_dones_ten.shape[0]-1, -1, -1):
                if n_dones_ten[i] == 0:
                    c += 1
                    index[-c] = i + 1
                    g = rewards_ten[i]
                g = rewards_ten[i] + self.discount*g
                gt[i] = g
            advantages = gt - values_adv

            advantages = (advantages - advantages.mean()) / advantages.std()

            loss = 0
            for i in range(c):
                log_prob = self.policy.get_log_prob(states_ten[index[i]: index[i+1]], action_ten[index[i]: index[i+1]])
                loss += -1/c * torch.sum(log_prob * advantages[index[i]: index[i+1]])
            self.optimizer_policy.zero_grad()
            loss.backward()
            self.optimizer_policy.step()
            vloss = F.mse_loss(self.value(states_ten), gt)
            self.optimizer_value.zero_grad()
            vloss.backward()
            self.optimizer_value.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="LunarLander-v2")           # Gymnasium environment name
    parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--n-iter", default=200, type=int)           # Maximum number of training iterations
    parser.add_argument("--discount", default=0.99, type=float)      # Discount factor
    parser.add_argument("--batch-size", default=5000, type=int)      # Training samples in each batch of training
    parser.add_argument("--lr", default=5e-3,type=float)             # Learning rate
    parser.add_argument("--gpu-index", default=0,type=int)           # GPU index
    parser.add_argument("--algo", default="Baseline",type=str)       # PG algorithm type. Baseline/Gt/Rt
    parser.add_argument("--v", default="",type=str)                  # Exp id
    args = parser.parse_args()

    # Making the environment    
    env = gym.make(args.env,continuous=True)

    # Setting seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    kwargs = {
        "state_dim":state_dim,
        "action_dim":action_dim,
        "discount":args.discount,
        "lr":args.lr,
        "gpu_index":args.gpu_index,
        "seed":args.seed,
        "env":args.env
    }   
    learner = PGAgent(**kwargs) # Creating the PG learning agent

    moving_window = deque(maxlen=10)
    x = []
    y1 = []
    y2 = []

    for e in range(args.n_iter):
        '''
        Steps of PG algorithm
            1. Sample environment to gather data using a policy
            2. Update the policy using the data
            3. Evaluate the updated policy
            4. Repeat 1-3
        '''
        states,actions,rewards,n_dones,train_reward = learner.sample_traj(batch_size=args.batch_size)
        learner.update(states,actions,rewards,n_dones,args.algo)
        eval_reward= learner.sample_traj(evaluate=True)
        moving_window.append(eval_reward)
        print('Training Iteration {} Training Reward: {:.2f} Evaluation Reward: {:.2f} \
        Average Evaluation Reward: {:.2f}'.format(e,train_reward,eval_reward,np.mean(moving_window)))
        x.append(e)
        y1.append(eval_reward)
        y2.append(np.mean(moving_window))
        """
        TODO: Write code for
        1. Logging and plotting
        2. Rendering the trained agent 
        """
        
    folder_name = args.env+'-'+args.algo+('-'+args.v if args.v else '')
    env = gym.make(args.env, continuous=True, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=folder_name, name_prefix=args.algo)

    learner.policy.to("cpu") #Move network to CPU for sampling

    state, _ = env.reset(seed=learner.seed)
    env.start_video_recorder()
    for t in range(1000):
        state_ten = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action = learner.policy(state_ten)[0][0].numpy()
        action = action.astype(np.float64)
        n_state,reward,terminated,truncated,_ = env.step(action) # Execute action in the environment
        done = terminated or truncated
        n_done = 0 if done else 1
        state = n_state
        env.render()
        if done:
            break
    env.close()
    plt.plot(x, y1, c='orange')
    plt.plot(x, y2, c='blue')
    plt.savefig('./'+folder_name+'/'+args.algo+'.png')