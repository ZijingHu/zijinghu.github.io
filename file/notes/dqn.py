"""
Code tested using
    1. gymnasium 0.29.1
    2. box2d-py  2.3.5
    3. pytorch   2.1.2
    4. Python    3.10.12
1 & 2 can be installed using pip install gymnasium[box2d]
"""
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
from collections import deque, namedtuple
import matplotlib.pyplot as plt


class ExperienceReplay:
    """ 
    Based on the Replay Buffer implementation of TD3 
    Reference: https://github.com/sfujim/TD3/blob/master/utils.py
    """
    def __init__(self, state_dim, action_dim, max_size, batch_size,gpu_index=0):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))     
        self.batch_size = batch_size
        self.device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')


    def add(self, state, action,reward,next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).long().to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )



class QNetwork(nn.Module):
    """
    Q Network: designed to take state as input and give out Q values of actions as output
    """

    def __init__(self, state_dim, action_dim):
        """
            state_dim (int): state dimenssion
            action_dim (int): action dimenssion
        """
        super(QNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)
        
    def forward(self, state):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(q))
        return self.l3(q)



class DQNAgent():

    def __init__(self,
     state_dim, 
     action_dim,
     discount=0.99,
     tau=1e-3,
     lr=5e-4,
     update_freq=4,
     max_size=int(1e5),
     batch_size=64,
     gpu_index=0
     ):
        """
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            discount (float): discount factor
            tau (float): used to update q-target
            lr (float): learning rate
            update_freq (int): update frequency of target network
            max_size (int): experience replay buffer size
            batch_size (int): training batch size
            gpu_index (int): GPU used for training
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lr = lr
        self.update_freq = update_freq
        self.batch_size = batch_size
        self.device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')


        # Setting up the NNs
        self.Q = QNetwork(state_dim, action_dim).to(self.device)
        self.Q_target = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.lr)

        # Experience Replay Buffer
        self.memory = ExperienceReplay(state_dim,1,max_size,self.batch_size,gpu_index)
        
        self.t_train = 0
    
    def step(self, state, action, reward, next_state, done):
        """
        1. Adds (s,a,r,s') to the experience replay buffer, and updates the networks
        2. Learns when the experience replay buffer has enough samples
        3. Updates target netowork
        """
        self.memory.add(state, action, reward, next_state, done)       
        self.t_train += 1 

        if self.memory.size > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.discount) #To be implemented
        
        if (self.t_train % self.update_freq) == 0:
            self.target_update(self.Q, self.Q_target, self.tau) #To be implemented 

    def select_action(self, state, epsilon=0.):
        """
        TODO: Complete this block to select action using epsilon greedy exploration 
        strategy
        Input: state, epsilon
        Return: Action
        Return Type: int    
        """
        if random.random() < epsilon:
            action = random.choice(np.arange(self.action_dim))
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.Q.eval()
            with torch.no_grad():
                Q_s = self.Q(state)
            self.Q.train()
            action = np.argmax(Q_s.cpu().data.numpy())
        return action

    def learn(self, experiences, discount):
        """
        TODO: Complete this block to update the Q-Network using the target network
        1. Compute target using  self.Q_target ( target = r + discount * max_b [Q_target(s,b)] )
        2. Compute Q(s,a) using self.Q
        3. Compute MSE loss between step 1 and step 2
        4. Update your network
        Input: experiences consisting of states,actions,rewards,next_states and discount factor
        Return: None
        """         
        states, actions, rewards, next_states, dones = experiences

        # DQN
        q_target_sa = self.Q_target(next_states).detach().max(axis=1)[0].unsqueeze(1)
        y = rewards + discount * q_target_sa * (1 - dones)
        q_sa = self.Q(states).gather(1, actions)
        
        # DDQN
        # q_pred = self.Q(next_states)
        # action_q_pred = torch.argmax(q_pred, dim=1).long().unsqueeze(1)
        # q_target_sa = self.Q_target(next_states).gather(1, action_q_pred)
        # y = rewards + discount * q_target_sa * (1 - dones)
        # q_sa = self.Q(states).gather(1, actions)
        
        # loss backprop
        self.optimizer.zero_grad()
        loss = F.mse_loss(q_sa, y)
        loss.backward()
        self.optimizer.step()

    def target_update(self, Q, Q_target, tau):
        """
        TODO: Update the target network parameters (param_target) using current Q parameters (param_Q)
        Perform the update using tau, this ensures that we do not change the target network drastically
        1. param_target = tau * param_Q + (1 - tau) * param_target
        Input: Q,Q_target,tau
        Return: None
        """ 

        for Q_param, Q_target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            Q_target_param.data.copy_(tau*Q_param.data + (1-tau)*Q_target_param.data)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="LunarLander-v2")             # Gymnasium environment name
    parser.add_argument("--seed", default=0, type=int)                 # sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--n-episodes", default=2000, type=int)        # maximum number of training episodes
    parser.add_argument("--batch-size", default=64, type=int)          # training batch size
    parser.add_argument("--discount", default=0.99, type=float)        # discount factor
    parser.add_argument("--lr", default=5e-4, type=float)              # learning rate
    parser.add_argument("--tau", default=0.001, type=float)            # soft update of target network
    parser.add_argument("--max-size", default=int(1e5),type=int)       # experience replay buffer length
    parser.add_argument("--update-freq", default=4, type=int)          # update frequency of target network
    parser.add_argument("--gpu-index", default=0,type=int)             # GPU index
    parser.add_argument("--max-esp-len", default=1000, type=int)       # maximum time of an episode
    #exploration strategy
    parser.add_argument("--epsilon-start", default=1, type=float)      # start value of epsilon
    parser.add_argument("--epsilon-end", default=0.01, type=float)     # end value of epsilon
    parser.add_argument("--epsilon-decay", default=0.995, type=float)  # decay value of epsilon
    #experiment name
    parser.add_argument("--exp-name", default='dqn')                   # name of experiment
    args = parser.parse_args()

    # making the environment    
    env = gym.make(args.env)

    #setting seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    kwargs = {
        "state_dim":state_dim,
        "action_dim":action_dim,
        "discount":args.discount,
        "tau":args.tau,
        "lr":args.lr,
        "update_freq":args.update_freq,
        "max_size":args.max_size,
        "batch_size":args.batch_size,
        "gpu_index":args.gpu_index
    }   
    learner = DQNAgent(**kwargs) #Creating the DQN learning agent
    window = []
    moving_window = deque(maxlen=100)
    epsilon = args.epsilon_start
    for e in range(args.n_episodes):
        state, _ = env.reset()
        curr_reward = 0
        for t in range(args.max_esp_len):
            action = learner.select_action(state,epsilon) #To be implemented
            n_state,reward,terminated,truncated,_ = env.step(action)
            done = terminated or truncated 
            learner.step(state,action,reward,n_state,done) #To be implemented
            state = n_state
            curr_reward += reward
            if done:
                break
        window.append(curr_reward)
        moving_window.append(curr_reward)
        """"
        TODO: Write code for decaying the exploration rate using args.epsilon_decay
        and args.epsilon_end. Note that epsilon has been initialized to args.epsilon_start  
        1. You are encouraged to try new methods
        """
        epsilon = max(args.epsilon_end, args.epsilon_decay*epsilon)
        
        if e % 100 == 0:
            print('Episode Number {} Average Episodic Reward (over 100 episodes): {:.2f}'.format(e, np.mean(moving_window)))
        
        """"
        TODO: Write code for
        1. Logging and plotting
        2. Rendering the trained agent 
        """
    
    folder_name = args.env+'-'+args.exp_name
    env = gym.make(args.env, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=folder_name, name_prefix=args.exp_name)

    state = env.reset()[0]
    env.start_video_recorder()
    done = False
    with torch.no_grad():
        while not done:
            action = learner.select_action(state,epsilon)
            n_state,reward,terminated,truncated,_ = env.step(action)
            done = terminated or truncated
            state = n_state
            env.render()
        env.close()
    w = 10
    plt.plot(range(len(window)), window, c='orange')
    plt.plot(range(w-1, len(window)), np.convolve(window, np.ones(w), 'valid') / w, c='blue')
    plt.savefig('./'+folder_name+'/'+args.exp_name+'.png')
