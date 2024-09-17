import numpy as np
from collections import deque
from gridworld import GridWorld
import random


class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.q_values     = np.zeros((self.state_space, self.action_space))  
        self.policy       = np.ones((self.state_space, self.action_space)) / self.action_space 
        self.policy_index = np.zeros(self.state_space, dtype=int)

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index
    
    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values



class MonteCarloPolicyIteration(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for MonteCarloPolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_eval_improve(self, state_trace, action_trace, reward_trace) -> None:
        """Evaluate the policy and update the values after one episode"""
        # TODO: Evaluate state value for each Q(s,a)
        G = 0
        for s, a, r in zip(state_trace[::-1], action_trace[::-1], reward_trace[::-1]):
            G = self.discount_factor*G + r
            self.q_values[s][a] += self.lr*(G-self.q_values[s][a])
        for state in state_trace:
            pi = np.zeros(self.action_space)
            idx = np.argmax(self.q_values[state])
            pi[idx] = 1
            self.policy[state] = pi
        

    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        # TODO: Improve the policy
        raise NotImplementedError

    def get_action(self, state, first):
        self.get_policy_index()
        if np.random.uniform(0,1) <= self.epsilon:
            action = np.random.randint(4)
        elif first:
            sample = np.random.choice(4,1, p=self.policy[state])
            action = sample[0]
        else:
            action = self.policy_index[state]
        return action


    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Monte Carlo policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        state_trace   = [current_state]
        action_trace  = []
        reward_trace  = []
        first = True
        self.policy_index = self.get_policy_index()
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            while True:
                action = self.get_action(current_state, first)
                next_state, reward, done = self.grid_world.step(action)
                action_trace.append(action)
                reward_trace.append(reward)
                current_state = next_state
                if done:
                    first = False
                    self.policy_eval_improve(state_trace, action_trace, reward_trace)
                    state_trace = [current_state]
                    action_trace  = []
                    reward_trace  = []
                    break
                state_trace.append(current_state)

            iter_episode += 1


class SARSA(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for SARSA

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        # TODO: Evaluate Q value after one step and improve the policy
        self.q_values[s][a] += self.lr * (r + (1-is_done)*self.discount_factor*self.q_values[s2][a2]-self.q_values[s][a])
        policy = np.zeros(4)
        i = np.argmax(self.q_values[s])
        policy[i] = 1
        self.policy[s] = policy
    
    def get_action(self, state):
        if np.random.uniform(0,1) <= self.epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(self.policy[state])
        return action

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the TD policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        action = self.get_action(current_state)
        is_done = False
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            while True:
                next_state, reward, is_done = self.grid_world.step(action)
                next_action = self.get_action(next_state)
                self.policy_eval_improve(current_state, action, reward, next_state, next_action, is_done)
                current_state = next_state
                action = next_action
                if is_done:
                    break
            iter_episode += 1


class Q_Learning(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, buffer_size: int, update_frequency: int, sample_batch_size: int):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr                = learning_rate
        self.epsilon           = epsilon
        self.buffer            = deque(maxlen=buffer_size)
        self.update_frequency  = update_frequency
        self.sample_batch_size = sample_batch_size
        self.discount_factor   = discount_factor

    def add_buffer(self, s, a, r, s2, d) -> None:
        # TODO: add new transition to buffer
        raise NotImplementedError

    def sample_batch(self) -> np.ndarray:
        # TODO: sample a batch of index of transitions from the buffer
        raise NotImplementedError

    def policy_eval_improve(self, s, a, r, s2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        #TODO: Evaluate Q value after one step and improve the policy
        max_q = np.max(self.q_values[s2])
        self.q_values[s][a] += self.lr*(r + self.discount_factor*max_q*(1-is_done) - self.q_values[s][a])
        
        policy = np.zeros(4)
        i = np.argmax(self.q_values[s])
        policy[i] = 1
        self.policy[s] = policy

        #raise NotImplementedError
    
    def get_action(self, state):
        if np.random.uniform(0,1) <= self.epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(self.policy[state])
        return action

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Q_Learning algorithm
        iter_episode = 0
        current_state = self.grid_world.reset()
        transition_count = 0
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            while True:
                action = self.get_action(current_state)
                next_state, reward, done = self.grid_world.step(action)
                self.buffer.append((current_state, action, reward, next_state, done))
                transition_count += 1
                if transition_count % self.update_frequency == 0:
                    sample_n = self.sample_batch_size if len(self.buffer) >= self.sample_batch_size else len(self.buffer)
                    B = random.sample(self.buffer, sample_n)
                    for batch in B:
                        s, a, r, s2, done = batch
                        self.policy_eval_improve(s, a, r, s2, done)
                current_state = next_state
                if done: 
                    break
            iter_episode += 1

