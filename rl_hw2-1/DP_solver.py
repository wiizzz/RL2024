import numpy as np
import json
from collections import defaultdict

from gridworld import GridWorld


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
        self.values       = np.zeros(self.state_space)

    def get_all_state_values(self) -> np.array:
        return self.values


class MonteCarloPrediction(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float):
        """Constructor for MonteCarloPrediction

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)

    def run(self) -> None:
        """Run the algorithm until self.grid_world.check() == False"""
        # TODO: Update self.values with first-visit Monte-Carlo method
        current_state = self.grid_world.reset()
        trajectory = []
        returns = {str(i): [] for i in range(self.state_space)}

        while self.grid_world.check():
            next_state, reward, done = self.grid_world.step()
            G = 0
            trajectory.append((current_state, reward))
            current_state = next_state
            if done:
                for idx, step in enumerate(trajectory[::-1]):
                    G = self.discount_factor*G + step[1]
                    if step[0] not in np.array(trajectory[::-1])[:,0][idx+1:]:
                        returns[str(step[0])].append(G)
                        self.values[step[0]] = np.mean(returns[str(step[0])])
                trajectory = []


class TDPrediction(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world, discount_factor)
        self.lr     = learning_rate

    def run(self) -> None:
        """Run the algorithm until self.grid_world.check() == False"""
        # TODO: Update self.values with TD(0) Algorithm  
        current_state = self.grid_world.reset()
        while self.grid_world.check():
            next_state, reward, done = self.grid_world.step()
            self.values[current_state] += self.lr * (reward + (1-done)*self.discount_factor * self.values[next_state] - self.values[current_state])
            current_state = next_state   
        

class NstepTDPrediction(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, num_step: int):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world, discount_factor)
        self.lr     = learning_rate
        self.n      = num_step

    def run(self) -> None:
        """Run the algorithm until self.grid_world.check() == False"""
        # TODO: Update self.values with N-step TD Algorithm

        current_state = self.grid_world.reset()
        while self.grid_world.check():
            S, R = [0], [0]
            t, T = 0, float('inf')
            while True:
                if t < T:
                    if self.grid_world.check() == False:
                        return
                    next_state, reward, done = self.grid_world.step()
                    S.append(next_state)
                    R.append(reward)
                    if done:
                        T = t+1
                tau = t - self.n + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau+1, min(tau+self.n+1, T+1)):
                        G += self.discount_factor**(i-tau-1)*R[i]
                    if tau + self.n < T:
                        G += self.discount_factor**self.n*self.values[S[tau+self.n]]
                    self.values[S[tau]] += self.lr*(G - self.values[S[tau]])
                if tau == T - 1:
                    break
                t += 1
