import numpy as np
import heapq

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
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        self.policy = np.zeros(grid_world.get_state_space(), dtype=int)  # pi(s)

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for convergence

        Args:
            threshold (float): threshold for convergence
        """
        self.threshold = threshold

    def get_policy(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy
        """
        return self.policy

    def get_values(self) -> np.ndarray:
        """Return the values

        Returns:
            np.ndarray: values
        """
        return self.values

    def get_q_value(self, state: int, action: int) -> float:
        """Get the q-value for a state and action

        Args:
            state (int)
            action (int)

        Returns:
            float
        """
        # TODO: Get reward from the environment and calculate the q-value
        next_state, reward, done = self.grid_world.step(state, action)
        return reward + (1-done)*self.discount_factor * self.values[next_state]
        raise NotImplementedError


class IterativePolicyEvaluation(DynamicProgramming):
    def __init__(
        self, grid_world: GridWorld, policy: np.ndarray, discount_factor: float
    ):
        """Constructor for IterativePolicyEvaluation

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): policy (probability distribution state_spacex4)
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)
        self.policy = policy

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float: value
        """
        # TODO: Get the value for a state by calculating the q-values
        val = 0

        for action in range(self.grid_world.get_action_space()):
            val += self.policy[state][action] * self.get_q_value(state, action)

        return val
        raise NotImplementedError

    def evaluate(self) -> None:
        """Evaluate the policy and update the values for one iteration"""
        # TODO: Implement the policy evaluation step
        while True:
            delta = 0
            v = np.copy(self.get_values())
            for state in range(v.shape[0]):
                v[state] = self.get_state_value(state)
            diff = np.max(np.abs(self.get_values()-v))
            self.values = v
            delta = max(delta, diff)

            if(delta < self.threshold):
                break
        return
        raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the iterative policy evaluation algorithm until convergence
        self.evaluate()
        return
        raise NotImplementedError


class PolicyIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        action = self.policy[state]
        val = self.get_q_value(state, action)
        return val
        raise NotImplementedError

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        while True:
            delta = 0
            v = np.copy(self.get_values())
            for state in range(v.shape[0]):
                v[state] = self.get_state_value(state)

            diff = np.max(np.abs(self.get_values()-v))
            self.values = v
            delta = max(delta, diff)

            if(delta < self.threshold):
                break
        return
        raise NotImplementedError

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        stable = True
        for state in range(self.grid_world.get_state_space()):
            old = np.copy(self.policy[state])
            q_s = np.zeros(self.grid_world.get_action_space())
            for action in range(self.grid_world.get_action_space()):
                q_s[action] = self.get_q_value(state, action)
            best = np.argmax(q_s)
            self.policy[state] = best
            if np.any(old != self.policy[state]):
                stable = False
        return stable
        raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the policy iteration algorithm until convergence
        self.policy = np.random.randint(4, size=(self.grid_world.get_state_space(),))
        while True:
            self.policy_evaluation()
            stable = self.policy_improvement()
            if stable:
                break
        return
        raise NotImplementedError


class ValueIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        q_s = np.zeros(self.grid_world.get_action_space())
        for action in range(self.grid_world.get_action_space()):
            q_s[action] = self.get_q_value(state, action)
        return np.max(q_s)
        raise NotImplementedError

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        while True:
            delta = 0
            v = np.copy(self.get_values())
            for state in range(self.grid_world.get_state_space()):
                best = self.get_state_value(state)
                delta = max(delta, abs(best - self.get_values()[state]))
                v[state] = best
            self.values = v
            if(delta < self.threshold):
                break
        return
        raise NotImplementedError

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        self.policy = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
            q_s = np.zeros(self.grid_world.get_action_space())
            for action in range(self.grid_world.get_action_space()):
                q_s[action] = self.get_q_value(state,action)
            best = np.argmax(q_s)
            self.policy[state] = best
        return
        raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        self.policy_evaluation()
        self.policy_improvement()
        return
        raise NotImplementedError


class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)
        self.model = {}
    
    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        q_s = np.zeros(self.grid_world.get_action_space())
        for action in range(self.grid_world.get_action_space()):
            q_s[action] = self.get_q_value(state, action)
        return np.max(q_s)
        raise NotImplementedError

    def policy_evaluation(self):    #count=242
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        while True:
            delta = 0
            for state in range(self.grid_world.get_state_space()):
                best = self.get_state_value(state)
                delta = max(delta, abs(best - self.get_values()[state]))
                self.values[state] = best
            if(delta < self.threshold):
                break
        return
        raise NotImplementedError

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        self.policy = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
            q_s = np.zeros(self.grid_world.get_action_space())
            for action in range(self.grid_world.get_action_space()):
                if len(self.model) == 0:
                    q_s[action] = self.get_q_value(state,action)
                else:
                    reward, next_state, done = self.model[(state, action)]
                    q_s[action] = reward + (1-done)*self.discount_factor * self.values[next_state]
            best = np.argmax(q_s)
            self.policy[state] = best
        return
        raise NotImplementedError
    
    def prioritized_sweeping(self):
        pq = []
        s_p = {}    #state to error mapping
        predecessors = {}

        for state in range(self.grid_world.get_state_space()):
            q_s = np.zeros(self.grid_world.get_action_space())
            for action in range(self.grid_world.get_action_space()):
                next_state, reward, done = self.grid_world.step(state, action)
                self.model[(state, action)] = (reward, next_state, done)
                q_s[action] = reward + self.discount_factor * self.values[next_state]*(1-done)
                # Record predecessors
                if next_state in predecessors:
                  predecessors[next_state].add(state)
                else:
                  predecessors[next_state] = {state}
            best = np.max(q_s)
            error = abs(best - self.values[state])
            if error > self.threshold:
                heapq.heappush(pq, (-error, state))
                s_p[state] = -error
                
        #count = 0   #calc updata times of self.values
        while len(pq) > 0:
            _, s = heapq.heappop(pq)
            q_s = np.zeros(self.grid_world.get_action_space())
            for action in range(self.grid_world.get_action_space()):
                reward, next_state, done = self.model[(s, action)]
                q_s[action] = reward + self.discount_factor * self.values[next_state]*(1-done)
            best = np.max(q_s)
            self.values[s] = best
            #count+=1
            for pre in predecessors[s]:
                q_s = np.zeros(self.grid_world.get_action_space())
                for action in range(self.grid_world.get_action_space()):
                    reward, next_state, done = self.model[(pre, action)]
                    q_s[action] = reward + self.discount_factor * self.values[next_state]*(1-done)
                best = np.max(q_s)
                error = abs(best - self.values[pre])
            
                if error > self.threshold:
                    if (s_p[pre], pre) not in pq:
                        heapq.heappush(pq, (-error, pre))
                        s_p[pre] = -error
                    else:
                        index = pq.index((s_p[pre], pre))
                        pq[index] = (-error, pre)
                        s_p[pre] = -error
        #print(count)
        return
    
    def realTime(self):
        def sample():
            history = []
            visited = set()
            state = np.random.randint(self.grid_world.get_state_space())
            while True:
                action = self.policy[state]
                history.append((state, action))
                next_state, _, done = self.grid_world.step(state, action)
                if done:
                    history.append((next_state, None))
                    break
                visited.add(state)

                if next_state in visited:
                    break
                state = next_state
            return history
        self.policy = np.random.randint(4, size = (self.grid_world.get_state_space(), ))
        max_iter = 90
        delta = 0
        for i in range(max_iter):
            history = sample()
            for s, a in history:
                q_s = np.zeros(self.grid_world.get_action_space())
                for action in range(self.grid_world.get_action_space()):
                    q_s[action] = self.get_q_value(s, action)
                best = np.max(q_s)
                best_p = np.argmax(q_s)
                delta = max(delta, abs(best - self.get_values()[s]))
                self.values[s] = best
                self.policy[s] = best_p
            if(delta < self.threshold):
                break
        return   
        raise NotImplementedError
    
    def heuristic(self):
        pq = []
        s_p = {}
        predecessors = {}
        h = {}
        
        for state in range(self.grid_world.get_state_space()):
            q_s = np.zeros(self.grid_world.get_action_space())
            for action in range(self.grid_world.get_action_space()):
                next_state, reward, done = self.grid_world.step(state, action)
                self.model[(state, action)] = (reward, next_state, done)
                q_s[action] = reward + self.discount_factor * self.values[next_state]*(1-done)
                # Record predecessors
                if next_state in predecessors:
                  predecessors[next_state].add(state)
                else:
                  predecessors[next_state] = {state}
            best = np.max(q_s)
            h[state] = best # use v(s) to initialize h(s)
            error = abs(best - self.values[state])
            if error > self.threshold:
                heapq.heappush(pq, (-error, state))
                s_p[state] = -error
                
        #count = 0
        while len(pq) > 0:
            d, s = heapq.heappop(pq)
            q_s = np.zeros(self.grid_world.get_action_space())
            for action in range(self.grid_world.get_action_space()):
                reward, next_state, done = self.model[(s, action)]
                q_s[action] = reward + self.discount_factor * h[next_state]*(1-done)
            best = np.max(q_s)
            h[s] = best
            self.values[s] = best
            #count+=1
        
            for pre in predecessors[s]:
                q_s = np.zeros(self.grid_world.get_action_space())
                for action in range(self.grid_world.get_action_space()):
                    reward, next_state, done = self.model[(pre, action)]
                    q_s[action] = reward + self.discount_factor * h[next_state]*(1-done)
                best = np.max(q_s)
                error = abs(best - self.values[pre])
                
                if error > self.threshold:
                    if (s_p[pre], pre) not in pq:
                        heapq.heappush(pq, (-error, pre))
                        s_p[pre] = -error
                    else:
                        index = pq.index((s_p[pre], pre))  # current index in the priority queue.
                        pq[index] = (-error, pre)
                        s_p[pre] = -error
        #print(count)
        return


    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence

        # Method 1: In-Place
        #self.policy_evaluation()

        # Method 2: Prioritized_sweeping (with model)
        #self.prioritized_sweeping()

        # Method 3: Real Time DP 
        #self.realTime()

        #Method 4: Heuristic + Prioritized_sweeping
        self.heuristic()
        self.policy_improvement()
        return
        raise NotImplementedError
