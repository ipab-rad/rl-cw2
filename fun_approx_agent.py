import sys
import cv2
import numpy as np
import pickle

from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState


class FunctionApproximationAgent(Agent):
    def __init__(self):
        super(FunctionApproximationAgent, self).__init__()

        self.grid_cols = 10
        self.horizon_row = 5
        self.fov_width = 5
        self.num_features = 9

        self.total_reward = 0
        self.speed_range = 50

        # Intialise theta optimistically with small positive values
        self.theta = 0.1 * np.ones((self.num_features + 1, 1))

        self.action_index = {Action.ACCELERATE: 0, Action.RIGHT: 1,
                             Action.LEFT: 2, Action.BRAKE: 3}
        self.index_action = {i: a for a, i in self.action_index.iteritems()}

        self.alpha = 0.01
        self.gamma = 0.9
        self.epsilon = 0.01

        self.log = []

    def initialise(self, road, cars, speed, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.
        """
        # Keep a log used for plotting later
        self.log.append((self.total_reward, self.theta))

        self.total_reward = 0
        self.next_state = self.buildState(road, cars, speed, grid)

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """
        self.state = self.next_state

        # Use softmax for action selection
        if np.random.uniform(0., 1.) < self.epsilon:
            Qs = self.Qs(self.state)
            probs = np.exp(Qs) / np.sum(np.exp(Qs))
            idx = np.random.choice(4, p=probs)
            self.action = self.index_action[idx]
        else:
            self.action = self.argmaxQs(self.state)

        self.reward = self.move(self.action)
        self.total_reward += self.reward

    def sense(self, road, cars, speed, grid):
        """ Constructs the next state from sensory signals.

        gird -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """
        self.next_state = self.buildState(road, cars, speed, grid)

        # Visualise the environment grid
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))

    def learn(self):
        """ Performs the learning procedure. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        Qsa = self.Q(self.state, self.action)
        maxQ = self.maxQs(self.next_state)
        error = self.reward + self.gamma * maxQ - Qsa
        self.error = error
        grad = self.gradQ(self.state, self.action)
        self.theta = self.theta + self.alpha * error * grad

    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        if iteration % 100 == 0:
            print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)

        # Perform learning rate decay every 1000 iterations
        # in order to stabilise the learning process
        if iteration % 1000 == 0:
            self.alpha = 0.99 * self.alpha

        if episode % 10 == 0:
            cv2.imshow("Enduro", self._image)
            cv2.waitKey(20)

    def buildState(self, road, cars, speed, grid):
        state = np.zeros(3)

        # Agent position
        [[_, x]] = np.argwhere(grid[:self.horizon_row, :] == 2)
        state[0] = x

        # Closest agent position
        opp_grid = grid[:self.horizon_row]
        rows = np.sum(opp_grid, axis=1)
        rows[0] -= 2
        rows = np.sort(np.argwhere(rows > 0).flatten())
        state[1] = -1

        if rows.size > 0:
            row = rows[0]
            for i, g in enumerate(grid[row, :]):
                if g == 1:
                    state[1] = i
                    break

        # Relative speed
        state[2] = speed

        return state

    def collision(self, cars):
        dist, angl = self.closestCar(cars)
        return dist < 18. and 0.1 * np.pi < angl and angl < 0.9 * np.pi

    def collisionPotential(self, grid):
        [[_, x]] = np.argwhere(grid[:self.horizon_row, :] == 2)

        l = x - self.fov_width / 2
        r = x + self.fov_width / 2
        while l < 0:
            l += 1
            r += 1

        while r >= self.grid_cols:
            l -= 1
            r -= 1

        return int(np.sum(grid[:, l:r + 1]) - 2)

    def closestCar(self, cars):
        if not cars['others']:
            return 200, np.pi / 2.

        x, y, _, _ = cars['self']

        min_dist = sys.float_info.max
        min_angle = 0.

        for c in cars['others']:
            cx, cy, _, _ = c
            dist = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                min_angle = np.arctan2(y - cy, cx - x)

        return min_dist, min_angle

    def features(self, s, a):
        (agent_x, opponent_x, relative_speed) = s

        # Initialise the features vector to 0s
        features = np.zeros(self.num_features + 1)

        # Positive relative speed
        features[0] = 1. if relative_speed >= 40 else 0.

        # Negative relative speed
        features[1] = 1. if relative_speed <= -40 else 0.

        # Potential collision if the agent is closer than 3 squares
        potential_collision = (opponent_x > -1 and
                               np.fabs(agent_x - opponent_x) < 3)
        features[2] = 1. if not potential_collision else -1.

        # Avoid opponent on the left
        if potential_collision and opponent_x > agent_x:
            if a == Action.LEFT:
                features[3] = 1.
            else:
                features[3] = -1.

        # Avoid opponent on the  right
        if potential_collision and opponent_x < agent_x:
            if a == Action.RIGHT:
                features[4] = 1.
            else:
                features[4] = -1.

        # Avoid opponent in front
        if potential_collision and opponent_x == agent_x:
            if a == Action.RIGHT or a == Action.LEFT:
                features[5] = 1.
            else:
                features[5] = -1.

        # Avoid stopping when no potential collision
        features[6] = 0.
        if not potential_collision:
            if a == Action.ACCELERATE:
                features[6] = 1.
            else:
                features[6] = -1.

        # Move to the center from the left edge
        features[7] = 0.
        if agent_x < 4:
            if a == Action.RIGHT:
                features[7] = 1.
            else:
                features[7] = -1.

        # Move to the center from the right edge
        features[8] = 0.
        if agent_x > 5:
            if a == Action.LEFT:
                features[8] = 1.
            else:
                features[8] = -1.

        # Add offset
        features[9] = 1.

        return features

    def Q(self, s, a):
        return np.asscalar(np.dot(self.features(s, a), self.theta))

    def gradQ(self, s, a):
        return self.features(s, a).reshape(self.num_features + 1, 1)

    def Qs(self, s):
        return np.asarray(
            map(lambda a: self.Q(s, a), self.getActionsSet()))

    def maxQs(self, s):
        return np.max(self.Qs(s))

    def argmaxQs(self, s):
        return self.index_action[np.argmax(self.Qs(s))]


if __name__ == "__main__":
    a = FunctionApproximationAgent()
    a.run(True, episodes=500, draw=True)
    pickle.dump(a.log, open("log.p", "wb"))
