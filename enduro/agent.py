import sys
import numpy as np

from ale_python_interface import ALEInterface
from enduro.action import Action
from enduro.control import Controller
from enduro.state import StateExtractor


class Agent(object):
    def __init__(self):
        self._ale = ALEInterface()
        self._ale.setInt('random_seed', 123)
        self._ale.setFloat('repeat_action_probability', 0.0)
        self._ale.setBool('color_averaging', False)
        self._ale.loadROM('roms/enduro.bin')
        self._controller = Controller(self._ale)
        self._extractor = StateExtractor(self._ale)
        self._image = None
        self._speed_range = 50

    def run(self, learn, episodes=1, draw=False):
        """ Implements the playing/learning loop.

        Args:
            learn(bool): Whether the self.learn() function should be called.
            episodes (int): The number of episodes to run the agent for.
            draw (bool): Whether to overlay the environment state on the frame.

        Returns:
            None
        """
        for e in range(episodes):
            self._relative_speed = -self._speed_range

            # Observe the environment to set the initial state
            (road, cars, grid, self._image) = self._extractor.run(draw=draw, scale=4.0)
            self.initialise(road, cars, self._relative_speed, grid)

            num_frames = self._ale.getFrameNumber()

            # Each episode lasts 6500 frames
            while self._ale.getFrameNumber() - num_frames < 6500:
                # Take an action
                self.act()

                # Update the environment grid
                (road, cars, grid, self._image) = self._extractor.run(draw=draw, scale=4.0)

                if self.collision(cars):
                    self._relative_speed = -self._speed_range

                self.sense(road, cars, self._relative_speed, grid)

                # Perform learning if required
                if learn:
                    self.learn()

                self.callback(learn, e + 1, self._ale.getFrameNumber() - num_frames)
            self._ale.reset_game()

    def collision(self, cars):
        if not cars['others']:
            return False

        x, y, _, _ = cars['self']

        min_dist = sys.float_info.max
        min_angle = 0.

        for c in cars['others']:
            cx, cy, _, _ = c
            dist = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                min_angle = np.arctan2(y - cy, cx - x)

        return min_dist < 18. and 0.1 * np.pi < min_angle and min_angle < 0.9 * np.pi

    def getActionsSet(self):
        """ Returns the set of all possible actions
        """
        return [Action.ACCELERATE, Action.RIGHT, Action.LEFT, Action.BRAKE]

    def move(self, action):
        """ Executes the action and advances the game to the next state.

        Args:
            action (int): The action which should executed. Make sure to use
                          the constants returned by self.getActionsSet()

        Returns:
           int: The obtained reward after executing the action
        """

        if action == Action.ACCELERATE:
            self._relative_speed = min(self._relative_speed + 1,
                                       self._speed_range)
        elif action == Action.BRAKE:
            self._relative_speed = max(self._relative_speed - 1,
                                       -self._speed_range)

        return self._controller.move(action)

    def initialise(self, road, cars, speed, grid):
        """ Called at the beginning of each episode, mainly used
        for state initialisation. For more information on the arguments
        have a look at the README.md

        Args:
            road: 2-dimensional array containing [x, y] points
                  in pixel coordinates of the road grid
            cars: dictionary which contains the location and the size
                  of the agent and the opponents in pixel coordinates
            speed: the relative speed of the agent with respect the others
            gird:  2-dimensional numpy array containing the latest grid
                   representation of the environment

        Returns:
            None
        """
        raise NotImplementedError

    def act(self):
        """ Called at each loop iteration to choose and execute an action.

        Returns:
            None
        """
        raise NotImplementedError

    def sense(self, road, cars, speed, grid):
        """ Called at each loop iteration to construct the new state from
        the update environment grid. For more information on the arguments
        have a look at the README.md

        Args:
            road: 2-dimensional array containing [x, y] points
                  in pixel coordinates of the road grid
            cars: dictionary which contains the location and the size
                  of the agent and the opponents in pixel coordinates
            speed: the relative speed of the agent with respect the others
            gird: 2-dimensional numpy array containing the latest grid
                  representation of the environment
        Returns:
            None
        """
        raise NotImplementedError

    def learn(self):
        """ Called at each loop iteration when the agent is learning. It should
        implement the learning procedure.

        Returns:
            None
        """
        raise NotImplementedError

    def callback(self, learn, episode, iteration):
        """ Called at each loop iteration mainly for reporting purposes.

        Args:
            learn (bool): Indicates whether the agent is learning or not.
            episode (int): The number of the current episode.
            iteration (int): The number of the current iteration.

        Returns:
            None
        """

        raise NotImplementedError
