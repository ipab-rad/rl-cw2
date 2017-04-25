# Reinforcement Learning: Coursework 2

This repository contains the same agent interface for playing the Enduro game as the one you used in [coursework 1](https://github.com/ipab-rad/rl-cw1), however the sensing capabilities of the agent have been extended. Instead of just sensing the environment grid, the agent can now sense the road and the others in pixel coordinates as well as its own speed. The main difference is in the `sense` function which now has the following prototype:

`def sense(self, road, cars, speed, grid)`

These new sensory signals will help you quickly construct more complex state spaces, compared to the ones based only on the environment grid, which you may need for the function approximation based agent.

## Road grid

The road grid is 2-dimensional array which contains `[x, y]` points in pixel coordinates corresponding to the corners of the road cells used to construct the environment gird. Those are the cooridnates used to draw the white grid on top of the road in the game frames. There are 11x10 cells in the environment grid and so there are 12x11 points stored in the road grid. The first dimension of the road grid corresponds to the horizontal lines while the second dimension correspondes to the intersection points along a horizontal line. Thus, if you would like to access the pixel cooridinates of the top left corner of the furhtest leftmost road cell you would have to access `road[0][0]`.

## Cars

The `cars` argument is a dictionary which contains two keys `'self'` and `'others'`. `cars['self']` returns a rectangle as a tuple `(x, y, w, h)` which represents the agent location and size in the game frame.  `x, y` are the top-left corner pixel coordinates of the rectangle and its size is `w, h`. `cars['self']` is visualised as the green rectangle overlayed on the game frame. `cars['others']` is a list of tuples which countains the same information for each opponent present. If there are no opponents on the road, then `cars['others']` is an empty list. The information in `cars['others']` is visualised as red rectangles around the opponent.

## Speed

This is a single scalar in the range `[-50, 50]` which represents the speed of the agent relative to the opponents. Thus `-50` means that the agent has just collided and `50` means that the agent is moving as fast as possible.

## Environment grid

The `grid` argument is the same environment grid that you have already used during the first coursework.

## Solution & Results
The example ugent uses 9 features:

1. Positive relative speed
2. Negative relative speed
3. Potential collision
4. Avoid opponent on the left
5. Avoid opponent on the right
6. Avoid opponent in front
7. Accelerate when no potential collision
8. Move to the center of the road from the left edge
9. Move to the center of the road from the right edge

As you can see there are features which depend just on the state, just on the action and on both as well. If you run the agent you should obtain similar performance to the one bellow.

![Performance](https://raw.githubusercontent.com/ipab-rad/rl-cw2/master/figs/performance.png)

You can inspect the results with:

```
python plot_log.py
```
