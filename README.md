![](vis.gif)

Code for https://www.youtube.com/watch?v=ZNbIKv9gCOg

A maze [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process) solved with [value iteration](https://en.wikipedia.org/wiki/Bellman_equation). Maze walls are also available to render.

The maze is a 2D grid of cells / grid boxes. Each cell is a state. There are five actions for each cell: up, down, left, right, and staying still. Moving into a wall is -100 reward and bumps the player back into the cell he/she was in, moving into another cell (or staying still) is -1 reward and reaching the end cell (in the bottom right corner) is +10000 reward.

A policy is just a way of making decisions. It assigns each grid cell (state) an action to take. For example, my policy could include always moving right whenever I'm in the top left square/grid cell and left for every other grid cell. It can't change, so if my policy is to always move right from the top left cell I will always move right whenever I find myself there.

The whiter a square, the greater the expected future reward of the optimal policy (or at least approximately, the video approaches the true value over time).

The Bellman equation is a key part of modern reinforcement learning. Watch how the expected value of cells propagates from the bottom right all the way to the top left!
