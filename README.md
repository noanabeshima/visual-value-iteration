Update: This old code is not beautiful or comprehensible. I've decided to leave it here because I'm still proud of the results.

![](vis.gif)

Code for https://www.youtube.com/watch?v=ZNbIKv9gCOg

A maze [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process) solved with [value iteration](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa11/slides/mdps-intro-value-iteration.pdf). Maze walls are also available to render in the code.

The maze is a 2D grid of cells / grid boxes. Each cell is a state. There are five actions for each cell: up, down, left, right, and staying still. Moving into a wall is -100 reward and bumps the player back into the cell he/she was in, moving into another cell (or staying still) is -1 reward and reaching the end cell (in the bottom right corner) is +10000 reward.

A policy is just a way of making decisions. It assigns each grid cell (state) to an action (up, down, left, right, still). For example, my policy could be always moving right whenever I'm in the top left square/grid cell and left for every other grid cell.

The whiter a square, the greater the estimated future reward of the optimal policy.

Watch how the estimated future reward of cells propagates from the bottom right all the way to the top left!
