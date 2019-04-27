import numpy as np
import random
import cv2
import pickle



# Height and width of maze (in cells)
H, W = 27, 43
# Pixels per side of maze square
PPS = 40
# Parameter affecting branching tendency of maze (higher <=> more branching)
ALT = 1

'''
Maze is a 2D list with each index containing another list
of form [0,0,1,0] or [up?, down?, left?, right?]
indicating whether or not one can move up, down, left or right.
[0,1,1,0] ==> down and left are moves without walls

Functions: 

    generateMaze(W, H) ==> 2D list of list [up?, down?, left?, right?]

    step(state, action, maze) ==> reward, new_state

    policyQ(policy, gamma, steps, maze) ==> Q value of (0,0) following policy
                                            for only 'steps' (default 100) steps

    prop(state, depth, gamma, maze) ==> Returns Q of optimal policy from (0,0) for depth steps.
                                        O(4^depth). **Terrible** and not fleshed out.

'''

def generateMaze(W=6, H=6, alt=False):
    # Returns maze to specs above

    def adjacent(maze, coord, visited=[]):
        # Return list of unvisited cells adjacent to coord
        H = len(maze)
        W = len(maze[0])
        considered = [(coord[0]-1, coord[1]),
                  (coord[0]+1, coord[1]),
                  (coord[0], coord[1]-1),
                  (coord[0], coord[1]+1)]
        result = []
        for adjacentCell in considered:
            if adjacentCell in visited:
                continue
            if adjacentCell[0] == -1 or adjacentCell[0] == H:
                continue
            if adjacentCell[1] == -1 or adjacentCell[1] == W:
                continue
            result.append(adjacentCell)
        return result

    maze = []
    for i in range(H):
        maze.append([])
        for j in range(W):
            maze[-1].append([0,0,0,0,0])

    lastCoord = (np.random.choice(H), np.random.choice(W))
    C = [lastCoord]
    visited = []

    while C:
        if alt is not False:
            if np.random.random() < alt:
                coord = random.choice(C)
                C.remove(coord)
                C.append(coord)
        next_cells = adjacent(maze=maze, coord=C[-1], visited=visited)
        if not next_cells:
            C.pop()
        else:
            next_cell = random.choice(next_cells)
            connection = (next_cell[0]-C[-1][0], next_cell[1]-C[-1][1])
            if connection == (-1,0):
                maze[C[-1][0]][C[-1][1]][0] = 1
                maze[next_cell[0]][next_cell[1]][1] = 1
            if connection == (1,0):
                maze[C[-1][0]][C[-1][1]][1] = 1
                maze[next_cell[0]][next_cell[1]][0] = 1
            if connection == (0,-1):
                maze[C[-1][0]][C[-1][1]][2] = 1
                maze[next_cell[0]][next_cell[1]][3] = 1
            if connection == (0,1):
                maze[C[-1][0]][C[-1][1]][3] = 1
                maze[next_cell[0]][next_cell[1]][2] = 1
            C.append(next_cell)
            visited.append(next_cell)
    return np.array(maze)

def step(SA, maze, H, W):
    # step(SA, maze) ==> reward, new_state
    # SA is state-action tuple
    if SA[2] == 4:
        new_state = (SA[0], SA[1])
        if new_state == (H-1,W-1):
            reward = 10000
        else:
            reward = -1
    if maze[SA[0:4]] == 1: # opening in direction of action
        if SA[2] == 0:
            new_state = (SA[0]-1, SA[1])
        if SA[2] == 1:
            new_state = (SA[0]+1, SA[1])
        if SA[2] == 2:
            new_state = (SA[0], SA[1]-1)
        if SA[2] == 3:
            new_state = (SA[0], SA[1]+1)
        if new_state == (H-1,W-1):
            reward = 10000
        else:
            reward = -1

    else: # bounce off wall
        new_state = SA[0:2]
        reward = -100
    return reward, new_state

def updateV(V, T, gamma=.999,):
    Vnew = np.zeros(V.shape)
    H, W = V.shape[0:2]
    for i in range(H):
        for j in range(W):
            Q_est = []
            for a in range(5): # possible action
                next_state = T[i,j,a,0:2]
                # reward + Q(next_state) for each possible action
                Q_est.append(T[i,j,a,2] + gamma*V[int(next_state[0]),int(next_state[1])][0])
            Vnew[i,j] = max(Q_est)
    return Vnew

def show(maze, PPS):
    H, W = maze.shape[0:2]
    transitions = np.zeros((*maze.shape, 3))
    # For each possible maze coordinate, action,
    # transitions stores [next_state[0], next_state[1], reward]
    for i in range(H):
        for j in range(W):
            for k in range(5):
                r, ns = step((i,j,k), maze, H=H, W=W) # reward, next_state
                transitions[i,j,k] = [*ns, r]

    T = transitions
    V = np.zeros((H, W, 1)) # Optimal Policy Value

    overlay1 = np.ones((PPS*H, PPS*W))
    overlay2 = np.zeros((PPS*H, PPS*W))
    for i in range(H):
            for j in range(W):
                if maze[i][j][0] == 0:
                    overlay1[i*PPS][j*PPS:(j+1)*PPS] = 0
                    overlay2[i*PPS][j*PPS:(j+1)*PPS] = .06
                if maze[i][j][1] == 0:
                    overlay1[(i+1)*PPS-1][j*PPS:(j+1)*PPS] = 0
                    overlay2[(i+1)*PPS-1][j*PPS:(j+1)*PPS] = .06
                if maze[i][j][2] == 0:
                    overlay1[(i*PPS):((i+1)*PPS),(j*PPS)] = 0
                    overlay2[(i*PPS):((i+1)*PPS),(j*PPS)] = .06
                if maze[i][j][3] == 0:
                    overlay1[(i*PPS):((i+1)*PPS),(j+1)*PPS-1] = 0
                    overlay2[(i*PPS):((i+1)*PPS),(j+1)*PPS-1] = .06

    # Uncomment below lines to save frames for video [i.e. the numpy arrays] in 'frames.pickle'.
    # Use video.py to convert these to an avi.

    # frames = []
    for i in range(2000):
        # frames.append(V)
        im = cv2.normalize(V, None, alpha=.01, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        im = np.kron(im, np.ones((PPS, PPS)))
        im = overlay2+overlay1*im
        cv2.imshow('Maze',im)
        V = updateV(V, T)
        if cv2.waitKey(1)==27:
            break

    # frames.append(overlay1)
    # frames.append(overlay2)

    # pickle_out = open("frames.pickle", "wb")
    # pickle.dump(frames, pickle_out)
    # pickle_out.close()

def main(H, W, PPS):
    maze = np.array(generateMaze(H=H, W=W, alt=ALT))
    show(maze, PPS=PPS)

if __name__ == '__main__':
    main(H, W, PPS)