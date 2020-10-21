import time
import numpy
from heapq import *
import numpy as np

t = 0 # Tie breaker element for heap.
# Coordinates of all elements of goal state, for simplifying heuristic calculation
goalStateCoordinates = {
    '0' : (0,0),
    '1' : (0,1),
    '2' : (0,2),
    '3' : (0,3),
    '4' : (1,0),
    '5' : (1,1),
    '6' : (1,2),
    '7' : (1,3),
    '8' : (2,0),
    '9' : (2,1),
    'A' : (2,2),
    'B' : (2,3),
    'C' : (3,0),
    'D' : (3,1),
    'E' : (3,2),
    'F' : (3,3)
}

def getNewHeuristic(old_state, new_state, i1, j1, i2, j2, old_h):
    """
    When we are generating neighbors,
    it is not necessary to go thru the entire board to calculate the heuristic val
    We can just update the old heuristic (old_h) based on the move that was played
    """
    old_h = old_h - ( (abs(goalStateCoordinates[old_state[i2,j2]][0] - i2)) + abs(goalStateCoordinates[old_state[i2,j2]][1] - j2) )
    old_h = old_h + ( (abs(goalStateCoordinates[new_state[i1,j1]][0] - i1)) + abs(goalStateCoordinates[new_state[i1,j1]][1] - j1) )

    return old_h


def getHeuristic(state):
    h_val = 0
    for i in range(4):
        for j in range(4):
            # Computing Manhattan-distace for the element in (i,j) position
            if state[i,j] != '0':
                h_val = h_val + abs(goalStateCoordinates[state[i,j]][0] - i) + abs(goalStateCoordinates[state[i,j]][1] - j)

    return h_val

def generateNeighbors(state):
    neighbors = []
    currStepCost = state[2]
    x_0 = state[3]
    y_0 = state[4]
    currState = state[5]
    old_h = state[0] - state[2]
    global t
    # Check if 'Up' is possible
    if x_0 > 0 and state[6] != 'Down':
        # Swap state[x0, y0] with state[x0-1, y0]
        new_state = np.copy(currState)
        new_state[x_0, y_0] = new_state[x_0-1, y_0]
        new_state[x_0-1, y_0] = '0'
        t-=1
        neighbors.append((getNewHeuristic(currState, new_state, x_0, y_0, x_0-1, y_0, old_h)+currStepCost+1, t, currStepCost+1, x_0-1, y_0, new_state, 'Up'))
    # Check if 'Down' is possible
    if x_0 < 3 and state[6] != 'Up':
        # Swap state[x0, y0] with state[x0+1, y0]
        new_state = np.copy(currState)
        new_state[x_0, y_0] = new_state[x_0+1, y_0]
        new_state[x_0+1, y_0] = '0'
        t-=1
        neighbors.append((getNewHeuristic(currState, new_state, x_0, y_0, x_0+1, y_0, old_h)+currStepCost+1, t, currStepCost+1, x_0+1, y_0, new_state, 'Down'))
    # Check if 'Left' is possible
    if y_0 > 0 and state[6] != 'Right':
        # Swap state[x0, y0] with state[x0, y0-1]
        new_state = np.copy(currState)
        new_state[x_0, y_0] = new_state[x_0, y_0-1]
        new_state[x_0, y_0-1] = '0'
        t-=1
        neighbors.append((getNewHeuristic(currState, new_state, x_0, y_0, x_0, y_0-1, old_h)+currStepCost+1, t, currStepCost+1, x_0, y_0-1, new_state, 'Left'))
    # Check if 'Right' is possible
    if y_0 < 3 and state[6] != 'Left':
        # Swap state[x0, y0] with state[x0, y0+1]
        new_state = np.copy(currState)
        new_state[x_0, y_0] = new_state[x_0, y_0+1]
        new_state[x_0, y_0+1] = '0'
        t-=1
        neighbors.append((getNewHeuristic(currState, new_state, x_0, y_0, x_0, y_0+1, old_h)+currStepCost+1, t, currStepCost+1, x_0, y_0+1, new_state, 'Right'))

    return neighbors



def FindMinimumPath(initialState,goalState):
    minPath=[] 
    nodesGenerated=0 
    heap = []
    goal = np.array(goalState)
    initial = np.array(initialState)
    explored = {}
        
    
    """
    The tuple which holds the state of the puzzle, contains the following information:
        1. The stepcost + heuristic value f(n)=g(n)+h(n)
        2. Tie-breaker element (t) for heap
        3. Step Cost
        4. x coordinate of zero-element in the state
        5. y coordinate of zero-element in the state
        6. The numpy array representing the state
        7. Operation in Transition Model which was performed to reach this state
    """
    x_zero = 0
    y_zero = 0
    for i in range(4):
        for j in range(4):
            if initial[i,j] == '0':
                x_zero = i
                y_zero = j

    global t
    t = 0
    init_tuple = (getHeuristic(initial), t, 0, x_zero, y_zero, initial, 'Null')
    explored[initial.tostring()] = init_tuple[0]

    parent = {}
    parent[init_tuple[5].tostring()] = ("-1", "-1")

    heappush(heap, init_tuple)
    while len(heap) != 0:
        heap_top = heappop(heap)
        # Goal Test
        if (heap_top[5] == goal).all():
            minPath.append(heap_top[6])
            break

        # Generate neighboring states, and push in heap
        neighbors = generateNeighbors(heap_top)

        for neighbor in neighbors:
            if neighbor[5].tostring() not in explored:
                nodesGenerated+=1
                explored[neighbor[5].tostring()] = neighbor[0]
                heappush(heap, neighbor)
                parent[neighbor[5].tostring()] = (heap_top[6], heap_top[5].tostring())
            elif explored[neighbor[5].tostring()] > neighbor[0]:
                nodesGenerated+=1
                explored[neighbor[5].tostring()] = neighbor[0]
                heappush(heap, neighbor)
                parent[neighbor[5].tostring()] = (heap_top[6], heap_top[5].tostring())
    
    c = goal.tostring()
    while parent[c][1] != "-1":
        minPath.append(parent[c][0])
        c = parent[c][1]

    minPath.pop()
    minPath.reverse()
    
    return minPath, nodesGenerated

def ReadInitialState():
    with open("initial_state.txt", "r") as file:
        initialState = [[x for x in line.split()] for i,line in enumerate(file) if i<4]
    return initialState

def ShowState(state,heading=''):
    print(heading)
    for row in state:
        print(*row, sep = " ")

def main():
    initialState = ReadInitialState()
    ShowState(initialState,'Initial state:')
    goalState = [['0','1','2','3'],['4','5','6','7'],['8','9','A','B'],['C','D','E','F']]
    ShowState(goalState,'Goal state:')
    
    start = time.time()
    minimumPath, nodesGenerated = FindMinimumPath(initialState,goalState)
    timeTaken = time.time() - start
    
    if len(minimumPath)==0:
        minimumPath = ['Up','Right','Down','Down','Left']
        print('Example output:')
    else:
        print('Output:')

    print('   Minimum path cost : {0}'.format(len(minimumPath)))
    print('   Actions in minimum path : {0}'.format(minimumPath))
    print('   Nodes generated : {0}'.format(nodesGenerated))
    print('   Time taken : {0} s'.format(round(timeTaken,4)))

if __name__=='__main__':
    main()
