

import time as t
from queue import Queue


class Basic_Elements:
    # Setting up various attributes needed for the puzzle to successfully run
    def __init__(self, key, state, parent, depth, gn, h_function, move, goal):
        self.key = key
        self.state = state
        self.parent = parent
        self.depth = depth
        self.gn = gn
        self.h_function = h_function
        self.move = move
        self.goal = goal
        self.hn = Basic_Elements.heuristic_cost(self.state, self.goal, self.h_function)
        self.total_cost = self.gn + self.hn
        self.moves_getter()

    def heuristic_cost(state, goal, func):
        heur_cost = 0

        # Misplaced tiles definition
        if func == "MisplacedTiles":
            for i in zip(state, goal):
                if i[0] != i[1]:
                    heur_cost += 1
                else:
                    continue
        # Manhattan distance definition
        elif func == "Manhattan":
            for i in goal:
                heur_cost += abs(goal.index(i) - state.index(i))
        elif func == "None":
            heur_cost = 0

        return heur_cost

    def moves_getter(self):
        self.moves = []
        # List of possible moves
        if self.state.index(0) == 0:
            self.moves.extend(("Left", "Up"))
        elif self.state.index(0) == 1:
            self.moves.extend(("Left", "Right", "Up"))
        elif self.state.index(0) == 2:
            self.moves.extend(("Right", "Up"))
        elif self.state.index(0) == 3:
            self.moves.extend(("Up", "Down", "Left"))
        elif self.state.index(0) == 4:
            self.moves.extend(("Up", "Down", "Left", "Right"))
        elif self.state.index(0) == 5:
            self.moves.extend(("Right", "Up", "Down"))
        elif self.state.index(0) == 6:
            self.moves.extend(("Down", "Left"))
        elif self.state.index(0) == 7:
            self.moves.extend(("Left", "Right", "Down"))
        else:
            self.moves.extend(("Right", "Down"))

    def m_piece(self, move):
        new_node = self.state[:]
        index_zero = new_node.index(0)
        # Setting up indexes after moving into different directions
        if move == "Left":
            new_index = index_zero + 1
        elif move == "Right":
            new_index = index_zero - 1
        elif move == "Up":
            new_index = index_zero + 3
        else:
            new_index = index_zero - 3

        new_value = self.state[new_index]
        new_node[index_zero] = new_value
        new_node[new_index] = 0

        return new_node, new_value


class Queue:
    # Setting up attributes for the queue
    def __init__(self, algorithm, goal_state):
        self.algorithm = algorithm
        self.queue = []

    def node_returned(self):
        # Specifying which type of nodes to be returned based on different kinds of costs
        if self.algorithm == "BreadthFirst":
            return self.queue[0]
        elif self.algorithm == "DepthFirst":
            return self.queue[-1]
        elif self.algorithm == "UniformCost":
            return sorted(self.queue, key=lambda x: x.gn)[0]
        elif self.algorithm == "BestFirst":
            return sorted(self.queue, key=lambda x: x.hn)[0]
        elif self.algorithm == "AStar":
            return sorted(self.queue, key=lambda x: x.total_cost)[0]
        elif self.algorithm == "IterativeDeepening":
            return sorted(self.queue, key=lambda x: x.depth)[0]


class Solver_Puzzle:
    # Setting up various attributes for the puzzle solver
    def __init__(self, original_node, goal_state, algorithm, itera_deepening):
        self.goal_state = original_node.goal
        self.root = original_node
        self.curr_node = original_node
        self.algorithm = algorithm
        self.heur_function = original_node.h_function
        self.limit = 0
        self.key = 0
        self.cnt_move = 0
        self.tree = {}
        self.visited = []
        self.cnt_depth = 0
        self.queue = Queue(self.algorithm, self.goal_state)
        self.queue.queue.append(self.root)
        self.itera_deepening = itera_deepening
        self.tree[0] = self.root
        self.solver()

    def solver(self):
        # Acquiring the current node in queue
        time_started = t.time()
        self.curr_node = self.queue.node_returned()

        while self.queue:
            length = []
            length.append(len(self.queue.queue))

            # Condition to check if the current state is equivalent to the desired goal state
            if self.curr_node.state != self.goal_state:
                if self.itera_deepening:
                    if self.cnt_depth > self.limit:
                        self.limit += 1
                        self.key = 0
                        self.cnt_move = 0
                        self.tree = {}
                        self.visited = []
                        self.cnt_depth = 0
                        self.queue = Queue(self.algorithm, self.goal_state)
                        self.queue.queue.append(self.root)
                        self.curr_node = self.root
                    else:
                        pass
                else:
                    pass

                # Repeated state
                if self.curr_node.state not in self.visited:
                    self.visited.append(self.curr_node.state[:])
                    self.cnt_move += 1

                    # Setting up the procedure to return moving cost and the new states
                    for move in self.curr_node.moves:
                        self.key += 1
                        new_state, gn = self.curr_node.m_piece(move)
                        gn += self.curr_node.gn
                        new_node = Basic_Elements(key=self.key, state=new_state, parent=self.curr_node.key, gn=gn,
                                                  depth=self.cnt_depth + 1, h_function=self.heur_function,
                                                  goal=self.goal_state, move=move)
                        self.tree[self.key] = new_node

                        # Checking on cost-based searches and their existing nodes' costs
                        if self.algorithm in ["UniformCost", "BestFirst", "AStar"]:
                            a = 0
                            if self.algorithm == "UniformCost":
                                sort = "gn"
                            elif self.algorithm == "AStar":
                                sort = "total_cost"
                            else:
                                sort = "hn"

                            for i in self.queue.queue:
                                if i.state == new_node.state:
                                    if getattr(i, sort) > getattr(new_node, sort):
                                        del self.queue.queue[a]
                                    else:
                                        a += 1
                                else:
                                    a += 1
                        else:
                            pass

                        self.queue.queue.append(new_node)

                    self.cnt_depth += 1
                    self.curr_node = self.queue.node_returned()

                else:
                    if self.algorithm == "DepthFirst":
                        index = -1
                    else:
                        index = 0

                    # Here defining how each algorithm is based on which type of costs: gn, hn or total
                    if self.algorithm == "UniformCost":
                        self.queue.queue = sorted(self.queue.queue, key=lambda x: x.gn)
                    elif self.algorithm == "BestFirst":
                        self.queue.queue = sorted(self.queue.queue, key=lambda x: x.hn)
                    elif self.algorithm == "AStar":
                        self.queue.queue = sorted(self.queue.queue, key=lambda x: x.total_cost)
                    else:
                        pass

                    # Important to get rid of the specified index item in the queue
                    del self.queue.queue[index]
                    self.curr_node = self.queue.node_returned()
            else:
                break

        time_ended = t.time()

        for i, j in self.tree.items():
            if j.state == self.goal_state:
                res = i
                break
            else:
                continue

                # Getting the correct path before reaching the root
        path = [res]
        while res != 0:
            path.insert(0, self.tree[res].parent)
            res = path[0]

        for i in path:
            print("Tile move:", self.tree[i].move, "\n", "Heuristic cost:", self.tree[i].hn, "\n",
                  "Total cost:", self.tree[i].gn, "\n", self.tree[i].state[0:3], "\n", self.tree[i].state[3:6], "\n",
                  self.tree[i].state[6:])

        # Minus 1 because the original state must not be counted
        print("Total number of moves:", len(path) - 1)

        print("Queue Length(maximum):", max(length), "\n", "Number of nodes popped:", self.cnt_move, "\n", "Duration:",
              time_ended - time_started)


# Setting up the UI
# Allowing players to choose difficulty level
print("Choose difficulty level: \nEasy \nMedium \nHard")

choice_difficulty = input()

if choice_difficulty == "Easy":
    state_1 = [1, 3, 4, 8, 6, 2, 7, 0, 5]
elif choice_difficulty == "Medium":
    state_1 = [2, 8, 1, 0, 4, 3, 7, 6, 5]
elif choice_difficulty == "Hard":
    state_1 = [5, 6, 7, 4, 0, 8, 3, 2, 1]

# Allowing players to choose which algorithm to use
print(
    "Choose algorithm: \nBreadth First Search \nDepth First Search \nUniform Cost Search \nIterative Deepening \nBest First Search \nA* Search")
choice_algo = input()
if choice_algo == "Breadth First Search":
    algorithm = "BreadthFirst"
    itera_deepening = False
    h_function = "None"
elif choice_algo == "Depth First Search":
    algorithm = "DepthFirst"
    itera_deepening = False
    h_function = "None"
elif choice_algo == "Uniform Cost Search":
    algorithm = "UniformCost"
    itera_deepening = False
    h_function = "None"
elif choice_algo == "Iterative Deepening":
    algorithm = "IterativeDeepening"
    itera_deepening = True
    h_function = "None"
elif choice_algo == "Best First Search":
    algorithm = "BestFirst"
    itera_deepening = False
    h_function = "None"
elif choice_algo == "A* Search":
    algorithm = "AStar"
    itera_deepening = False
    print("Choose heuristics: \nNumber of misplaced tiles \nManhattan Distance")
    choice_heur = input()
    if choice_heur == "Number of misplaced tiles":
        h_function = "MisplacedTiles"
    elif choice_heur == "Manhattan Distance":
        h_function = "Manhattan"
    else:
        print("Please try again")
else:
    print("Please try again")

goal = [1, 2, 3, 8, 0, 4, 7, 6, 5]

Result = Basic_Elements(key=0, state=state_1, parent=0, depth=0, gn=0, h_function=h_function, goal=goal,
                        move="Original state")
Process = Solver_Puzzle(Result, goal_state=goal, algorithm=algorithm, itera_deepening=False)