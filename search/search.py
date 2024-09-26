# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    
    stack = util.Stack()
    stack.push((problem.getStartState(), [], 0))
    visited = [problem.getStartState()]

    while not stack.isEmpty():
        node, path, cost = stack.pop()
        visited = visited + [node]
        if problem.isGoalState(node):
            return path
        for node2, path2, cost2 in problem.getSuccessors(node):
            if node2 not in stack.list and node2 not in visited:
                stack.push([node2, path + [path2], cost])
    return False

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    
    queue = util.Queue()
    queue.push((problem.getStartState(), [], 0))

    visited = [problem.getStartState()]

    while not queue.isEmpty():
        node, path, cost = queue.pop()
        if problem.isGoalState(node):
            return path
        for node2, path2, cost2 in problem.getSuccessors(node):
            if node2 not in queue.list and node2 not in visited:
                queue.push([node2, path + [path2], cost])
                visited = visited + [node2]
    return False

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    
    queue = util.PriorityQueue()
    queue.push((problem.getStartState(), [], 0), 0)

    visited = [problem.getStartState()]

    while not queue.isEmpty():
        node, path, cost = queue.pop()
        if problem.isGoalState(node):
            return path
        """if node in visited:
            continue
        visited = visited + [node]"""
        for node2, path2, cost2 in problem.getSuccessors(node):
            if node2 not in queue.heap and node2 not in visited:
                queue.push([node2, path + [path2], cost2 + cost], cost2 + cost)
                visited = visited + [node2]
    return False

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    queue = util.PriorityQueue()
    cost = {}
    parents = {}
    directions = {}
    
    # start state
    start = problem.getStartState()
    queue.push(start, heuristic(start, problem))
    cost[start] = 0
    
    while not queue.isEmpty():
        # node with the lowest cost + heuristic
        current_node = queue.pop()
        
        # Check if we've reached the goal
        if problem.isGoalState(current_node):
            solution = []
            while current_node in parents:
                solution.append(directions[current_node])
                current_node = parents[current_node]
            return list(reversed(solution))
        
        # Expand
        for successor, action, step_cost in problem.getSuccessors(current_node):
            new_cost = cost[current_node] + step_cost
            
            # If the successor has not been visited or found a cheaper path
            if successor not in cost or new_cost < cost[successor]:
                cost[successor] = new_cost
                priority = new_cost + heuristic(successor, problem)
                queue.push(successor, priority)
                parents[successor] = current_node
                directions[successor] = action

    return []  

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
