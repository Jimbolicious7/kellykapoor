# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        curFood = currentGameState.getFood().count()
        if len(newFood.asList()) == curFood:
            score = 99999
            for food in newFood.asList():
                if manhattanDistance(food, newPos) < score:
                    score = manhattanDistance(food, newPos)
        else:
            score = 0
        for ghost in newGhostStates:
            score += 4 ** (2 - manhattanDistance(ghost.getPosition(), newPos))
        score = 99999 - score
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(index, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if index == 0:
                maxOfMin = max(minimax(1, depth, gameState.generateSuccessor(index, state)) for state in gameState.getLegalActions(index))
                return maxOfMin
            else:
                next = index + 1
                if gameState.getNumAgents() == next:
                    next = 0
                    depth += 1
                minOfMin = min(minimax(next, depth, gameState.generateSuccessor(index, state)) for state in gameState.getLegalActions(index))
                return minOfMin

        bigNum = float('-inf')
        move = Directions.STOP
        for curState in gameState.getLegalActions(0):
            value = minimax(1, 0, gameState.generateSuccessor(0, curState))
            if bigNum == float('-inf') or value > bigNum:
                bigNum = value
                move = curState
        return move

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maximizer(index, depth, game_state, alpha, beta):
            bigNum = float("-inf")
            for newState in game_state.getLegalActions(index):
                bigNum = max(bigNum, alphabeta(1, depth, game_state.generateSuccessor(index, newState), alpha, beta))
                if bigNum > beta:
                    return bigNum
                alpha = max(alpha, bigNum)
            return bigNum

        def minimizer(index, depth, game_state, alpha, beta):
            bigNum = float("inf")
            next = index + 1
            if game_state.getNumAgents() == next:
                next = 0
                depth += 1
            for state in game_state.getLegalActions(index):
                abcheck = alphabeta(next, depth, game_state.generateSuccessor(index, state), alpha, beta)
                bigNum = min(bigNum, abcheck)
                if bigNum < alpha:
                    return bigNum
                beta = min(beta, bigNum)
            return bigNum

        def alphabeta(index, depth, game_state, alpha, beta):
            if game_state.isLose() or game_state.isWin() or depth == self.depth:
                return self.evaluationFunction(game_state)
            if index == 0:
                return maximizer(index, depth, game_state, alpha, beta)
            else:
                return minimizer(index, depth, game_state, alpha, beta)

        utility = float("-inf")
        move = Directions.STOP
        alpha = float("-inf")
        beta = float("inf")
        for state in gameState.getLegalActions(0):
            value = alphabeta(1, 0, gameState.generateSuccessor(0, state), alpha, beta)
            if value > utility:
                utility = value
                move = state
            if utility > beta:
                return utility
            alpha = max(alpha, utility)
        return move

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(index, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if index == 0:
                return max(expectimax(1, depth, gameState.generateSuccessor(index, state)) for state in gameState.getLegalActions(index))
            else:
                next = index + 1
                if gameState.getNumAgents() == next:
                    next = 0
                    depth += 1
                nodes = sum(expectimax(next, depth, gameState.generateSuccessor(index, state)) for state in gameState.getLegalActions(index))
                actions = float(len(gameState.getLegalActions(index)))
                return  nodes / actions

        bigNum = float("-inf")
        action = Directions.STOP
        for state in gameState.getLegalActions(0):
            value = expectimax(1, 0, gameState.generateSuccessor(0, state))
            if value > bigNum or bigNum == float("-inf"):
                bigNum = value
                action = state
        return action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Find number of moves to nearest food pellet
    Create a variable which stores the sums of all distances to ghosts
    Find how many ghosts are within one space from Pacman
    Find number of available capsules
    Return a value which is near the evaluation function for the state, but it is increased slightly
    if Pacman is closer to a pellet, and is decreased slightly if Pacman is near to a ghost and if
    there are more capsules (to see if Pacman eats a capsule)
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    foodList = newFood.asList()
    minDist = -1
    for food in foodList:
        foodDist = util.manhattanDistance(newPos, food)
        if foodDist >= foodDist or minDist == -1:
            minDist = foodDist
    totalGhostDist = 1
    danger = 0
    for ghost in currentGameState.getGhostPositions():
        ghostDist = util.manhattanDistance(newPos, ghost)
        totalGhostDist += ghostDist
        if ghostDist <= 1:
            danger += 1
    capsules = len(currentGameState.getCapsules())
    minDistReciprocal = (1 / float(minDist))
    ghostDistReciprocal = (1 / float(totalGhostDist))
    return minDistReciprocal + currentGameState.getScore() - ghostDistReciprocal - danger - capsules


# Abbreviation
better = betterEvaluationFunction
