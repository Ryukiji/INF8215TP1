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
import time

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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

##
# this method uses the DFS algorithm to find a path till the final state
# 
# @params problem: data of the maze 
# 
# @return a list of action that pac man has to do to reach the final state 
##
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    """
    node = problem.getStartState()
    if(problem.isGoalState(node)):
        return []
    fringe = util.Stack()
    fringe.push([node, 0, 0])
    explored =  []
    while (not fringe.isEmpty()):
        nodeWithParent = fringe.pop()
        node = nodeWithParent[0]
        parent = nodeWithParent[1]
        action = nodeWithParent[2]
        if (not checkIfExplored(explored, node)):
            explored.append((node, (parent, action)))
            if (problem.isGoalState(node)):
                return buildPath(problem, explored, node)
            for child, action, cost  in problem.getSuccessors(node):
                fringe.push([child, node, action])
    return "there is not solution"

##
# this method allows to know if a given node has already been explored
#
# @params explored is a list of couple: first element = node already explored, second element = parent of the key node and action corresponding
# @params node: we want to know if this node has already been explored
#
# @return boolean: true if the node has already been explored/ false otherwise
##
def checkIfExplored(explored, node):
    for alreadyExplored in explored:
        if(node == alreadyExplored[0]):
            return True
    return False

##
# this method allows to build the path that pac man has to follow if he wants to reach the final state
#
# @params problem: data of the maze
# @params explored is a list of couple: first element = node already explored, second element = parent of the key node and action corresponding
# @params goalState: coordonates of the final state
# 
# @return a list of action that pac man has to do to reach the final state 
##
def buildPath(problem, explored, goalState):
    actionsList = []
    node = goalState
    while(problem.getStartState() != node):
        node, action = getAction(explored, node)
        actionsList.insert(0,action)
    return actionsList

##
# this method allows to find a node in explored and to return his corresponding data
#
# @params explored is a list of couple: first element = node already explored, second element = parent of the key node and action corresponding
# @params node: it is the node whose information is sought
#
# @return the parent of the node, and the action done to go from the parent to the node
##
def getAction(explored, node):
    for element in explored:
        if element[0] == node:
            return element[1]

##
# this method uses the BFS algorithm to find a path till the final state
# 
# @params problem: data of the maze 
# 
# @return a list of action that pac man has to do to reach the final state 
##
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    node = problem.getStartState()
    if(problem.isGoalState(node)):
        return []
    fringe = util.Queue()
    fringe.push(node)
    explored =  [(node,0)]
    while (not fringe.isEmpty()):
        node = fringe.pop()
        if (problem.isGoalState(node)):
            return buildPath(problem, explored, node)
        for child, action, cost  in problem.getSuccessors(node):
            if((checkIfExplored(explored, child)) == False):
                fringe.push(child)
                explored.append((child,(node, action)))

    return "there is not solution"

##
# this method uses the UCS algorithm to find a path till the final state
# 
# @params problem: data of the maze 
# 
# @return a list of action that pac man has to do to reach the final state 
##
def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    node = problem.getStartState()
    if(problem.isGoalState(node)):
        return []    
    
    fringe = util.PriorityQueue()
    fringe.push((node, []) ,0)
    explored =  []
    while(not fringe.isEmpty()):
        node, actions = fringe.pop()
        if problem.isGoalState(node):
            return actions
        if node not in explored:
            for child in problem.getSuccessors(node):
                location = child[0]
                nextAction = child[1]
                newActions = actions + [nextAction]
                fringe.push((location, newActions), problem.getCostOfActions(newActions))
        explored.append(node)
    return actions

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

##
# this method uses the A* algorithm to find a path till the final state
# 
# @params problem: data of the maze 
# @params heuristic: method that define the heuristic used in the A* algorithm
# 
# @return a list of action that pac man has to do to reach the final state 
##
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    node = problem.getStartState()
    if(problem.isGoalState(node)):
        return []    
    
    fringe = util.PriorityQueue()
    fringe.push((node, []) ,0)
    explored =  []
    while(not fringe.isEmpty()):
        node, actions = fringe.pop()
        if problem.isGoalState(node):
            return actions
        if node not in explored:
            children = problem.getSuccessors(node)
            for c in children:
                location = c[0]
                nextAction = c[1]
                newActions = actions + [nextAction]
                fringe.push((location, newActions), problem.getCostOfActions(newActions) + heuristic(location, problem))
        explored.append(node)
    return actions

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
