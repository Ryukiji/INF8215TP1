#!/usr/bin/env python3
"""
Quoridor agent.
Copyright (C) 2013, 
<<<<<<<<<<<  
2162553 Corentin HUBERT 
2040063 Adam Mouttaki Bensouda 
>>>>>>>>>>>
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; version 2 of the License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.
"""
from quoridor import *
import random
import timeit
import math
class MyAgent(Agent):
    """My Quoridor agent."""
    def play(self, percepts, player, step, time_left):
        """
        This function is used to play a move according
        to the percepts, player and time left provided as input.
        It must return an action representing the move the player
        will perform.
        :param percepts: dictionary representing the current board
            in a form that can be fed to `dict_to_board()` in quoridor.py.
        :param player: the player to control in this step (0 or 1)
        :param step: the current step number, starting from 1
        :param time_left: a float giving the number of seconds left from the time
            credit. If the game is not time-limited, time_left is None.
        :return: an action
          eg: ('P', 5, 2) to move your pawn to cell (5,2)
          eg: ('WH', 5, 2) to put a horizontal wall on corridor (5,2)
          for more details, see `Board.get_actions()` in quoridor.py
        """
        print("percept:", percepts)
        print("player:", player)
        print("step:", step)
        print("time left:", time_left if time_left else '+inf')
        
        board = dict_to_board(percepts)
         
        actions= MyAgent.greatActions(board, player)
        depth = MyAgent.chooseDepth(len(actions), time_left)

        """
        If red player, best start is Mirror strategy, untill symetry is broken by blue
        """
        if player == 1:
            isSym, action = MyAgent.isBoardSymmetric(board)
            if isSym: return action
        
        

        """
        Choose Startegy for blue player
        """
        strategy = MyAgent.chooseStrategy(board, player, step, time_left)

        if(strategy in ('Charge','Finish')):
            action = MyAgent.chargeStrategy(board, player)
                        
        elif(strategy == 'Initiate'):
            action = MyAgent.initiateStrategy(board, player)    

        elif(strategy == 'AlphaBeta'):
            value, action = MyAgent.maxValue(board, player, -float("inf"), float("inf"), depth)
            # If the best action is to move, we prioritize charge strategy to put pressure
            if action[0] == "P":
                action = MyAgent.chargeStrategy(board, player)

        return action

    def maxValue(board, player, alpha, beta, depth):
        """
        The max value function of an alpha beta algorithm
        """
        
        if board.is_finished() or depth==0:
            return MyAgent.utility(board, player), 0
        bestValue= -float("inf")
        bestMove=[]
        actions= MyAgent.greatActions(board, player)
        for action in actions:
            newBoard = board.clone().play_action(action , player)
            value, move = MyAgent.minValue(newBoard, player, alpha, beta, depth)
            if value == bestValue:
                bestMove = bestMove + [action]
            if value > bestValue:
                bestValue = value
                bestMove = [action]
                alpha = max(alpha, bestValue)
            if bestValue > beta: return (bestValue, bestMove[0])
        
        return (bestValue, random.choice(bestMove))
    
    def minValue(board, player, alpha, beta, depth):
        """
        The min value function of an alpha beta algorithm
        """
        if board.is_finished():
            return MyAgent.utility(board, player), 0
        bestValue= float("inf")
        bestMove=[]
        actions= MyAgent.greatActions(board, MyAgent.changePlayer(player))
        for action in actions:
            newBoard = board.clone().play_action(action , MyAgent.changePlayer(player))
            value, move = MyAgent.maxValue(newBoard, player, alpha, beta, depth-1)
            if value == bestValue:
                bestMove = bestMove + [action]
            if value < bestValue:
                bestValue = value
                bestMove = [action]
                beta = min(beta, bestValue)
            if bestValue < alpha: return (bestValue, bestMove[0])
        return (bestValue, random.choice(bestMove))
        
    def utility(board, player):
        """
        This function is used to evaluate a given state.
        :param board: state that has to be evaluate
        :param player: the player that wants to maximize the result
        :return: the evaluation of the state 
        """
        opponent =MyAgent.changePlayer(player)
        try:
            opponentDistance = board.min_steps_before_victory(opponent)
            playerDistance = board.min_steps_before_victory(player)
        except NoPath: #if we don't know the distance, we prefer to not consider this move, so the move is badly graded
            playerDistance = 100000 
            opponentDistance = 0 
        if opponentDistance == 0:
            return -float("inf")
        if playerDistance == 0:
            return float("inf")
        
        distance_differential_from_finish = opponentDistance - playerDistance
        wall_differential = board.nb_walls[player]*MyAgent.valueWall(opponentDistance) - board.nb_walls[opponent]*MyAgent.valueWall(playerDistance)
        manhattanDistance_differential_from_finish = 0.15*MyAgent.manhattanDistance(board, opponent) - 0.15*MyAgent.manhattanDistance(board, player)
        
        utility = distance_differential_from_finish + wall_differential + manhattanDistance_differential_from_finish
        
        return utility

    def greatActions(board, player):
        """
        This function is used to select actions that we consider as interesting.
        It allows to limit the number of branches in the algorithm.
        :param board: state of the game
        :param player: the player whose turn it is to play
        :return: list of actions
        """
        actions = board.get_legal_pawn_moves(player)
        if board.nb_walls[player] > 0:
            [(coord10,coord11), (coord20,coord21)] = board.pawns
            wallCoordsPlayer = [(coord10,coord11),(coord10-1,coord11),(coord10,coord11-1),(coord10-1,coord11-1)]
            wallCoordsOpponent = [(coord20,coord21),(coord20-1,coord21),(coord20,coord21-1),(coord20-1,coord21-1)]
            
            for wallCoord in wallCoordsOpponent: #just to avoid duplicated element in case the players are close
                if wallCoord not in wallCoordsPlayer:
                    wallCoordsPlayer.append(wallCoord)
            for wallCoord in wallCoordsPlayer:
                if board.is_wall_possible_here(wallCoord , True):
                    actions.append(("WH", wallCoord[0], wallCoord[1]))
                if board.is_wall_possible_here(wallCoord , False) and MyAgent.is_wall_not_alone(board, wallCoord):
                    actions.append(("WV", wallCoord[0], wallCoord[1]))
            
            if len(actions) < 6: #If there are not enough action, we look further around the oppnement for more actions
                i,j = board.pawns[MyAgent.changePlayer(player)]
                wallCoordsOpponentFurther = [("WV", i-1, j+1), ("WV", i, j+1), ("WV", i-1, j-2), ("WV", i, j-2), ("WH", i-2, j-1), ("WH", i-2, j), ("WH", i+1, j), ("WH", i+1, j-1)]
                for wallCoord in wallCoordsOpponentFurther:
                    if board.is_action_valid(wallCoord, player):
                        actions.append(wallCoord)
        return actions
    
    def is_wall_not_alone(board, wallCoord):
        """
        This function is used to avoid selecting walls that we consider as not interesting.
        This wall are verticals walls that are not attached to another wall.
        :param board: the state of the game
        :param wallCoord: the wall that we want to know if it is interesting or not
        :return: a boolean, true if the wall is intersting; false if not
        """
        x,y = wallCoord
        if (x-2,y) in board.verti_walls or [x+2,y] in board.verti_walls:
            return True
        for i in [-1, 0, 1]:
            if (x+i,y+1) in board.horiz_walls or (x+i,y-1) in board.horiz_walls:
                return True
        return False

    def valueWall(distance):
        """
        This function is used to know the value of the wall that you have depending of the
        distance between your opponent and its finish line.
        :param distance: distance of the opponent player
        :return: the value of one wall
        """
        if distance == 1:
            return 0
        if distance == 2:
            return 0.35
        if distance == 3:
            return 0.7
        if distance == 4:
            return 0.95
        return (distance - 1) * 0.4

    def chooseStrategy(board, player, step, time):
        """
        This function decides for the player which strategy to take.
        It forces initation, finish and ingame strategy to optimize computation time.
        :param board: state of the game
        :param player: the player whose turn it is to play
        :param step: current step number
        :param time: time left ingame for the player
        :return: strategy for next action
        """
        if(step == 1):
            return 'Initiate'
        if(board.nb_walls[player] == 0 or time < 15):
            return 'Charge'  
        if(board.min_steps_before_victory(player) <= 1):
            return 'Finish'    
        return 'AlphaBeta'

    def chargeStrategy(board, player):
        """
        This function chooses the movement that makes the player most close to finish line.
        :param board: state of the game
        :param player: the player whose turn it is to play
        :return: next action closest to finish line
        """    
        actions = list(board.get_shortest_path(player))
        return ('P', actions[0][0], actions[0][1])

    def initiateStrategy(board, player):
        """
        This function makes the player move one step forward.
        It forces initation strategy to optimize computation time.
        It could've decided from an opening strategies database, but in our case, it is simple.
        :param board: state of the game
        :param player: the player whose turn it is to play
        :return: action to move one step forward
        """        
        actions = list(board.get_shortest_path(player))
        return ('P', 1, 4)

    def changePlayer(player):
        """
        This function is used to get the number of the other player.
        :param player: number of the player that is playing
        :return: the number of the player that waits his turn.
        """
        if player ==0: return 1
        return 0

    def isBoardSymmetric(board):
        """
        This function is used to know if the game can be symetric in the next step.
        If it is the case, it gives the move that has to be played to reached the symetric state
        :param board: the state of the game
        :return isSym: a boolean, true if the next state can be symetric
        :return move: the move that allow to reache a symetric state
        """
        p1Coord, p2Coord = board.pawns
        k = 0
        move = 0
        isSym = False
        if p1Coord != [8-p2Coord[0], p2Coord[1]]:
            k+=1
            move = ('P',8-p1Coord[0], p1Coord[1])
        for wall in board.horiz_walls:
            if (7-wall[0], wall[1]) not in board.horiz_walls:
                k+=1
                move = ('WH', 7-wall[0], wall[1])
        for wall in board.verti_walls:
            if (7-wall[0], wall[1]) not in board.verti_walls:
                k+=1
                move = ('WV', 7-wall[0], wall[1])
        
        if k==1 and board.is_action_valid(move, 1):
            isSym = True
        
        return isSym, move

    def chooseDepth(nbActions, time):
        """
        This function decides what depth we can afford for the alphaBeta algorithm without compromising computation time.
        :param nbActions: state of the game
        :param time: time left ingame for the player
        :return: depth to use for AlphaBeta pruning algorithm
        """        
        if time < 60 or nbActions >= 8:
            return 1
        return 2

    def manhattanDistance(board, player):
        """
        This function calculates the Manhattan distance from the finish line.
        It gives an idea of the player's position in the board
        :param board: state of the game
        :param player: the player whose turn it is to play
        :return: Distance from the finish line
        """                
        return board.goals[0] - board.pawns[player][0]

if __name__ == "__main__":
    agent_main(MyAgent())