import sys
sys.path.append("/kaggle_simulations/agent")
import os

import random
import numpy as np

from connect4.utils import *
from connect4.MCTS import MCTS

from connect4.Connect4Game import Connect4Game
from connect4.NNet import NNetWrapper as NNet

g = Connect4Game()
n1 = NNet(g)
n1.load_checkpoint('best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)

def submit_agent(obs, config):
    game_board = g.getInitBoard()
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    for i in range(config.rows):
        for j in range(config.columns):
            if grid[i][j] == 1:
                game_board[i][j] = 1
            elif grid[i][j] == 2:
                game_board[i][j] = -1
            else:
                game_board[i][j] = 0
    if obs.mark == 2:
        game_board = g.getCanonicalForm(game_board, -1)
    action = np.argmax(mcts1.getActionProb(game_board, temp=0))
    return action.item()