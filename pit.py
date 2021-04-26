import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
# from othello.OthelloPlayers import *
# from othello.pytorch.NNet import NNetWrapper as NNet

from connect4.Connect4Game import Connect4Game
from connect4.Connect4Players import *
from connect4.tensorflow.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""


def run_pit():
    mini_othello = False  # Play in 6x6 instead of the normal 8x8.
    human_vs_cpu = False

    if mini_othello:
        g = OthelloGame()
        #g = Connect4Game()
    else:
        #g = OthelloGame(8)
        g = Connect4Game()

    # all players
    rp = RandomPlayer(g).play
    # gp = GreedyOthelloPlayer(g).play
    # hp = HumanOthelloPlayer(g).play
    gp = OneStepLookaheadConnect4Player(g).play
    hp = HumanConnect4Player(g).play

    # nnet players
    n1 = NNet(g)
    if mini_othello:
        n1.load_checkpoint('./pretrained_models/othello/pytorch/', '6x100x25_best.pth.tar')
    else:
        #n1.load_checkpoint('./pretrained_models/othello/pytorch/','8x8_100checkpoints_best.pth.tar')
        n1.load_checkpoint('./server_models/run2googleInstanceConnect4/versions/', 'checkpoint_44.pth.tar')
    args1 = dotdict({'numMCTSSims': 150, 'cpuct': 1.0})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    if human_vs_cpu:
        player2 = hp
    else:
        n2 = NNet(g)
        n2.load_checkpoint('./server_models/run1googleInstanceConnect4/versions/', 'checkpoint_62.pth.tar')
        args2 = dotdict({'numMCTSSims': 150, 'cpuct': 1.0})
        mcts2 = MCTS(g, n2, args2)
        n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

        player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

    arena = Arena.Arena(n1p, player2, g, display=Connect4Game.display)

    print(arena.playGames(10, verbose=True))


def pit_versions(version_1_name, version_2_name):
    game = Connect4Game()
    args = dotdict({'numMCTSSims': 100, 'cpuct': 1.0})

    net1 = NNet(game)
    net1.load_checkpoint('./server_models/run1googleInstance/versions/', version_1_name)
    mcts1 = MCTS(game, net1, args)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    net2 = NNet(game)
    net2.load_checkpoint('./server_models/run1googleInstance/versions/', version_2_name)
    mcts2 = MCTS(game, net2, args)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    arena = Arena.Arena(n1p, n2p, game, display=Connect4Game.display)
    return arena.playGames(2, verbose=True)

run_pit()