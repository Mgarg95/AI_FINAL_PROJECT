# CIS667_Final_Project_Chess_Manha_Garg

The pseudocode for implementing minimax has been referred from GeeksforGeeks. The link is - https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-4-alpha-beta-pruning/

Following libraries have been used in the project and should be insatlled before running-

numpy

matplotlib

random

torch

copy

re

## Following are the steps to run interactive domain program-

1. Run PlayChess.py
2. You will be prompted to choose the board size-
  
  Select from the following board sizes
  1. 4x4
  2. 5x5
  3. 6x6
  4. 7x7
  5. 8x8
  
    Choose the desired option number. For eg, if you want to play with 6x6 board, enter 2 as your choice
3. Then you will be prompted to choose the mode for each player - 

  Select mode of each player:
  1. Player 1 - Human Controlled   Player 2 - Human Controlled
  2. Player 1 - AI Controlled   Player 2 - Human Controlled
  3. Player 1 - AI Controlled   Player 2 - AI Controlled

    Choose the desired option number, as explained earlier.
  
4. In case of human controlled player, Player will be prompted as follows-
   #### Enter which piece you want to move
   Provide the input as coordinates on the board with format r,c. All indexes are zero based. For eg, you wan to move piece at (4,0), give input as 4,0
5. The program shows you list of possible move and swaps, and asks for target position, like as follows - 
  Enter which piece you want to move 4,0

  Possible Moves:

  [(3, 0), [(5, 0), (5, 1), (5, 3), (5, 4), (5, 5)]]

  Possible Swapes:

  [(5, 0), (5, 1), (5, 3), (5, 4), (5, 5)]

  Enter target position

6. Choose target position from given options and enter in the same format as before, For eg - 5,1
7. The game will go on till there is a winner.
8. Choose option 3 to play with a tree based AI.
9. Choose option 3 to make baseline AI and Tree bAsed AI to play with each other.
10. Final score and winner is printed in the end, like - 

  Score of Player 1 -  240
  
  Score of Player 2 -  1390
  
  Winner of the game is  White
  
## Following are the steps to conduct experiments-

#### To conduct experiments between Baseline AI and Tree Based AI, run experiments.py
1. This program will prompt the user with following options
    Select from the following board sizes for experiment
     1. 4x4
     2. 5x5
     3. 6x6
     4. 7x7
     5. 8x8
2. Choose the desired option number, and experiment of 100 games on thte respective board size will be executed. 
3. Three graphs are plotted in the end, one after the other in the order of - White Player final scores, Black player final scores, Node Counts
#### To conduct experiments Baseline AI and Tree + NN AI, run CNN.py
1. This program will train and test the CNN and plot a learning curve.
2. The default depth can be changed to 1 or 2 while conducting experiments.
3. Then the experiment with 6x6 board size start running.
4. Three graphs are plotted in the end, one after the other in the order of - White Player final scores, Black player final scores, Node Counts
