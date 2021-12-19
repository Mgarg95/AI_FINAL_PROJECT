import re
import chessboard
import moves
import numpy as np
import copy
import random
import torch as tr
import matplotlib.pyplot as pt

class Node():
    def __init__(self, data):
        self.data = data
        self.children = []
        self.score =  -10000
        self.count = 0

    def add_child(self, obj):
        self.children.append(obj)

class Minimax():
    count = 0
    def evaluation (board, playerColor):
        boardSize = np.shape(board)
        score = 0
        pawn = re.compile("pawn")
        rook = re.compile("rook")
        queen = re.compile("queen")
        king = re.compile("king")
        if(playerColor == "black"):
            pattern = "white+"
            color = "black+"
        else:
             pattern = "black+"
             color= "white+"
        for x in range(0,boardSize[0]):
            for y in range(0,boardSize[1]):
                if(re.match(color,board[x][y])):
                    if(pawn.search(board[x][y])):
                        score += 30
                    if(rook.search(board[x][y])):
                        score += 60
                    if(queen.search(board[x][y])):
                        score += 80
                    if(king.search(board[x][y])):
                        score += 1000
                if(re.match(pattern,board[x][y])):
                    if(pawn.search(board[x][y])):
                        score -= 30
                    if(rook.search(board[x][y])):
                        score -= 60
                    if(queen.search(board[x][y])):
                        score -= 80
                    if(king.search(board[x][y])):
                        score -= 1000
        return score/100
    
    #call minimax(root, 0, True, -99999,99999,"black"). Credit:geekforgeeeks pseudocode
    def minimax(node, depth,isMaxPlayer, alpha, beta, AIColor):
        if(len(node.children) == 0): 
            node.score = Minimax.evaluation(node.data, AIColor)
            return node
        if(isMaxPlayer):
            bestValue = -99999
            for i in node.children:
                value = Minimax.minimax(i, depth+1,False, alpha, beta ,AIColor).score
                bestValue = max(value, bestValue)
                alpha = max(alpha, bestValue)
                i.score = bestValue
                if(bestValue == value):
                   chosenNode = i
                if(beta < alpha):
                    break
            return chosenNode
        else:
            bestValue = 99999
            for i in node.children:
                value = Minimax.minimax(i, depth+1,True, alpha, beta ,AIColor).score
                bestValue = min(value, bestValue)
                beta = min(beta, bestValue)
                i.score = bestValue
                if(bestValue == value):
                    chosenNode = i
                if(beta < alpha):
                    break
            return chosenNode
    
    def humanpossibleSwape(board,current):
        possible_states = []
        boardCopy = copy.deepcopy(board)
        pattern = "white+"
        if(re.search(pattern,boardCopy[current[0]][current[1]]) ):
            color= "white+"
        else:
            color = "black+"
 
        boardSize = np.shape(boardCopy)
        #print("board size",boardSize)
        pieceA = boardCopy[current[0]][current[1]]
        for i in range(boardSize[0]):
            for j in range(boardSize[1]):
                if((i != current[0] or j != current[1]) and re.match(color,boardCopy[i][j]) and (boardCopy[i][j] != boardCopy[current[0]][current[1]]) and (boardCopy[i][j] != "black-king" and boardCopy[current[0]][current[1]] != "black-king") and (boardCopy[i][j] != "white-king" and boardCopy[current[1]][current[1]] != "white-king")):
                    possible_states.append((i,j))
        return possible_states

    def possibleSwape(board, player):
        possible_states = []
        boardCopy = copy.deepcopy(board)
        if(player == "b"):
            pattern = "white+"
            color = "black+"
        else:
             pattern = "black+"
             color= "white+"
 
        boardSize = np.shape(boardCopy)
        #print("board size",boardSize)
        for x in range(boardSize[0]):
            for y in range(boardSize[1]):
                pieceA = boardCopy[x][y]
                if(not re.match(color,pieceA)):
                    break
                for i in range(x,boardSize[0]):
                    for j in range(boardSize[1]):
                        if(not (i <= x and j <= y)):
                            if(re.match(color,boardCopy[i][j]) and (boardCopy[i][j] != boardCopy[x][y]) and (boardCopy[i][j] != "black-king" and boardCopy[x][y] != "black-king") and (boardCopy[i][j] != "white-king" and boardCopy[x][y] != "white-king")):
                                boardCopy2 = copy.deepcopy(board)
                                boardCopy2[i][j], boardCopy2[x][y] = boardCopy2[x][y], boardCopy2[i][j]
                                if(board != boardCopy2):
                                    possible_states.append(boardCopy2)
        return possible_states

    def possibleStates(board, player): 
        possible_states = []
        
        possible_moves = moves.Rules.moves(board, player)

        for current_position in possible_moves:
            new_positions = possible_moves[current_position]

            for new_position in new_positions:
                possible_states.append(moves.Rules.makeMove(board,current_position,new_position))

        return possible_states    

    def create_tree(self,board, player, depth = 3):

        if(player == 'b'):
            AIplayer = 'w'
        else:
            AIplayer = 'b'
        root = Node(board)
        states1 = Minimax.possibleSwape(board, AIplayer)
        states1 += Minimax.possibleStates(board, AIplayer)
        AIplayer,player  = player,AIplayer
        for j in states1:
            child = Node(j)
            root.add_child(child)
            if((depth-1) != 0):
                node = Minimax.create_tree(self,j, player, depth-1)
                for k in node.children:
                    child.add_child(k)

        return root

    def create_tree_with_node_count(self,board, player, depth = 3):

        node_count = 0
        if(player == 'b'):
            AIplayer = 'w'
        else:
            AIplayer = 'b'
        root = Node(board)
        states1 = Minimax.possibleSwape(board, AIplayer)
        states1 += Minimax.possibleStates(board, AIplayer)
        AIplayer,player  = player,AIplayer
        for j in states1:
            child = Node(j)
            root.add_child(child)
            node_count += 1
            if((depth-1) != 0):
                node = Minimax.create_tree(self,j, player, depth-1)
                for k in node.children:
                    child.add_child(k)
                    node_count += 1

        return root,node_count


    def displayChessboard(chessboard):
        print("|___________________________________________________________________________|")
        print("|                                                                           |")
        for i in range(np.shape(chessboard)[0]):
            for j in range(np.shape(chessboard)[1]):
                print("|"+chessboard[i][j]+"|", end=' ')
            print(end='\n')
            print("|___________________________________________________________________________|")
        print("|                                                                           |")
    
class Baseline_AI():
    def baseline_AI(board,AIplayer):
        states = Minimax.possibleStates(board, AIplayer)
        states += Minimax.possibleSwape(board, AIplayer)
        updated_board = random.choice(states)

        return updated_board

def random_state(depth=0, size=3):
    board = chessboard.Chess()
    for d in range(depth):
        board = Baseline_AI.baseline_AI(board,'w')
        board = Baseline_AI.baseline_AI(board,'b')
    return board

def generate(num_examples, depth, size):
    examples = []
    for n in range(num_examples):
        state = random_state(depth, size)
        utility = Minimax.evaluation()
        examples.append((state, utility))
    return examples

def encode(state):
    symbols = np.array(["    __    ","black_pawn","black_rook","blackqueen","black-king","white_pawn","white_rook","whitequeen","white-king"]).reshape(-1,1,1)
    onehot = (symbols == state.board).astype(np.float32)
    return tr.tensor(onehot)
'''
class LinNet(tr.nn.Module):
    def __init__(self, size, hid_features):
        super(LinNet, self).__init__()
        self.to_hidden = tr.nn.Linear(3*size**2, hid_features)
        self.to_output = tr.nn.Linear(hid_features, 1)
    def forward(self, x):
        h = tr.relu(self.to_hidden(x.reshape(x.shape[0],-1)))
        y = tr.tanh(self.to_output(h))
        return y
'''
class ConvNet(tr.nn.Module):
    def __init__(self, size, hid_features):
        super(ConvNet, self).__init__()
        self.to_hidden = tr.nn.Conv2d(3, hid_features, 2)
        self.to_output = tr.nn.Linear(hid_features*(size-1)**2, 1)
    def forward(self, x):
        h = tr.relu(self.to_hidden(x))
        y = tr.tanh(self.to_output(h.reshape(x.shape[0],-1)))
        return y

'''
def cnn_viz(net):
    numrows = net.to_hidden.weight.shape[0] # out channels
    numcols = net.to_hidden.weight.shape[1] # in channels
    pt.figure(figsize=(numcols, numrows))

    sp = 0
    for r in range(numrows):
        for c in range(numcols):
            sp += 1
            pt.subplot(numrows, numcols, sp)
            pt.imshow(net.to_hidden.weight[r,c].detach().numpy(), cmap="gray")
    pt.show()
'''


# Calculates the error on one training example
def example_error(net, example):
    state, utility = example
    x = encode(state).unsqueeze(0)
    y = net(x)
    e = (y - utility)**2
    return e

# Calculates the error on a batch of training examples
def batch_error(net, batch):
    states, utilities = batch
    u = utilities.reshape(-1,1).float()
    y = net(states)
    e = tr.sum((y - u)**2) / utilities.shape[0]
    return e

def nn_eval(state):
    net = Net(size=4, hid_features=8)
    utility = net(encode(state).unsqueeze(0))

if __name__ == "__main__":
    training_examples = generate(num_examples = 100, depth=10, size=4)
    # whether to loop over individual training examples or batch them
    batched = True

    # Make the network and optimizer
    # net = LinNet(size=4, hid_features=4)
    # optimizer = tr.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net = ConvNet(size=4, hid_features=4)
    optimizer = tr.optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)

    # Convert the states and their minimax utilities to tensors
    states, utilities = zip(*training_examples)
    training_batch = tr.stack(tuple(map(encode, states))), tr.tensor(utilities)

    states, utilities = zip(*testing_examples)
    testing_batch = tr.stack(tuple(map(encode, states))), tr.tensor(utilities)

    # Run the gradient descent iterations
    curves = [], []
    for epoch in range(50000):
    
        # zero out the gradients for the next backward pass
        optimizer.zero_grad()

        # batch version (fast)
        if batched:
            e = batch_error(net, training_batch)
            e.backward()
            training_error = e.item()

            with tr.no_grad():
                e = batch_error(net, testing_batch)
                testing_error = e.item()

        # take the next optimization step
        optimizer.step()    
        
        # print/save training progress
        if epoch % 1000 == 0:
            print("%d: %f, %f" % (epoch, training_error, testing_error))
        curves[0].append(training_error)
        curves[1].append(testing_error)