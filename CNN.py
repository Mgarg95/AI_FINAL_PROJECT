import numpy as np
import torch as tr
import MiniMax
import chessboard
import random
import matplotlib.pyplot as pt

def encode(state):
    symbols = np.array(["    __    ","black_pawn","black_pawn","white_pawn","black_rook","white_rook","blackqueen","whitequeen","black-king","white-king"]).reshape(-1,1,1)
    onehot = (symbols == state).astype(np.float32)
    return tr.tensor(onehot)


def get_random_state(depth,player):
    current_board = chessboard.Chess(6).chessboard

    for n in range(depth):
        board = MiniMax.Baseline_AI.baseline_AI(current_board, player)
        current_board = board
        if player == "b":
            player= "w"
        else:
            player = "b"    
    #MiniMax.Minimax.displayChessboard(board)        
    return board   


def generate_training_set_1(num_examples, depth, size):
    training_examples = []
    players = ["b","w"]
    current_board = chessboard.Chess(6).chessboard
    for i in range(num_examples):
        player = random.choice(players)
        board = get_random_state(depth,player)
        if player == "b":
            player_color = "black"
        else:
            player_color = "white"    
        utilitity = MiniMax.Minimax.evaluation(board,player_color)
        training_examples.append((current_board,utilitity))
        current_board = board

    return training_examples    


class ConvNet(tr.nn.Module):
    def __init__(self, size, hid_features):
        super(ConvNet, self).__init__()
        self.to_hidden = tr.nn.Conv2d(10, hid_features, 2)
        self.to_output = tr.nn.Linear(hid_features*(size-1)**2, 1)
    def forward(self, x):
        h = tr.relu(self.to_hidden(x))
        y = tr.tanh(self.to_output(h.reshape(x.shape[0],-1)))
        return y        


def batch_error(net,batch):
    states, utilities = batch
    u = utilities.reshape(-1,1).float()
    y = net(states)
    e = tr.sum((y - u)**2) / utilities.shape[0]
    return e


def nn_evaluation(state):
    with tr.no_grad():
        utility = net(encode(state).unsqueeze(0))
    return utility


def dl_minimax(node, depth,isMaxPlayer, alpha, beta, AIColor):
    if(len(node.children) == 0): 
        node.score = nn_evaluation(node.data)
        return node
    if(isMaxPlayer):
        bestValue = -99999
        for i in node.children:
            value = dl_minimax(i, depth+1,False, alpha, beta ,AIColor).score
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
            value = dl_minimax(i, depth+1,True, alpha, beta ,AIColor).score
            bestValue = min(value, bestValue)
            beta = min(beta, bestValue)
            i.score = bestValue
            if(bestValue == value):
                chosenNode = i
            if(beta < alpha):
                break
        return chosenNode
    

def experiment_nn(size):
    game = chessboard.Chess(size)
    black_player_scores = []
    white_player_scores = []
    node_counts = []
    for i in range(100):
        count = 0
        game = chessboard.Chess(size)
        print("Game ",i)
        while not chessboard.Chess.is_game_over(game):
            player1 = "w"
            player2 = "b"
            print("AI 1's Turn")
            game.chessboard = MiniMax.Baseline_AI.baseline_AI(game.chessboard,"b")
            game.displayChessboard()  

            if chessboard.Chess.is_game_over(game):
                black_player_score,white_player_score = chessboard.Chess.get_score(game.chessboard)
                black_player_scores.append(black_player_score)
                white_player_scores.append(white_player_score)
                print(black_player_score)
                print(white_player_score)
                break

            print("AI 2's Turn")   
            root,node_count = MiniMax.Minimax.create_tree_with_node_count(game, game.chessboard, player2)
            game.chessboard = dl_minimax(root, 2, True, -99999,99999,"white").data
            count += node_count 
            game.displayChessboard()
            

            if chessboard.Chess.is_game_over(game):
                black_player_score,white_player_score = chessboard.Chess.get_score(game.chessboard)
                black_player_scores.append(black_player_score)
                white_player_scores.append(white_player_score)
                print(black_player_score)
                print(white_player_score)
        node_counts.append(count)        

    print(white_player_scores)
    print(black_player_scores)
    print(node_counts)

    pt.hist(white_player_scores,bins=3,rwidth=0.95)
    pt.show()

    pt.hist(black_player_scores,bins=3,rwidth=0.95)
    pt.show()

    pt.hist(node_counts,bins=3,rwidth=0.95)
    pt.show()

print("Generating training examples")
training_examples_1 = generate_training_set_1(num_examples = 500, depth=3, size=6)
print("Generating testing examples")
testing_examples_1 = generate_training_set_1(num_examples = 500, depth=3, size=6)


#if __name__=="main":

batched = True

net = ConvNet(size=6, hid_features=4)
optimizer = tr.optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)

states, utilities = zip(*training_examples_1)
training_batch = tr.stack(tuple(map(encode, states))), tr.tensor(utilities)

states, utilities = zip(*testing_examples_1)
testing_batch = tr.stack(tuple(map(encode, states))), tr.tensor(utilities)

_, utilities = zip(*training_examples_1)
print(utilities)
baseline_error =sum((u-0)**2 for u in utilities) / len(utilities)
print("Baseline error ",baseline_error)


curves = [], []
for epoch in range(50000):
    optimizer.zero_grad()
    e = batch_error(net, training_batch)
    e.backward()
    training_error = e.item()
    with tr.no_grad():
        e = batch_error(net, testing_batch)
        testing_error = e.item()
    optimizer.step()    

    if epoch % 1000 == 0:
        print("%d: %f, %f" % (epoch, training_error, testing_error))
    curves[0].append(training_error)
    curves[1].append(testing_error)

pt.plot(curves[0], 'b-')
pt.plot(curves[1], 'r-')
pt.plot([0, len(curves[1])], [baseline_error, baseline_error], 'g-')
pt.plot()
pt.legend(["Train","Test","Baseline"])
pt.show()    


experiment_nn(6)

