from numpy import select
import chessboard
import chessPieces
import moves
import MiniMax


def get_user_action():
    
    current = input("Enter which piece you want to move ")
    c = tuple(map(int, current.split(',')))
    possibleMoves = moves.Rules.possible_moves_of_piece(c,game.chessboard)
    print("Possible Moves:")
    print(possibleMoves)
    humanpossibleSwape = MiniMax.Minimax.humanpossibleSwape(game.chessboard,c)
    print("Possible Swapes:")
    print(humanpossibleSwape)
    new = input("Enter target position")
    
    n = tuple(map(int, new.split(',')))
    return c,n


if __name__ == '__main__':
    tieCount = 0
    game = chessboard.Chess()

    print("Select mode of each player:")
    print("1. Player 1 - Human Controlled   Player 2 - Human Controlled")
    print("2. Player 1 - AI Controlled   Player 2 - Human Controlled")
    print("3. Player 1 - AI Controlled   Player 2 - AI Controlled")

    mode = int(input("Enter your choice"))

    player = "w"
    AI_player = "b"
    #print(MiniMax.Minimax.possibleStates(game.chessboard, player))
    state = game.chessboard,player

    print("Starting game. Here is the board")
    game.displayChessboard() 

    if(mode == 1):
        player1 = "black"
        player2 = "white"
        while not chessboard.Chess.is_game_over(game):
            print("Player 1's Turn")
            current,new = get_user_action()

            if new in moves.Rules.possible_moves_of_player(current,game.chessboard,player1):
                game.chessboard = moves.Rules.makeMove(game.chessboard, current, new)
                game.displayChessboard()
            else:
            
                while new not in moves.Rules.possible_moves_of_player(current,game.chessboard,player1):
                    print("invalid move") 
                    current,new = get_user_action()

                game.chessboard = moves.Rules.makeMove(game.chessboard, current, new)
                game.displayChessboard()
                

            print("Player 2's Turn")
            current,new = get_user_action()

            if new in moves.Rules.possible_moves_of_player(current,game.chessboard,player2):
                game.chessboard = moves.Rules.makeMove(game.chessboard, current, new)
                game.displayChessboard()
            
            else:
                while new not in moves.Rules.possible_moves_of_player(current,game.chessboard,player2):
                    print("invalid move") 
                    current,new = get_user_action()

                game.chessboard = moves.Rules.makeMove(game.chessboard, current, new)
                game.displayChessboard()    
                
                    



    elif(mode == 2):
        while not chessboard.Chess.is_game_over(game):
 
            print("AI's Turn")
            root = MiniMax.Minimax.create_tree(game, game.chessboard, player)
            game.chessboard = MiniMax.Minimax.minimax(root, 0, True, -99999,99999,"black").data
            game.displayChessboard()  

            if chessboard.Chess.is_game_over(game):
                break 
 
            print("Player's Turn")
            current,new = get_user_action()
            
            if new in moves.Rules.possible_moves_of_piece(current,game.chessboard):
                game.chessboard = moves.Rules.makeMove(game.chessboard, current, new)
                game.displayChessboard()
            else:

                while new not in moves.Rules.possible_moves_of_piece(current,game.chessboard):
                    print("invalid move") 
                    current,new = get_user_action()

                game.chessboard = moves.Rules.makeMove(game.chessboard, current, new)
                game.displayChessboard()    



    elif(mode == 3):
        while not chessboard.Chess.is_game_over(game):
            player1 = "w"
            player2 = "b"
            print("AI 1's Turn")
            root = MiniMax.Minimax.create_tree(game, game.chessboard, player1)
            game.chessboard = MiniMax.Minimax.minimax(root, 0, True, -99999,99999,"black").data
            game.displayChessboard()  
            
            if chessboard.Chess.is_game_over(game):
                break
            
            print("AI 2's Turn")   
            root = MiniMax.Minimax.create_tree(game, game.chessboard, player2)
            game.chessboard = MiniMax.Minimax.minimax(root, 0, True, -99999,99999,"white").data
            game.displayChessboard() 
            
    
    if game.winner != "none":
        print("Winner of the game is ",game.winner)    
