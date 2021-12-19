#from chessPieces import Rook
import chessPieces
import moves

class Chess():

    def __init__(self):

        #chessboard
        self.chessboard = [["    __    "] * 6 for i in range (6)]
        
        #pawns
        for i in range(6):
            self.chessboard[1][i] = 'black_pawn'
            self.chessboard[4][i] = 'white_pawn'
        #self.chessboard[1][2] = "    __    "
        #rooks
        self.chessboard[0][0] = 'black_rook'
        self.chessboard[0][1] = 'black_rook'
        self.chessboard[0][4] = 'black_rook'
        self.chessboard[0][5] = 'black_rook'
        
        self.chessboard[5][0] = 'white_rook'
        self.chessboard[5][1] = 'white_rook'
        self.chessboard[5][4] = 'white_rook'
        self.chessboard[5][5] = 'white_rook'

        #queen
        self.chessboard[0][2] = 'blackqueen'
        self.chessboard[5][2] = 'whitequeen'

        #king
        self.chessboard[0][3] = 'black-king'
        self.chessboard[5][3] = 'white-king'
        
        self.current_player = "b"
        self.winner = "none"

    def is_game_over(game):

        is_game_over = False
        count = 0
        kings = []
        for i in range (6):
            for j in range(6):
                if "king" in game.chessboard[i][j]:
                    count += 1
                    kings.append(game.chessboard[i][j])

        if count == 1:
            is_game_over = True
            if "white" in  kings[0]:
                game.winner = "White"

            else:
                game.winner = "Black"


        return is_game_over  

    def is_tied(chessboard):
        return False

    def winner(chessboard):
        winner = ""

        return winner


    def displayChessboard(self):
        #print(self.chessboard)
        
        print("|___________________________________________________________________________|")
        print("|                                                                           |")
        for i in range(6):
            for j in range(6):
                print("|"+self.chessboard[i][j]+"|", end=' ')
            print(end='\n')
            print("|___________________________________________________________________________|")
        print("|                                                                           |")