import MiniMax as MM
import chessboard

if __name__ == '__main__':
    board = chessboard.Chess()
    board.displayChessboard()
    root = MM.Minimax().create_tree(board.chessboard,"w")
    result = MM.minimax(root, 0, True, -99999,99999,"black")
    print(0)