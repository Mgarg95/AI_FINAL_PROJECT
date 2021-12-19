import re
#Pawn
class Pawn():
    '''
    After its first move the Pawn may only advance one square at a time. 
    The Pawn captures by moving diagonally one square forward in each direction.
    '''

    def possible_moves(board,coordinates):
        possible_moves = []
        x = coordinates[0]
        y = coordinates[1]
        
        if(re.match("black+",board[x][y])):
            # move downwards
            #x+1,y --> 
            if(x+1 < 6 and board[x+1][y] == "    __    "):
                possible_moves.append((x+1,y))

            #diagonal
            if(x+1 < 6 and y+1 < 6 and re.match("white+",board[x+1][y+1])):
                possible_moves.append((x+1,y+1))
            if(x+1 < 6 and y-1 >= 0 and re.match("white+",board[x+1][y-1])): 
                possible_moves.append((x+1,y-1))  

        if(re.match("white+",board[x][y])):
            #move upwards
            if(x-1 >= 0 and board[x-1][y] == "    __    "):
                possible_moves.append((x-1,y))

            #diagonal    
            if(x-1 >= 0 and y+1 < 6 and re.match("black+",board[x-1][y+1])):
                possible_moves.append((x-1,y+1))
            if(x-1 >= 0 and y-1 >= 0 and re.match("black+",board[x-1][y-1])):
                possible_moves.append((x-1,y-1))


        return possible_moves


#Rook
class Rook():
    '''
    Rook moves in a straight line any number of squares, horizontally or vertically.
    '''
     #coordenation is a 1d array 
    #function possibleMoves will return a list contains 2 int array: possibleMoves[0] which will return the first possible move for the rook
    def possible_moves(board,coordenation):
        if(re.match("black+",board[coordenation[0]][coordenation[1]])):
            pattern = "white+"
            color = "black+"
        else:
             pattern = "black+"
             color= "white+"
        possibleMoves=[[],[]]

        for i in range(coordenation[0],6):#move down
           
            if(i+1 < 6):
                if(re.match(color,board[i+1][coordenation[1]])):
                    break
                possibleMoves[0].append(i+1)
                possibleMoves[1].append(coordenation[1])
                if(re.match(pattern,board[i+1][coordenation[1]])):
                    break

        for i in range(coordenation[0],0,-1):#move up
            if(i-1 >=0):
                if(re.match(color,board[i-1][coordenation[1]])):
                    break
                possibleMoves[0].append(i-1)
                possibleMoves[1].append(coordenation[1])
                if(re.match(pattern,board[i-1][coordenation[1]])):
                    break

        for i in range(coordenation[1],0,-1):#move left
            if(i-1 >= 0):
                if(re.match(color,board[coordenation[0]][i-1])):
                   break
                possibleMoves[0].append(coordenation[0])
                possibleMoves[1].append(i-1)
                if(re.match(pattern,board[coordenation[0]][i-1])):
                   break

        for i in range(coordenation[1],6):#move right
            if(i+1 < 6):
                if(re.match(color,board[coordenation[0]][i+1])):
                  break
                possibleMoves[0].append(coordenation[0])
                possibleMoves[1].append(i+1)
                if(re.match(pattern,board[coordenation[0]][i+1])):
                  break
        return list(zip(possibleMoves[0],possibleMoves[1]))

#King
class King():
    '''
    King moves 1 square in any direction, till no piece is blocking its path.
    '''
    def possible_moves(board,coordenation):
        if(re.match("black+",board[coordenation[0]][coordenation[1]])):
            pattern = "white+"
            color = "black+"
        else:
             pattern = "black+"
             color= "white+"
        possibleMoves=[[],[]]

        for i in range(coordenation[0],6):#move down
           
            if(i+1 < 6):
                if(re.match(color,board[i+1][coordenation[1]])):
                    break
                possibleMoves[0].append(i+1)
                possibleMoves[1].append(coordenation[1])
                break

        for i in range(coordenation[0],0,-1):#move up
            if(i-1 >=0):
                if(re.match(color,board[i-1][coordenation[1]])):
                    break
                possibleMoves[0].append(i-1)
                possibleMoves[1].append(coordenation[1])
                break

        for i in range(coordenation[1],0,-1):#move left
            if(i-1 >= 0):
                if(re.match(color,board[coordenation[0]][i-1])):
                   break
                possibleMoves[0].append(coordenation[0])
                possibleMoves[1].append(i-1)
                break

        for i in range(coordenation[1],6):#move right
            if(i+1 < 6):
                if(re.match(color,board[coordenation[0]][i+1])):
                  break
                possibleMoves[0].append(coordenation[0])
                possibleMoves[1].append(i+1)
                break

        if(coordenation[0] -1 >=0 and coordenation[1] -1 >=0 and not(re.match(color,board[coordenation[0]-1][coordenation[1] -1])) ):
             possibleMoves[0].append(coordenation[0] -1)
             possibleMoves[1].append(coordenation[1] -1)
        if(coordenation[0] -1 >=0 and coordenation[1] +1 < 6 and not(re.match(color,board[coordenation[0]-1][coordenation[1] +1]))):
             possibleMoves[0].append(coordenation[0] -1)
             possibleMoves[1].append(coordenation[1] +1)
        if(coordenation[0] +1 <6 and coordenation[1] - 1 >= 0 and not(re.match(color,board[coordenation[0]+1][coordenation[1] -1]))):
             possibleMoves[0].append(coordenation[0] +1)
             possibleMoves[1].append(coordenation[1] -1)
        if(coordenation[0] +1 <6 and coordenation[1] + 1 < 6 and not(re.match(color,board[coordenation[0]+ 1][coordenation[1] +1]))):
             possibleMoves[0].append(coordenation[0] +1)
             possibleMoves[1].append(coordenation[1] +1)
        return list(zip(possibleMoves[0],possibleMoves[1]))


#Queen
class Queen():
    
    # Queen moves any number of squares straight or diagonally in any direction.
    
    def possible_moves(board,coordenation):
        if(re.match("black+",board[coordenation[0]][coordenation[1]])):
            pattern = "white+"
            color = "black+"
        else:
             pattern = "black+"
             color= "white+"
        possibleMoves=[[],[]]
        for i in range(coordenation[0],6):#move down
           
            if(i+1 < 6):
                if(re.match(color,board[i+1][coordenation[1]])):
                    break
                possibleMoves[0].append(i+1)
                possibleMoves[1].append(coordenation[1])
                if(re.match(pattern,board[i+1][coordenation[1]])):
                    break

        for i in range(coordenation[0],0,-1):#move up
            if(i-1 >=0):
                if(re.match(color,board[i-1][coordenation[1]])):
                    break
                possibleMoves[0].append(i-1)
                possibleMoves[1].append(coordenation[1])
                if(re.match(pattern,board[i-1][coordenation[1]])):
                    break

        for i in range(coordenation[1],0,-1):#move left
            if(i-1 >= 0):
                if(re.match(color,board[coordenation[0]][i-1])):
                   break
                possibleMoves[0].append(coordenation[0])
                possibleMoves[1].append(i-1)
                if(re.match(pattern,board[coordenation[1]][i-1])):
                   break

        for i in range(coordenation[1],6):#move right
            if(i+1 < 6):
                if(re.match(color,board[coordenation[0]][i+1])):
                  break
                possibleMoves[0].append(coordenation[0])
                possibleMoves[1].append(i+1)
                if(re.match(pattern,board[coordenation[1]][i+1])):
                  break
        x = coordenation[0]
        y = coordenation[1]
        while(x-1 >= 0 and y- 1 >= 0): #move upleft
            if(re.match(color,board[x-1][y-1])):
                  break
            possibleMoves[0].append(x-1)
            possibleMoves[1].append(y-1)
            x = x-1
            y = y-1
            if(re.match(pattern,board[x-1][y-1])):
                  break
        x = coordenation[0]
        y = coordenation[1]

        while(x-1 >= 0 and y+ 1 < 6): #move upright
            if(re.match(color,board[x-1][y+1])):
                  break
            possibleMoves[0].append(x-1)
            possibleMoves[1].append(y+1)
            if(re.match(pattern,board[x-1][y+1])):
                  break
            x = x-1
            y = y+1

        x = coordenation[0]
        y = coordenation[1]
        while(x+1 < 6 and y - 1 >= 0): #move downleft
            if(re.match(color,board[x+1][y-1])):
                  break
            possibleMoves[0].append(x+1)
            possibleMoves[1].append(y-1)
            if(re.match(pattern,board[x+1][y-1])):
                  break
            x = x+1
            y = y-1

        x = coordenation[0]
        y = coordenation[1]
        while(x+1 < 6 and y + 1 < 6 ): #move downright
            if(re.match(color,board[x+1][y+1])):
                  break
            possibleMoves[0].append(x+1)
            possibleMoves[1].append(y+1)
            if(re.match(pattern,board[x+1][y+1])):
                  break
            x = x+1
            y = y+1

        return list(zip(possibleMoves[0],possibleMoves[1])) 

            
    
