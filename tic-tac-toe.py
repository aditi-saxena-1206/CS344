# CS386 - Artificial Intelligence Lab
# Lab Assignment 2 - Tic-Tac-Toe Game using Minimax algorithm
# Date of Assignemnt : October 1, 2020
# Name : Aditi Saxena
# Roll no. : 180010002
#==========================================

#function to return a value from the utility_value dictionary depending on 
#which player wins or game still continues or tie occurs

def state_game(board):
    utility_value = {'AI': 1,'Human':-1,'Cont':10,'Tie':0}

    #check for three consecutive Xs or Os in row, column or diagonal
    for row in range(3):
        if(board[row][0] == board[row][1] and board[row][1] == board[row][2] and board[row][0] != '-'):
            if(board[row][0] == 'O'):
                return utility_value['AI']
            else:
                return utility_value['Human']
    for column in range(3):
        if(board[0][column] == board[1][column] and board[1][column] == board[2][column] and board[0][column] != '-'):
            if(board[0][column] == 'O'):
                return utility_value['AI']
            else:
                return utility_value['Human']
    if(board[0][0] == board[1][1] and board[1][1] == board[2][2] and board[0][0] != '-'):
        if(board[0][0] == 'O'):
            return utility_value['AI']
        else:
            return utility_value['Human']
    if(board[0][2] == board[1][1] and board[1][1] == board[2][0] and board[0][2]!= '-'):
        if(board[0][2] == 'O'):
            return utility_value['AI']
        else:
            return utility_value['Human']

    #Checking if any more empty spaces left for the game to continue
    for i in range(3):
        for j in range(3):
            if(board[i][j] == '-'):
                return utility_value['Cont']
    # In case of a tie
    return utility_value['Tie']
#==============================================

#function to implement minimax algorithm in a recursive manner and returns
#the best score for a move by either player

def minimax_algo(board, agent_turn):
    score = state_game(board)

    #if an end node reached
    if(score != 10):
        return score
    #Else if the game continues, checking at an increased depth

    #if agent's turn, take maximizer
    if(agent_turn):
        max_score = -10
        for i in range(3):
            for j in range(3):
                if(board[i][j] == '-'):
                    board[i][j] = 'O'
                    max_score = max(max_score,minimax_algo(board, not agent_turn))
                    board[i][j] = '-'
        return max_score

    #if human's turn, take minimizer
    else:
        min_score = 10
        for i in range(3):
            for j in range(3):
                if(board[i][j] == '-'):
                    board[i][j] = 'X'
                    min_score = min(min_score, minimax_algo(board, not agent_turn))
                    board[i][j] = '-'
        return min_score
#==================================================

#function to determine the next best move for the AI agent by
#calling the minimax algorithm function

def next_move_AI(board):
    max_score = -10
    I = 0
    J = 0
    for i in range(3):
        for j in range(3):
            #place the 'O' at all empty spaces and return the position which has the 
            #highest score based on minimax algorithm
            if(board[i][j] == '-'):
                board[i][j] = 'O'
                score = minimax_algo(board,False)
                board[i][j] = '-'
                if(score > max_score):
                    max_score = score
                    I = i
                    J = j
    return (I,J)

#===================================================

#function to control the game play until either player wins or a tie occurs

def tic_tac_toe_game(board):
    #check if still moves left
    while(state_game(board) == 10):
        
        # Human's turn
        print('Enter space-separated row and column location for human player')
        l = input().split(' ')
        r = int(l[0])
        c = int(l[1])
        if(r<0 or r>2 or c<0 or c>2 or board[r][c] != '-'):
            print('Invalid input.Try again!')
            continue
        board[r][c] = 'X'
        print("After the human player move-")
        show_board_state(board)
        if(state_game(board) != 10):
            break
        
        #added for better visualization
        x = input("Press enter to continue...")

        # AI agent's move
        R, C = next_move_AI(board)
        #print(R," ",C)
        board[R][C] = 'O'
        print("After AI agent move-")
        show_board_state(board)
    
    #game has reached an end state
    #Either player has won or a tie has occured
    if(state_game(board) == 0):
        print('Tie!')
    elif(state_game(board) == -1):
        print('Human-Player wins!')
    else:
        print('AI-Player wins!')
#======================================================

#function to print the board at any instance

def show_board_state(board):
    print("Current state of the board")
    for i in range(3):
        for j in range(3):
            print(board[i][j],end='')
        print()

#=======================================================

#main function to begin execution

def main():
    board = [['-','-','-'],['-','-','-'],['-','-','-']]
    print("The initial state of the board")
    show_board_state(board)
    tic_tac_toe_game(board)

if __name__ == '__main__':
    main()
