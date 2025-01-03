import pygame
import sys
import numpy as np
import random
import time

# Initialize pygame
pygame.init()
pygame.mixer.init()

# Screen settings
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
LINE_WIDTH = 15
BACKGROUND_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (84, 84, 84)

# Sound effects
click_sound = pygame.mixer.Sound('click.wav')
win_sound = pygame.mixer.Sound('win.wav')
draw_sound = pygame.mixer.Sound('draw.wav')

# Sound functions
def play_click_sound():
    click_sound.play()

def play_win_sound():
    win_sound.play()

def play_draw_sound():
    draw_sound.play()


# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Tic Tac Toe')

# Create the board
board = np.zeros((3, 3))

# Game stats
stats = {"wins": 0, "losses": 0, "draws": 0}

# Draw the grid lines
def draw_grid():
    screen.fill(BACKGROUND_COLOR)
    pygame.draw.line(screen, LINE_COLOR, (200, 0), (200, 600), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (400, 0), (400, 600), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (0, 200), (600, 200), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (0, 400), (600, 400), LINE_WIDTH)

# Draw shapes on the board
def draw_shapes():
    for row in range(3):
        for col in range(3):
            if board[row][col] == 1:
                pygame.draw.circle(screen, CIRCLE_COLOR, (int(col * 200 + 100), int(row * 200 + 100)), 60, 15)
            elif board[row][col] == 2:
                pygame.draw.line(screen, CROSS_COLOR, (col * 200 + 55, row * 200 + 200 - 55),
                                 (col * 200 + 200 - 55, row * 200 + 55), LINE_WIDTH)
                pygame.draw.line(screen, CROSS_COLOR, (col * 200 + 55, row * 200 + 55),
                                 (col * 200 + 200 - 55, row * 200 + 200 - 55), LINE_WIDTH)

# Display game stats
def display_stats():
    font = pygame.font.Font(None, 36)
    text = f"Wins: {stats['wins']} | Losses: {stats['losses']} | Draws: {stats['draws']}"
    text_surface = font.render(text, True, (255, 255, 255))
    screen.blit(text_surface, (10, 10))

# Check for a winner
def check_winner():
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] and board[row][0] != 0:
            return board[row][0]
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != 0:
            return board[0][col]
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != 0:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != 0:
        return board[0][2]
    return None

# Reset the board
def reset_board():
    global board, player_turn
    board = np.zeros((3, 3))
    player_turn = True

# Minimax algorithm to calculate the best move for the AI
def minimax(b, depth, is_maximizing):
    winner = check_winner()
    if winner == 2:
        return 10 - depth
    elif winner == 1:
        return depth - 10
    elif np.all(b != 0):
        return 0

    if is_maximizing:
        best_score = float('-inf')
        for row in range(3):
            for col in range(3):
                if b[row][col] == 0:
                    b[row][col] = 2
                    score = minimax(b, depth + 1, False)
                    b[row][col] = 0
                    best_score = max(best_score, score)
        return best_score
    else:
        best_score = float('inf')
        for row in range(3):
            for col in range(3):
                if b[row][col] == 0:
                    b[row][col] = 1
                    score = minimax(b, depth + 1, True)
                    b[row][col] = 0
                    best_score = min(best_score, score)
        return best_score

# AI move based on difficulty
def ai_move(difficulty):
    if difficulty == 'easy':
        random_ai_move()
    elif difficulty == 'medium':
        limited_minimax_move()
    else:
        best_move()

# Random AI move for Easy difficulty
def random_ai_move():
    empty_cells = [(row, col) for row in range(3) for col in range(3) if board[row][col] == 0]
    if empty_cells:
        row, col = random.choice(empty_cells)
        board[row][col] = 2

# Limited-depth Minimax for Medium difficulty
def limited_minimax_move():
    best_score = float('-inf')
    move = None
    for row in range(3):
        for col in range(3):
            if board[row][col] == 0:
                board[row][col] = 2
                score = minimax(board, 2, False)  # Limit depth to 2
                board[row][col] = 0
                if score > best_score:
                    best_score = score
                    move = (row, col)
    if move:
        board[move[0]][move[1]] = 2

# Find the best move for Hard difficulty
def best_move():
    best_score = float('-inf')
    move = None
    for row in range(3):
        for col in range(3):
            if board[row][col] == 0:
                board[row][col] = 2
                score = minimax(board, 0, False)
                board[row][col] = 0
                if score > best_score:
                    best_score = score
                    move = (row, col)
    if move:
        board[move[0]][move[1]] = 2

# Main game loop
running = True
player_turn = True
difficulty = input("Choose difficulty: Easy, Medium, or Hard: ").lower()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if player_turn and event.type == pygame.MOUSEBUTTONDOWN:
            mouseX, mouseY = event.pos
            clicked_row, clicked_col = mouseY // 200, mouseX // 200

            if board[clicked_row][clicked_col] == 0:
                board[clicked_row][clicked_col] = 1
                play_click_sound()
                player_turn = False

    if not player_turn:
        ai_move(difficulty)
        player_turn = True

    winner = check_winner()
    if winner is not None:
        if winner == 1:
            stats["wins"] += 1
            play_win_sound()
        elif winner == 2:
            stats["losses"] += 1
            play_win_sound()
        reset_board()

    if np.all(board != 0):
        stats["draws"] += 1
        play_draw_sound()
        reset_board()

    draw_grid()
    draw_shapes()
    display_stats()
    pygame.display.update()
