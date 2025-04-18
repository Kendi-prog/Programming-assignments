import tkinter as tk
from tkinter import ttk
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import random
import threading

class TicTacToe:
    def __init__(self):
        # Initialize empty 3x3 board
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'  # X starts
        
    def reset(self):
        # Reset the board to empty state
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
    
    def available_moves(self):
        # Return list of available moves as (row, col) tuples
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    moves.append((i, j))
        return moves
    
    def make_move(self, row, col):
        # Make a move on the board
        if self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False
    
    def undo_move(self, row, col):
        # Undo a move (used in the Minimax algorithm)
        self.board[row][col] = ' '
        self.current_player = 'O' if self.current_player == 'X' else 'X'
    
    def is_winner(self, player):
        # Check if the specified player has won
        # Check rows
        for i in range(3):
            if all(self.board[i][j] == player for j in range(3)):
                return True
        
        # Check columns
        for j in range(3):
            if all(self.board[i][j] == player for i in range(3)):
                return True
        
        # Check diagonals
        if all(self.board[i][i] == player for i in range(3)):
            return True
        if all(self.board[i][2-i] == player for i in range(3)):
            return True
        
        return False
    
    def is_board_full(self):
        # Check if the board is full (draw)
        return all(self.board[i][j] != ' ' for i in range(3) for j in range(3))
    
    def game_over(self):
        # Check if the game is over (win or draw)
        return self.is_winner('X') or self.is_winner('O') or self.is_board_full()
    
    def get_result(self):
        # Return the result of the game: 'X' for X wins, 'O' for O wins, 'Draw' for draw
        if self.is_winner('X'):
            return 'X'
        elif self.is_winner('O'):
            return 'O'
        elif self.is_board_full():
            return 'Draw'
        return None

class MinimaxPlayer:
    def __init__(self, algorithm='minimax'):
        self.algorithm = algorithm  # 'minimax' or 'alphabeta'
        self.nodes_evaluated = 0
    
    def get_move(self, game):
        self.nodes_evaluated = 0
        start_time = time.time()
        
        if self.algorithm == 'minimax':
            best_score = float('-inf') if game.current_player == 'X' else float('inf')
            best_move = None
            
            for move in game.available_moves():
                row, col = move
                game.make_move(row, col)
                
                if game.current_player == 'O':  # X just played, now O's turn
                    score = self.minimax(game, False)  # False = O's turn (minimizing)
                else:  # O just played, now X's turn
                    score = self.minimax(game, True)  # True = X's turn (maximizing)
                
                game.undo_move(row, col)
                
                # Update best move
                if game.current_player == 'X':  # X is maximizing
                    if score > best_score:
                        best_score = score
                        best_move = move
                else:  # O is minimizing
                    if score < best_score:
                        best_score = score
                        best_move = move
        
        else:  # Alpha-Beta pruning
            best_move = self.alphabeta_decision(game)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return best_move, execution_time, self.nodes_evaluated
    
    def minimax(self, game, is_maximizing):
        self.nodes_evaluated += 1
        
        # Check if the game is over
        if game.is_winner('X'):
            return 1  # X wins
        elif game.is_winner('O'):
            return -1  # O wins
        elif game.is_board_full():
            return 0  # Draw
        
        # Continue the search
        if is_maximizing:  # X's turn
            best_score = float('-inf')
            for move in game.available_moves():
                row, col = move
                game.make_move(row, col)
                score = self.minimax(game, False)
                game.undo_move(row, col)
                best_score = max(score, best_score)
            return best_score
        
        else:  # O's turn (minimizing)
            best_score = float('inf')
            for move in game.available_moves():
                row, col = move
                game.make_move(row, col)
                score = self.minimax(game, True)
                game.undo_move(row, col)
                best_score = min(score, best_score)
            return best_score
    
    def alphabeta_decision(self, game):
        best_value = float('-inf')
        beta = float('inf')
        best_move = None
        alpha = float('-inf')
        
        for move in game.available_moves():
            row, col = move
            game.make_move(row, col)
            
            if game.current_player == 'O':  # X just played, now O's turn
                value = self.alphabeta_min_value(game, alpha, beta)
            else:  # O just played, now X's turn
                value = self.alphabeta_max_value(game, alpha, beta)
            
            game.undo_move(row, col)
            
            if value > best_value:
                best_value = value
                best_move = move
                alpha = max(alpha, best_value)
        
        return best_move
    
    def alphabeta_max_value(self, game, alpha, beta):
        self.nodes_evaluated += 1
        
        # Check if the game is over
        if game.is_winner('X'):
            return 1
        elif game.is_winner('O'):
            return -1
        elif game.is_board_full():
            return 0
        
        value = float('-inf')
        
        for move in game.available_moves():
            row, col = move
            game.make_move(row, col)
            value = max(value, self.alphabeta_min_value(game, alpha, beta))
            game.undo_move(row, col)
            
            if value >= beta:
                return value
            alpha = max(alpha, value)
        
        return value
    
    def alphabeta_min_value(self, game, alpha, beta):
        self.nodes_evaluated += 1
        
        # Check if the game is over
        if game.is_winner('X'):
            return 1
        elif game.is_winner('O'):
            return -1
        elif game.is_board_full():
            return 0
        
        value = float('inf')
        
        for move in game.available_moves():
            row, col = move
            game.make_move(row, col)
            value = min(value, self.alphabeta_max_value(game, alpha, beta))
            game.undo_move(row, col)
            
            if value <= alpha:
                return value
            beta = min(beta, value)
        
        return value

class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe: Minimax vs Alpha-Beta")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        
        # Create the game object
        self.game = TicTacToe()
        
        # Create players
        self.minimax_player = MinimaxPlayer(algorithm='minimax')
        self.alphabeta_player = MinimaxPlayer(algorithm='alphabeta')
        
        # Initialize performance metrics
        self.minimax_times = []
        self.alphabeta_times = []
        self.minimax_nodes = []
        self.alphabeta_nodes = []
        self.results = {"X": 0, "O": 0, "Draw": 0}
        
        # Create frames
        self.create_frames()
        
        # Create game board
        self.create_board()
        
        # Create controls
        self.create_controls()
        
        # Create stats display
        self.create_stats_display()
        
        # Game state
        self.is_game_running = False
        self.move_delay = 1000  # 1 second delay between moves
        self.current_algorithm = "minimax"  # Current algorithm being displayed
        
    def create_frames(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Game board frame (left)
        self.board_frame = ttk.LabelFrame(main_frame, text="Game Board", padding="10")
        self.board_frame.pack(side="left", fill="both", expand=True)
        
        # Control panel (right)
        self.control_frame = ttk.Frame(main_frame, padding="10")
        self.control_frame.pack(side="right", fill="y")
        
        # Stats frame (bottom right)
        self.stats_frame = ttk.LabelFrame(self.control_frame, text="Statistics", padding="10")
        self.stats_frame.pack(side="bottom", fill="x", pady=(10, 0))
        
    def create_board(self):
        # Create canvas for the game board
        self.canvas = tk.Canvas(self.board_frame, width=400, height=400, bg="white")
        self.canvas.pack(padx=10, pady=10)
        
        # Draw the grid
        self.draw_grid()
        
    def draw_grid(self):
        # Clear canvas
        self.canvas.delete("grid")
        
        # Draw grid lines
        for i in range(1, 3):
            # Vertical lines
            self.canvas.create_line(i * 133, 0, i * 133, 400, width=2, tags="grid")
            # Horizontal lines
            self.canvas.create_line(0, i * 133, 400, i * 133, width=2, tags="grid")
    
    def update_board(self):
        # Clear previous pieces
        self.canvas.delete("piece")
        
        # Draw pieces based on the current game state
        for i in range(3):
            for j in range(3):
                if self.game.board[i][j] == 'X':
                    self.draw_x(i, j)
                elif self.game.board[i][j] == 'O':
                    self.draw_o(i, j)
    
    def draw_x(self, row, col):
        # Calculate position
        x_center = col * 133 + 66.5
        y_center = row * 133 + 66.5
        offset = 40
        
        # Draw X
        self.canvas.create_line(
            x_center - offset, y_center - offset,
            x_center + offset, y_center + offset,
            width=5, fill="blue", tags="piece"
        )
        self.canvas.create_line(
            x_center + offset, y_center - offset,
            x_center - offset, y_center + offset,
            width=5, fill="blue", tags="piece"
        )
    
    def draw_o(self, row, col):
        # Calculate position
        x_center = col * 133 + 66.5
        y_center = row * 133 + 66.5
        radius = 40
        
        # Draw O
        self.canvas.create_oval(
            x_center - radius, y_center - radius,
            x_center + radius, y_center + radius,
            width=5, outline="red", tags="piece"
        )
    
    def create_controls(self):
        # Algorithm selection
        ttk.Label(self.control_frame, text="Select Algorithms:").pack(anchor="w", pady=(0, 5))
        
        # X player algorithm
        x_frame = ttk.Frame(self.control_frame)
        x_frame.pack(fill="x", pady=2)
        ttk.Label(x_frame, text="Player X:").pack(side="left")
        self.x_algorithm = tk.StringVar(value="minimax")
        ttk.Radiobutton(x_frame, text="Minimax", variable=self.x_algorithm, value="minimax").pack(side="left")
        ttk.Radiobutton(x_frame, text="Alpha-Beta", variable=self.x_algorithm, value="alphabeta").pack(side="left")
        
        # O player algorithm
        o_frame = ttk.Frame(self.control_frame)
        o_frame.pack(fill="x", pady=2)
        ttk.Label(o_frame, text="Player O:").pack(side="left")
        self.o_algorithm = tk.StringVar(value="alphabeta")
        ttk.Radiobutton(o_frame, text="Minimax", variable=self.o_algorithm, value="minimax").pack(side="left")
        ttk.Radiobutton(o_frame, text="Alpha-Beta", variable=self.o_algorithm, value="alphabeta").pack(side="left")
        
        # Speed control
        speed_frame = ttk.Frame(self.control_frame)
        speed_frame.pack(fill="x", pady=(10, 5))
        ttk.Label(speed_frame, text="Move Speed:").pack(side="left")
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(
            speed_frame, from_=0.1, to=2.0, 
            variable=self.speed_var, orient="horizontal"
        )
        speed_scale.pack(side="left", fill="x", expand=True)
        
        # Buttons
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(fill="x", pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start Game", command=self.start_game)
        self.start_button.pack(side="left", padx=5)
        
        self.reset_button = ttk.Button(button_frame, text="Reset", command=self.reset_game)
        self.reset_button.pack(side="left", padx=5)
        
        # Result display
        self.result_var = tk.StringVar(value="Ready to play")
        result_label = ttk.Label(self.control_frame, textvariable=self.result_var, font=("Arial", 12, "bold"))
        result_label.pack(pady=10)
        
        # Current player display
        self.player_var = tk.StringVar(value="Current: Player X")
        player_label = ttk.Label(self.control_frame, textvariable=self.player_var)
        player_label.pack(pady=5)
        
        # Thinking display
        self.thinking_var = tk.StringVar(value="")
        thinking_label = ttk.Label(self.control_frame, textvariable=self.thinking_var, font=("Arial", 10, "italic"))
        thinking_label.pack(pady=5)
        
        # Run comparison button
        comparison_frame = ttk.Frame(self.control_frame)
        comparison_frame.pack(fill="x", pady=10)
        
        ttk.Label(comparison_frame, text="Run Comparison (5 games each):").pack(anchor="w")
        
        self.compare_button = ttk.Button(
            comparison_frame, text="Compare Algorithms", 
            command=self.run_comparison_thread
        )
        self.compare_button.pack(fill="x", pady=5)
        
        self.show_graph_button = ttk.Button(
            comparison_frame, text="Show Graph", 
            command=self.show_graphs, state="disabled"
        )
        self.show_graph_button.pack(fill="x", pady=5)
        
    def create_stats_display(self):
        # Stats variables
        self.minimax_time_var = tk.StringVar(value="Minimax Time: N/A")
        self.alphabeta_time_var = tk.StringVar(value="Alpha-Beta Time: N/A")
        self.minimax_nodes_var = tk.StringVar(value="Minimax Nodes: N/A")
        self.alphabeta_nodes_var = tk.StringVar(value="Alpha-Beta Nodes: N/A")
        self.speedup_var = tk.StringVar(value="Speedup: N/A")
        
        # Stats labels
        ttk.Label(self.stats_frame, textvariable=self.minimax_time_var).pack(anchor="w")
        ttk.Label(self.stats_frame, textvariable=self.alphabeta_time_var).pack(anchor="w")
        ttk.Label(self.stats_frame, textvariable=self.minimax_nodes_var).pack(anchor="w")
        ttk.Label(self.stats_frame, textvariable=self.alphabeta_nodes_var).pack(anchor="w")
        ttk.Label(self.stats_frame, textvariable=self.speedup_var).pack(anchor="w")
    
    def start_game(self):
        if self.is_game_running:
            return
        
        self.reset_game()
        self.is_game_running = True
        self.start_button.configure(text="Game Running...", state="disabled")
        self.result_var.set("Game in progress...")
        
        # Create players based on selected algorithms
        self.x_player = MinimaxPlayer(algorithm=self.x_algorithm.get())
        self.o_player = MinimaxPlayer(algorithm=self.o_algorithm.get())
        
        # Start the game loop in a separate thread
        threading.Thread(target=self.game_loop, daemon=True).start()
    
    def game_loop(self):
        while self.is_game_running and not self.game.game_over():
            # Determine current player's algorithm
            current_player = self.x_player if self.game.current_player == 'X' else self.o_player
            self.player_var.set(f"Current: Player {self.game.current_player}")
            
            # Update thinking message
            algorithm_name = self.x_algorithm.get() if self.game.current_player == 'X' else self.o_algorithm.get()
            self.thinking_var.set(f"Thinking using {algorithm_name.capitalize()}...")
            self.root.update()
            
            # Get the move from the AI
            move, execution_time, nodes_evaluated = current_player.get_move(self.game)
            
            # Record performance metrics
            if algorithm_name == 'minimax':
                self.minimax_time_var.set(f"Minimax Time: {execution_time:.6f} s")
                self.minimax_nodes_var.set(f"Minimax Nodes: {nodes_evaluated}")
            else:  # alphabeta
                self.alphabeta_time_var.set(f"Alpha-Beta Time: {execution_time:.6f} s")
                self.alphabeta_nodes_var.set(f"Alpha-Beta Nodes: {nodes_evaluated}")
            
            # Make the move
            if move:
                row, col = move
                self.game.make_move(row, col)
                
                # Update the board
                self.root.after(0, self.update_board)
                self.thinking_var.set("")
                
                # Wait for the specified delay
                time.sleep(2.0 / self.speed_var.get())
        
        # Game over
        self.is_game_running = False
        result = self.game.get_result()
        
        if result == 'Draw':
            self.result_var.set("Game ended in a Draw!")
        else:
            algorithm = self.x_algorithm.get() if result == 'X' else self.o_algorithm.get()
            self.result_var.set(f"Player {result} ({algorithm}) wins!")
        
        self.player_var.set("")
        self.thinking_var.set("")
        self.start_button.configure(text="Start Game", state="normal")
    
    def reset_game(self):
        # Stop any ongoing game
        self.is_game_running = False
        
        # Reset the game state
        self.game.reset()
        
        # Update the display
        self.update_board()
        self.result_var.set("Ready to play")
        self.player_var.set("Current: Player X")
        self.thinking_var.set("")
        self.start_button.configure(text="Start Game", state="normal")
    
    def run_comparison_thread(self):
        # Disable controls during comparison
        self.compare_button.configure(text="Running...", state="disabled")
        self.start_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        
        # Run comparison in a separate thread
        threading.Thread(target=self.run_comparison, daemon=True).start()
    
    def run_comparison(self, num_games=5):
        # Reset metrics
        self.minimax_times = []
        self.alphabeta_times = []
        self.minimax_nodes = []
        self.alphabeta_nodes = []
        self.results = {"X": 0, "O": 0, "Draw": 0}
        
        # Update status
        self.result_var.set("Running comparison...")
        
        for game_num in range(num_games):
            self.thinking_var.set(f"Running game {game_num + 1}/{num_games}...")
            
            # Run a game with Minimax
            game = TicTacToe()
            minimax_player = MinimaxPlayer(algorithm='minimax')
            minimax_game_time, minimax_game_nodes = self.play_comparison_game(game, minimax_player)
            self.minimax_times.append(minimax_game_time)
            self.minimax_nodes.append(minimax_game_nodes)
            
            # Record result
            result = game.get_result()
            if result:
                self.results[result] += 1
            
            # Run a game with Alpha-Beta pruning
            game = TicTacToe()
            alphabeta_player = MinimaxPlayer(algorithm='alphabeta')
            alphabeta_game_time, alphabeta_game_nodes = self.play_comparison_game(game, alphabeta_player)
            self.alphabeta_times.append(alphabeta_game_time)
            self.alphabeta_nodes.append(alphabeta_game_nodes)
        
        # Update stats display
        avg_minimax_time = sum(self.minimax_times) / len(self.minimax_times)
        avg_alphabeta_time = sum(self.alphabeta_times) / len(self.alphabeta_times)
        avg_minimax_nodes = sum(self.minimax_nodes) / len(self.minimax_nodes)
        avg_alphabeta_nodes = sum(self.alphabeta_nodes) / len(self.alphabeta_nodes)
        speedup = avg_minimax_time / avg_alphabeta_time if avg_alphabeta_time > 0 else 0
        
        self.minimax_time_var.set(f"Minimax Time: {avg_minimax_time:.6f} s")
        self.alphabeta_time_var.set(f"Alpha-Beta Time: {avg_alphabeta_time:.6f} s")
        self.minimax_nodes_var.set(f"Minimax Nodes: {avg_minimax_nodes:.2f}")
        self.alphabeta_nodes_var.set(f"Alpha-Beta Nodes: {avg_alphabeta_nodes:.2f}")
        self.speedup_var.set(f"Speedup: {speedup:.2f}x")
        
        # Enable graph button and controls
        self.root.after(0, lambda: self.show_graph_button.configure(state="normal"))
        self.root.after(0, lambda: self.compare_button.configure(text="Compare Algorithms", state="normal"))
        self.root.after(0, lambda: self.start_button.configure(state="normal"))
        self.root.after(0, lambda: self.reset_button.configure(state="normal"))
        self.root.after(0, lambda: self.result_var.set("Comparison complete! Click 'Show Graph' to see results."))
        self.root.after(0, lambda: self.thinking_var.set(""))
    
    def play_comparison_game(self, game, player):
        total_time = 0
        total_nodes = 0
        
        while not game.game_over():
            move, execution_time, nodes_evaluated = player.get_move(game)
            
            total_time += execution_time
            total_nodes += nodes_evaluated
            
            if move:
                row, col = move
                game.make_move(row, col)
        
        return total_time, total_nodes
    
    def show_graphs(self):
        # Create a new window for the graphs
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Algorithm Performance Comparison")
        graph_window.geometry("800x600")
        
        # Create a figure with two subplots
        fig = plt.Figure(figsize=(10, 8))
        
        # Plot execution times
        ax1 = fig.add_subplot(211)
        games = range(1, len(self.minimax_times) + 1)
        
        ax1.bar([i - 0.2 for i in games], self.minimax_times, width=0.4, label='Minimax', color='blue', alpha=0.7)
        ax1.bar([i + 0.2 for i in games], self.alphabeta_times, width=0.4, label='Alpha-Beta', color='red', alpha=0.7)
        
        ax1.set_xlabel('Game Number')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time Comparison: Minimax vs Alpha-Beta Pruning')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot nodes evaluated
        ax2 = fig.add_subplot(212)
        
        ax2.bar([i - 0.2 for i in games], self.minimax_nodes, width=0.4, label='Minimax', color='blue', alpha=0.7)
        ax2.bar([i + 0.2 for i in games], self.alphabeta_nodes, width=0.4, label='Alpha-Beta', color='red', alpha=0.7)
        
        ax2.set_xlabel('Game Number')
        ax2.set_ylabel('Nodes Evaluated')
        ax2.set_title('Nodes Evaluated Comparison: Minimax vs Alpha-Beta Pruning')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Display the results
        results_text = (
            f"Results Summary:\n"
            f"X wins: {self.results.get('X', 0)}\n"
            f"O wins: {self.results.get('O', 0)}\n"
            f"Draws: {self.results.get('Draw', 0)}\n\n"
            f"Average Minimax time: {sum(self.minimax_times)/len(self.minimax_times):.6f} seconds\n"
            f"Average Alpha-Beta time: {sum(self.alphabeta_times)/len(self.alphabeta_times):.6f} seconds\n"
            f"Average Minimax nodes evaluated: {sum(self.minimax_nodes)/len(self.minimax_nodes):.2f}\n"
            f"Average Alpha-Beta nodes evaluated: {sum(self.alphabeta_nodes)/len(self.alphabeta_nodes):.2f}\n"
            f"Alpha-Beta speedup: {sum(self.minimax_times)/sum(self.alphabeta_times):.2f}x\n"
            f"Alpha-Beta node reduction: {sum(self.minimax_nodes)/sum(self.alphabeta_nodes):.2f}x"
        )
        
        results_label = ttk.Label(graph_window, text=results_text, justify="left", font=("Arial", 10))
        results_label.pack(pady=10)
        
        # Add the graph to the window
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Make sure the window stays on top and give it focus
        graph_window.transient(self.root)
        graph_window.focus_set()
        graph_window.grab_set()

def main():
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()