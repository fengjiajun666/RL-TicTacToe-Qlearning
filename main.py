import tkinter as tk
from tkinter import messagebox
import numpy as np
import random
import matplotlib.pyplot as plt


# --- 1. Game Environment ---
class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current_winner = None

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        # Check rows
        row_ind = square // 3
        row = self.board[row_ind * 3: (row_ind + 1) * 3]
        if all([spot == letter for spot in row]): return True
        # Check columns
        col_ind = square % 3
        column = [self.board[col_ind + i * 3] for i in range(3)]
        if all([spot == letter for spot in column]): return True
        # Check diagonals
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([spot == letter for spot in diagonal1]): return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([spot == letter for spot in diagonal2]): return True
        return False

    def get_state(self):
        return tuple(self.board)


# --- 2. Q-Learning Agent ---
class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_moves):
        # Epsilon-greedy implementation
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_moves)  # Explore

        q_values = [self.get_q_value(state, a) for a in available_moves]
        max_q = max(q_values)
        best_actions = [a for a in available_moves if self.get_q_value(state, a) == max_q]
        return random.choice(best_actions)  # Exploit

    def learn(self, state, action, reward, next_state, next_available_moves):
        # Bellman equation update
        current_q = self.get_q_value(state, action)
        max_next_q = max(
            [self.get_q_value(next_state, a) for a in next_available_moves]) if next_available_moves else 0.0
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q


# --- 3. GUI and Enhanced Training Loop ---
class TicTacToeUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Q-Learning Tic-Tac-Toe")
        self.game = TicTacToe()
        self.agent = QLearningAgent(epsilon=0.0)  # Turn off exploration for human play

        self.info_label = tk.Label(root, text="Training Agent... Please wait.", font=('Arial', 14))
        self.info_label.grid(row=0, column=0, columnspan=3)

        self.buttons = []
        for i in range(9):
            btn = tk.Button(root, text=' ', font=('Arial', 40), width=4, height=2,
                            command=lambda i=i: self.human_click(i))
            btn.grid(row=(i // 3) + 1, column=i % 3)
            self.buttons.append(btn)

        # Execute enhanced training with 50,000 episodes
        self.train_agent(50000)
        self.info_label.config(text="Training complete! You: X | Agent: O\nYour Turn!")

    def train_agent(self, episodes):
        # Start with 100% exploration
        train_agent = QLearningAgent(epsilon=1.0)
        batch_size = 500
        win_rates = []
        agent_wins = 0

        for ep in range(episodes):
            # Core Improvement 1: Epsilon Decay
            train_agent.epsilon = max(0.01, train_agent.epsilon * 0.9999)
            env = TicTacToe()
            last_state_action = None  # Safe initialization

            while True:
                # Core Improvement 2: Self-Play Proxy (Opponent uses Q-table 70% of the time)
                if random.uniform(0, 1) < 0.3:
                    x_action = random.choice(env.available_moves())
                else:
                    x_action = train_agent.choose_action(env.get_state(), env.available_moves())

                env.make_move(x_action, 'X')

                if env.current_winner == 'X':
                    if last_state_action:
                        train_agent.learn(last_state_action[0], last_state_action[1], -10, env.get_state(), [])
                    break
                elif not env.available_moves():
                    if last_state_action:
                        train_agent.learn(last_state_action[0], last_state_action[1], 5, env.get_state(), [])
                    break

                # Agent (O) Turn
                state = env.get_state()
                available = env.available_moves()
                action = train_agent.choose_action(state, available)
                env.make_move(action, 'O')
                next_state = env.get_state()

                if env.current_winner == 'O':
                    train_agent.learn(state, action, 10, next_state, [])
                    agent_wins += 1
                    break
                elif not env.available_moves():
                    train_agent.learn(state, action, 5, next_state, [])
                    break
                else:
                    train_agent.learn(state, action, 0, next_state, env.available_moves())

                last_state_action = (state, action)

            # Record data for plotting
            if (ep + 1) % batch_size == 0:
                win_rates.append(agent_wins / batch_size)
                agent_wins = 0

        self.agent.q_table = train_agent.q_table

        # Display chart (Close the chart window to reveal the Tkinter game UI)
        plt.figure(figsize=(8, 5))
        plt.plot(range(batch_size, episodes + 1, batch_size), win_rates, label='Agent Win Rate (Self-Play)',
                 color='purple')
        plt.title('Q-Learning Agent Win Rate Over 50,000 Episodes')
        plt.xlabel('Training Episodes')
        plt.ylabel('Win Rate')
        plt.grid(True)
        plt.legend()
        plt.show()

        # --- UI Interaction Logic ---

    def human_click(self, index):
        if self.game.board[index] == ' ' and not self.game.current_winner:
            self.game.make_move(index, 'X')
            self.update_ui()

            if self.game.current_winner == 'X':
                self.info_label.config(text="You Win! (Agent received Penalty)")
                self.game_over("You Win!")
                return
            elif not self.game.available_moves():
                self.info_label.config(text="Draw! (Agent received Draw Reward)")
                self.game_over("Draw!")
                return

            self.info_label.config(text="Agent is thinking...")
            self.root.after(300, self.agent_move)

    def agent_move(self):
        state = self.game.get_state()
        available = self.game.available_moves()
        action = self.agent.choose_action(state, available)

        self.game.make_move(action, 'O')
        self.update_ui()

        if self.game.current_winner == 'O':
            self.info_label.config(text="Agent Wins! (Agent received Reward)")
            self.game_over("Agent Wins!")
        elif not self.game.available_moves():
            self.info_label.config(text="Draw! (Agent received Draw Reward)")
            self.game_over("Draw!")
        else:
            self.info_label.config(text="Your Turn! (Agent Action: " + str(action) + ")")

    def update_ui(self):
        for i in range(9):
            self.buttons[i].config(text=self.game.board[i])

    def game_over(self, msg):
        messagebox.showinfo("Game Over", msg)
        self.reset_game()

    def reset_game(self):
        self.game = TicTacToe()
        self.update_ui()
        self.info_label.config(text="You: X | Agent: O\nYour Turn!")


if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeUI(root)
    root.mainloop()