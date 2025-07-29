import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from tqdm import tqdm
import os
import multiprocessing as mp
import time

# --- 1. 超参数和全局设置 ---
class Config:
    def __init__(self):
        self.BOARD_WIDTH = 9      # 棋盘宽度
        self.BOARD_HEIGHT = 9     # 棋盘高度
        self.N_IN_ROW = 5         # 获胜所需的连子数
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.EPOCHS = 5000
        self.BATCH_SIZE = 512
        self.SELF_PLAY_GAMES = 500
        self.LEARNING_RATE = 2e-3
        self.BUFFER_SIZE = 20000
        self.CHECKPOINT_INTERVAL = 10
        self.C_PUCT = 5
        self.MCTS_SIMULATIONS = 400
        self.DIRICHLET_ALPHA = 0.3
        self.NOISE_EPSILON = 0.25
        self.NUM_WORKERS = max(1, mp.cpu_count() - 2)

# --- (模块 2, 3, 4: GomokuEnv, PolicyValueNet, MCTS 的代码与之前完全相同) ---
# 为了保证完整性，这里将它们全部包含进来

# --- 2. 游戏环境 ---
class GomokuEnv:
    def __init__(self, config):
        self.width = config.BOARD_WIDTH
        self.height = config.BOARD_HEIGHT
        self.n_in_row = config.N_IN_ROW
        self.board = np.zeros((self.height, self.width))
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((self.height, self.width))
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        state = np.zeros((3, self.height, self.width))
        state[0] = (self.board == self.current_player)
        state[1] = (self.board == -self.current_player)
        if self.current_player == 1:
            state[2] = 1.0
        return state

    def get_available_moves(self):
        return list(zip(*np.where(self.board == 0)))

    def step(self, action):
        row, col = action
        if self.board[row, col] != 0:
            raise ValueError("Invalid move")
        self.board[row, col] = self.current_player
        is_win, winner = self.check_winner()
        is_tie = not is_win and len(self.get_available_moves()) == 0
        self.current_player *= -1
        next_state = self.get_state()
        if is_win:
            return next_state, winner, True
        if is_tie:
            return next_state, 0, True
        return next_state, 0, False

    def check_winner(self):
        for r in range(self.height):
            for c in range(self.width):
                player = self.board[r, c]
                if player == 0: continue
                if c + self.n_in_row <= self.width and np.all(self.board[r, c:c+self.n_in_row] == player): return True, player
                if r + self.n_in_row <= self.height and np.all(self.board[r:r+self.n_in_row, c] == player): return True, player
                if r + self.n_in_row <= self.height and c + self.n_in_row <= self.width and np.all(np.diag(self.board[r:r+self.n_in_row, c:c+self.n_in_row]) == player): return True, player
                if r + self.n_in_row <= self.height and c - self.n_in_row + 1 >= 0:
                    sub_board = self.board[r:r+self.n_in_row, c-self.n_in_row+1:c+1]
                    if np.all(np.diag(np.fliplr(sub_board)) == player):
                         return True, player
        return False, 0

# --- 3. 策略价值网络 ---
class ResBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class PolicyValueNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.width = config.BOARD_WIDTH
        self.height = config.BOARD_HEIGHT
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64), ResBlock(64), ResBlock(64), ResBlock(64)
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(),
            nn.Linear(2 * self.width * self.height, self.width * self.height)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.BatchNorm2d(1), nn.ReLU(), nn.Flatten(),
            nn.Linear(1 * self.width * self.height, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Tanh()
        )
    def forward(self, x):
        x = self.conv_block(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return F.softmax(policy_logits, dim=1), value

# --- 4. MCTS ---
class MCTSNode:
    def __init__(self, parent, prior_p):
        self.parent = parent; self.children = {}; self.n_visits = 0
        self.Q = 0; self.u = 0; self.P = prior_p
    def expand(self, action_priors):
        for action, prob in enumerate(action_priors):
            if prob > 0 and action not in self.children:
                self.children[action] = MCTSNode(self, prob)
    def select(self, c_puct):
        return max(self.children.items(), key=lambda item: item[1].get_value(c_puct))
    def get_value(self, c_puct):
        self.u = c_puct * self.P * np.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.Q + self.u
    def update(self, leaf_value):
        self.n_visits += 1; self.Q += (leaf_value - self.Q) / self.n_visits
    def update_recursive(self, leaf_value):
        if self.parent: self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

class MCTS:
    def __init__(self, policy_value_fn, config):
        self.root = MCTSNode(None, 1.0); self.policy_value_fn = policy_value_fn; self.config = config
    def _playout(self, env):
        node = self.root
        while True:
            is_end, _ = env.check_winner()
            if is_end or not node.children: break
            action, node = node.select(self.config.C_PUCT)
            row, col = action // self.config.BOARD_WIDTH, action % self.config.BOARD_WIDTH
            env.step((row, col))
        is_end, winner = env.check_winner()
        if not is_end:
            state = env.get_state()
            action_probs, leaf_value = self.policy_value_fn(state)
            available_moves_indices = [m[0]*self.config.BOARD_WIDTH + m[1] for m in env.get_available_moves()]
            if available_moves_indices:
                valid_action_probs = action_probs[available_moves_indices]
                valid_action_probs_sum = np.sum(valid_action_probs)
                if valid_action_probs_sum > 0: valid_action_probs /= valid_action_probs_sum
                full_probs = np.zeros_like(action_probs); full_probs[available_moves_indices] = valid_action_probs
                node.expand(full_probs)
            else: leaf_value = 0.0
        else: leaf_value = float(winner) if winner !=0 else 0.0
        node.update_recursive(-leaf_value)
    def get_move_probs(self, env, temp=1e-3):
        for _ in range(self.config.MCTS_SIMULATIONS):
            env_copy = GomokuEnv(self.config); env_copy.board = np.copy(env.board); env_copy.current_player = env.current_player
            self._playout(env_copy)
        act_visits = [(act, node.n_visits) for act, node in self.root.children.items()]
        if not act_visits: return [],[]
        acts, visits = zip(*act_visits)
        act_probs = F.softmax(torch.tensor(visits, dtype=torch.float32) / temp, dim=0).numpy()
        return acts, act_probs
    def update_with_move(self, last_move):
        if last_move in self.root.children: self.root = self.root.children[last_move]; self.root.parent = None
        else: self.root = MCTSNode(None, 1.0)


# --- 5. 并行化工作函数及初始化器 ---

# <<< 关键改动点 1: 为每个工作进程设置的全局变量 >>>
worker_model = None
worker_config = None

def init_worker(config, model_state_dict):
    """
    进程池中每个工作进程的初始化函数。
    只在进程启动时调用一次。
    """
    global worker_model, worker_config
    worker_config = config
    
    # 在子进程中重新构建模型，并加载从主进程传来的权重
    worker_model = PolicyValueNet(worker_config).to('cpu')
    worker_model.load_state_dict(model_state_dict)
    worker_model.eval()

def run_one_game(_):
    """
    这个函数在一个独立的进程中运行，负责完成一盘自我对弈。
    它不再需要接收参数，而是直接使用本进程中的全局变量。
    """
    global worker_model, worker_config
    env = GomokuEnv(worker_config)
    
    def policy_value_fn(state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to('cpu')
        with torch.no_grad():
            log_act_probs, value = worker_model(state_tensor)
        return log_act_probs.exp().cpu().numpy()[0], value.item()

    mcts = MCTS(policy_value_fn, worker_config)
    
    states, mcts_probs, players = [], [], []
    while True:
        acts, probs = mcts.get_move_probs(env)
        if not acts: break
        noise = np.random.dirichlet(worker_config.DIRICHLET_ALPHA * np.ones(len(probs)))
        policy_with_noise = (1 - worker_config.NOISE_EPSILON) * probs + worker_config.NOISE_EPSILON * noise
        policy_with_noise /= np.sum(policy_with_noise)
        move_idx = np.random.choice(len(acts), p=policy_with_noise)
        move = acts[move_idx]
        states.append(env.get_state())
        pi = np.zeros(worker_config.BOARD_WIDTH**2); pi[list(acts)] = probs
        mcts_probs.append(pi); players.append(env.current_player)
        row, col = move // worker_config.BOARD_WIDTH, move % worker_config.BOARD_WIDTH
        _, winner, done = env.step((row, col))
        mcts.update_with_move(move)
        if done:
            game_data = []
            z = np.zeros(len(players))
            if winner != 0:
                z[np.array(players) == winner] = 1.0
                z[np.array(players) != winner] = -1.0
            for i in range(len(states)):
                game_data.append((states[i], mcts_probs[i], z[i]))
            return game_data
    return []

# --- 6. 训练流程 ---
class Trainer:
    def __init__(self, config):
        self.config = config
        self.policy_value_net = PolicyValueNet(config).to(config.DEVICE)
        self.try_load_checkpoint()
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
        self.data_buffer = deque(maxlen=config.BUFFER_SIZE)

    def try_load_checkpoint(self):
        checkpoint_files = [f for f in os.listdir('.') if f.startswith('gomoku_model_epoch_') and f.endswith('.pth')]
        if not checkpoint_files:
            print("未找到检查点，从头开始训练。")
            self.start_epoch = 0
            return
        latest_epoch = max([int(f.split('_')[-1].split('.')[0]) for f in checkpoint_files])
        latest_file = f"gomoku_model_epoch_{latest_epoch}.pth"
        print(f"找到最新检查点: {latest_file}, 加载模型继续训练。")
        self.policy_value_net.load_state_dict(torch.load(latest_file, map_location=self.config.DEVICE))
        self.start_epoch = latest_epoch

    def collect_self_play_data(self):
        model_state_dict = {k: v.cpu() for k, v in self.policy_value_net.state_dict().items()}
        
        print(f"使用 {self.config.NUM_WORKERS} 个CPU核心并行进行自我对弈...")
        
        # <<< 关键改动点 2: 使用 initializer 初始化进程池 >>>
        with mp.Pool(processes=self.config.NUM_WORKERS, initializer=init_worker, initargs=(self.config, model_state_dict)) as pool:
            results = list(tqdm(pool.imap_unordered(run_one_game, range(self.config.SELF_PLAY_GAMES)),
                                total=self.config.SELF_PLAY_GAMES, desc="Self-Play (Parallel)"))
        games_data = [item for sublist in results for item in sublist]
        self.data_buffer.extend(games_data)

    def train_step(self):
        if len(self.data_buffer) < self.config.BATCH_SIZE:
            return None, None, None
        mini_batch = random.sample(self.data_buffer, self.config.BATCH_SIZE)
        state_batch = torch.tensor(np.array([data[0] for data in mini_batch]), dtype=torch.float32).to(self.config.DEVICE)
        mcts_probs_batch = torch.tensor(np.array([data[1] for data in mini_batch]), dtype=torch.float32).to(self.config.DEVICE)
        winner_batch = torch.tensor(np.array([data[2] for data in mini_batch]), dtype=torch.float32).to(self.config.DEVICE)
        self.policy_value_net.train()
        self.optimizer.zero_grad()
        log_act_probs, value = self.policy_value_net(state_batch)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs_batch * log_act_probs, 1))
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        return loss.item(), value_loss.item(), policy_loss.item()

    def run(self):
        print(f"开始训练，主进程设备: {self.config.DEVICE}")
        if self.config.DEVICE.type == 'cuda':
            print(f"显卡: {torch.cuda.get_device_name(0)}")
            
        for epoch in range(self.start_epoch, self.config.EPOCHS):
            print(f"--- Epoch {epoch+1}/{self.config.EPOCHS} ---")
            self.collect_self_play_data()
            if len(self.data_buffer) >= self.config.BATCH_SIZE:
                loss_total, v_loss_total, p_loss_total, steps = 0, 0, 0, 0
                train_steps = min(500, len(self.data_buffer) // self.config.BATCH_SIZE)
                for _ in tqdm(range(train_steps), desc="Training on GPU"):
                    l, v, p = self.train_step()
                    if l is not None:
                        loss_total += l; v_loss_total += v; p_loss_total += p; steps += 1
                if steps > 0:
                    print(f"Loss: {loss_total/steps:.4f}, Value Loss: {v_loss_total/steps:.4f}, Policy Loss: {p_loss_total/steps:.4f}")
            if (epoch + 1) % self.config.CHECKPOINT_INTERVAL == 0:
                filename = f"gomoku_model_epoch_{epoch+1}.pth"
                torch.save(self.policy_value_net.state_dict(), filename)
                print(f"模型已保存至 {filename}")

# --- 7. 主程序入口 ---
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    config = Config()
    trainer = Trainer(config)
    trainer.run()