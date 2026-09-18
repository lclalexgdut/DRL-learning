"""最小化 Q-learning 示例：PyTorch + CUDA + 可视化

保留核心内容：
1. GridWorld 环境
2. ε-greedy 动作选择
3. Q-learning TD 更新
4. CUDA / CPU 自动设备选择
5. 训练统计
6. 训练曲线、Q 表、最终策略与路径可视化

说明：
- 这是表格型 Q-learning，不使用神经网络、反向传播或优化器。
- 默认 device='auto'：有 CUDA 就使用 CUDA，否则使用 CPU。
- 本例 Q 表只有 24×4，规模极小；CUDA 在这里主要用于演示设备管理，
  并不意味着它会比 CPU 更快。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch


# ============================================================
# 0. 全局设置
# ============================================================
SEED = 7
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "q_learning_figures"


def set_seed(seed: int) -> None:
    """固定 CPU/CUDA 随机种子。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_name: str = "auto") -> torch.device:
    """选择运行设备：auto / cpu / cuda。"""
    device_name = device_name.lower()

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "你指定了 --device cuda，但当前 PyTorch 未检测到可用 CUDA。\n"
                "请检查 NVIDIA 驱动、CUDA 版 PyTorch 是否正确安装，"
                "或改用 --device cpu。"
            )
        return torch.device("cuda")

    if device_name == "cpu":
        return torch.device("cpu")

    raise ValueError("device 只能是 auto、cpu 或 cuda")


def print_device_info(device: torch.device) -> None:
    """输出 PyTorch 与 CUDA 设备信息。"""
    print("=== 运行设备 ===")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print(f"当前设备: {device}")

    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        total_memory_gb = props.total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(index)}")
        print(f"显存: {total_memory_gb:.2f} GB")
        print(f"CUDA runtime: {torch.version.cuda}")


# ============================================================
# 1. 环境定义
# ============================================================
@dataclass
class StepResult:
    next_state: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor


class TorchGridWorld:
    """一个 5×5 的确定性网格环境。"""

    ACTIONS = ("上", "右", "下", "左")
    DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def __init__(self, device: torch.device):
        self.device = device
        self.rows, self.cols = 5, 5

        self.start = (4, 0)
        self.goal = (0, 4)
        self.traps = {(1, 3), (3, 2)}
        self.walls = {(2, 2)}

        # 墙不能进入，因此 25 个格子中只有 24 个有效状态。
        self.positions = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.walls
        ]
        self.pos_to_state = {pos: i for i, pos in enumerate(self.positions)}
        self.state_to_pos = {i: pos for pos, i in self.pos_to_state.items()}

        self.n_states = len(self.positions)
        self.n_actions = len(self.ACTIONS)

        # 环境转移表与 Q 表位于同一个 device。
        self.next_states = torch.empty(
            (self.n_states, self.n_actions),
            dtype=torch.long,
            device=self.device,
        )
        self.rewards = torch.empty(
            (self.n_states, self.n_actions),
            dtype=torch.float32,
            device=self.device,
        )
        self.dones = torch.empty(
            (self.n_states, self.n_actions),
            dtype=torch.bool,
            device=self.device,
        )

        self._build_transition_tables()
        self.state = torch.tensor(
            self.pos_to_state[self.start],
            dtype=torch.long,
            device=self.device,
        )

    def _build_transition_tables(self) -> None:
        """预先建立所有 (state, action) 对应的环境反馈。"""
        for state, pos in self.state_to_pos.items():
            for action, (dr, dc) in enumerate(self.DELTAS):
                candidate = (pos[0] + dr, pos[1] + dc)

                collision = (
                    candidate[0] < 0
                    or candidate[0] >= self.rows
                    or candidate[1] < 0
                    or candidate[1] >= self.cols
                    or candidate in self.walls
                )

                new_pos = pos if collision else candidate

                if collision:
                    reward = -1.0
                elif new_pos == self.goal:
                    reward = 10.0
                elif new_pos in self.traps:
                    reward = -10.0
                else:
                    reward = -0.1

                done = new_pos == self.goal or new_pos in self.traps

                self.next_states[state, action] = self.pos_to_state[new_pos]
                self.rewards[state, action] = reward
                self.dones[state, action] = done

    def reset(self) -> torch.Tensor:
        self.state = torch.tensor(
            self.pos_to_state[self.start],
            dtype=torch.long,
            device=self.device,
        )
        return self.state.clone()

    def step(self, action: int | torch.Tensor) -> StepResult:
        action = torch.as_tensor(action, dtype=torch.long, device=self.device)

        next_state = self.next_states[self.state, action]
        reward = self.rewards[self.state, action]
        done = self.dones[self.state, action]

        self.state = next_state.clone()
        return StepResult(
            next_state=next_state.clone(),
            reward=reward.clone(),
            done=done.clone(),
        )


# ============================================================
# 2. ε-greedy 动作选择
# ============================================================
def select_action(
    q_row: torch.Tensor,
    epsilon: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """以 ε 概率随机探索，否则利用当前 Q 表。"""

    # 随机数直接在 q_row 所在设备生成。
    random_value = torch.rand(
        (), device=q_row.device, generator=generator
    )

    if random_value.item() < epsilon:
        return torch.randint(
            q_row.numel(),
            (),
            device=q_row.device,
            generator=generator,
        )

    # 多个动作 Q 值并列最大时随机打破平局。
    best_actions = torch.where(q_row == q_row.max())[0]
    pick = torch.randint(
        best_actions.numel(),
        (),
        device=q_row.device,
        generator=generator,
    )
    return best_actions[pick]


# ============================================================
# 3. Q-learning 训练
# ============================================================
def train_q_learning(
    env: TorchGridWorld,
    episodes: int = 800,
    alpha: float = 0.15,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.992,
    max_steps: int = 100,
    seed: int = SEED,
):
    """训练表格型 Q-learning。

    非终止状态：
        target = r + gamma * max_a' Q(s', a')

    终止状态：
        target = r

    更新：
        Q(s,a) <- Q(s,a) + alpha * [target - Q(s,a)]
    """

    # CUDA 使用 CUDA generator；CPU 使用 CPU generator。
    generator = torch.Generator(device=env.device).manual_seed(seed)

    # Q 表真正放到选择的 CPU/CUDA 设备上。
    Q = torch.zeros(
        (env.n_states, env.n_actions),
        dtype=torch.float32,
        device=env.device,
    )

    # 训练日志保留在 CPU，方便打印与 Matplotlib 可视化。
    returns = torch.empty(episodes, dtype=torch.float32)
    successes = torch.empty(episodes, dtype=torch.bool)
    lengths = torch.empty(episodes, dtype=torch.long)
    epsilons = torch.empty(episodes, dtype=torch.float32)

    epsilon = epsilon_start

    with torch.no_grad():
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0.0
            success = False

            for step in range(1, max_steps + 1):
                # 1) 根据当前 Q[s] 使用 ε-greedy 选择动作。
                action = select_action(Q[state], epsilon, generator)

                # 2) 与环境交互，得到 r、s' 和 done。
                result = env.step(action)

                # 3) 构造 TD target。
                if result.done.item():
                    next_best = torch.zeros((), device=env.device)
                else:
                    next_best = Q[result.next_state].max()

                td_target = result.reward + gamma * next_best

                # 4) Q-learning 核心更新。
                td_error = td_target - Q[state, action]
                Q[state, action] += alpha * td_error

                # 5) 进入下一状态。
                total_reward += result.reward.item()
                state = result.next_state

                if result.done.item():
                    success = env.state_to_pos[state.item()] == env.goal
                    break

            # 训练日志写到 CPU。
            returns[episode] = total_reward
            successes[episode] = success
            lengths[episode] = step
            epsilons[episode] = epsilon

            # 逐回合降低探索率。
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

    history = {
        "returns": returns,
        "successes": successes,
        "lengths": lengths,
        "epsilons": epsilons,
    }
    return Q, history


# ============================================================
# 4. 贪心测试
# ============================================================
def greedy_rollout(
    env: TorchGridWorld,
    Q: torch.Tensor,
    max_steps: int = 50,
):
    """训练完成后关闭探索，只执行 argmax Q(s,a)。"""

    state = env.reset()
    route = [env.state_to_pos[state.item()]]
    total_reward = 0.0

    with torch.no_grad():
        for _ in range(max_steps):
            action = Q[state].argmax()
            result = env.step(action)

            total_reward += result.reward.item()
            state = result.next_state
            route.append(env.state_to_pos[state.item()])

            if result.done.item():
                break

    reached_goal = route[-1] == env.goal
    return reached_goal, route, total_reward


# ============================================================
# 5. 可视化辅助函数
# ============================================================
def moving_average(values: torch.Tensor, window: int = 50):
    """使用纯 PyTorch 计算滑动平均。"""
    values = values.detach().float().cpu()

    if values.numel() < window:
        x = torch.arange(1, values.numel() + 1)
        return x, values

    cumsum = torch.cat([torch.zeros(1), values.cumsum(dim=0)])
    smooth = (cumsum[window:] - cumsum[:-window]) / window
    x = torch.arange(window, values.numel() + 1)
    return x, smooth


def save_figure(fig, filename: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / filename, dpi=200, bbox_inches="tight")


def plot_training_reward(history: dict, window: int = 50) -> None:
    """图1：回合奖励及其滑动平均。"""
    returns = history["returns"].cpu()
    episodes = torch.arange(1, len(returns) + 1)
    x_smooth, y_smooth = moving_average(returns, window)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(episodes.numpy(), returns.numpy(), alpha=0.25, label="Episode return")
    ax.plot(
        x_smooth.numpy(),
        y_smooth.numpy(),
        linewidth=2,
        label=f"Moving average ({window})",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Q-learning Training Reward")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "01_training_reward.png")


def plot_success_and_epsilon(history: dict, window: int = 50) -> None:
    """图2：滑动成功率与 ε 衰减。二者都在 [0,1]，可以直接同轴比较。"""
    success = history["successes"].float().cpu()
    epsilons = history["epsilons"].cpu()
    episodes = torch.arange(1, len(epsilons) + 1)
    x_success, success_rate = moving_average(success, window)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(
        x_success.numpy(),
        success_rate.numpy(),
        linewidth=2,
        label=f"Success rate ({window})",
    )
    ax.plot(
        episodes.numpy(),
        epsilons.numpy(),
        linewidth=2,
        linestyle="--",
        label="Epsilon",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Exploration Decay and Success Rate")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "02_success_and_epsilon.png")


def plot_episode_length(history: dict, window: int = 50) -> None:
    """图3：每回合步数及滑动平均。"""
    lengths = history["lengths"].float().cpu()
    episodes = torch.arange(1, len(lengths) + 1)
    x_smooth, y_smooth = moving_average(lengths, window)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(episodes.numpy(), lengths.numpy(), alpha=0.25, label="Episode length")
    ax.plot(
        x_smooth.numpy(),
        y_smooth.numpy(),
        linewidth=2,
        label=f"Moving average ({window})",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Length During Training")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "03_episode_length.png")


def plot_q_table(Q: torch.Tensor, env: TorchGridWorld) -> None:
    """图4：最终 Q 表热力图。绘图前显式移回 CPU。"""
    q_cpu = Q.detach().cpu()

    fig, ax = plt.subplots(figsize=(7.5, 8))
    image = ax.imshow(q_cpu.numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(env.n_actions))
    ax.set_xticklabels([f"a={i}" for i in range(env.n_actions)])
    ax.set_yticks(range(env.n_states))
    ax.set_yticklabels([str(i) for i in range(env.n_states)], fontsize=8)
    ax.set_xlabel("Action")
    ax.set_ylabel("State")
    ax.set_title("Final Q Table")
    fig.colorbar(image, ax=ax, label="Q(s, a)")
    fig.tight_layout()
    save_figure(fig, "04_q_table.png")


def plot_final_policy(
    env: TorchGridWorld,
    Q: torch.Tensor,
    route: list[tuple[int, int]],
) -> None:
    """图5：最终贪心策略 + 实际测试路线。"""
    q_cpu = Q.detach().cpu()

    # grid value 仅用于提供浅色背景，不承担数值含义。
    grid = torch.zeros((env.rows, env.cols), dtype=torch.float32)
    for pos in env.walls:
        grid[pos] = 1.0
    for pos in env.traps:
        grid[pos] = 0.6
    grid[env.goal] = 0.3

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(grid.numpy(), cmap="Greys", vmin=0.0, vmax=1.2)

    # 画最终贪心策略箭头。
    for state, pos in env.state_to_pos.items():
        if pos == env.goal or pos in env.traps:
            continue

        action = int(q_cpu[state].argmax().item())
        dr, dc = env.DELTAS[action]
        ax.arrow(
            pos[1],
            pos[0],
            dc * 0.25,
            dr * 0.25,
            head_width=0.10,
            head_length=0.08,
            length_includes_head=True,
        )

    # 叠加贪心测试得到的实际路线。
    xs = [p[1] for p in route]
    ys = [p[0] for p in route]
    ax.plot(xs, ys, marker="o", linewidth=2.5, label="Greedy route")

    ax.text(env.start[1], env.start[0], "S", ha="center", va="center")
    ax.text(env.goal[1], env.goal[0], "G", ha="center", va="center")
    for pos in env.traps:
        ax.text(pos[1], pos[0], "X", ha="center", va="center")
    for pos in env.walls:
        ax.text(pos[1], pos[0], "W", ha="center", va="center")

    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_xticks([x - 0.5 for x in range(1, env.cols)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, env.rows)], minor=True)
    ax.grid(which="minor", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Final Greedy Policy and Route")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "05_final_policy_route.png")


def visualize_results(
    history: dict,
    Q: torch.Tensor,
    env: TorchGridWorld,
    route: list[tuple[int, int]],
    show: bool = True,
) -> None:
    """统一生成全部可视化。"""
    plot_training_reward(history)
    plot_success_and_epsilon(history)
    plot_episode_length(history)
    plot_q_table(Q, env)
    plot_final_policy(env, Q, route)

    print(f"可视化图片已保存到: {FIG_DIR.resolve()}")

    if show:
        plt.show()
    else:
        plt.close("all")


# ============================================================
# 6. 主程序
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal tabular Q-learning with CUDA and visualization"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="运行设备：auto=有 CUDA 就用 CUDA，否则 CPU",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=800,
        help="训练回合数",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="只保存图片，不弹出 Matplotlib 窗口",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(SEED)
    device = choose_device(args.device)
    print_device_info(device)

    env = TorchGridWorld(device)

    Q, history = train_q_learning(
        env,
        episodes=args.episodes,
    )

    reached_goal, route, total_reward = greedy_rollout(
        TorchGridWorld(device),
        Q,
    )

    print("\n=== Q-learning 训练完成 ===")
    print(f"状态数: {env.n_states}")
    print(f"动作数: {env.n_actions}")
    print(f"Q 表形状: {tuple(Q.shape)}")
    print(f"Q 表所在设备: {Q.device}")
    print(f"最后100回合平均奖励: {history['returns'][-100:].mean().item():.2f}")
    print(
        f"最后100回合成功率: "
        f"{history['successes'][-100:].float().mean().item():.1%}"
    )
    print(
        f"最后100回合平均步数: "
        f"{history['lengths'][-100:].float().mean().item():.1f}"
    )

    print("\n=== 关闭探索后的贪心测试 ===")
    print(f"是否到达终点: {reached_goal}")
    print(f"总奖励: {total_reward:.1f}")
    print(f"路线: {route}")

    print("\n=== 最终 Q 表（转到 CPU 后打印）===")
    print(Q.detach().cpu())

    visualize_results(
        history=history,
        Q=Q,
        env=env,
        route=route,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
