"""10x10 stochastic GridWorld Q-learning demo: PyTorch + CUDA + visualization.

升级目标：
1. 从 5x5 确定性环境升级到 10x10 随机环境；
2. 保留表格型 Q-learning，不引入 DQN；
3. 动作存在随机偏移：默认 80% 按原动作执行，10% 左偏，10% 右偏；
4. 增加更多墙与陷阱，让训练过程更明显；
5. 支持 CPU/CUDA 自动切换；
6. 输出训练曲线、环境地图、Q 表、最终策略和策略演化图。

说明：
- 这仍然是“表格型 Q-learning”教学案例，Q 表不需要神经网络、反向传播和优化器。
- CUDA 在本例中主要用于演示张量设备管理；该 Q 表规模仍然很小，GPU 不一定更快。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch


SEED = 7
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "q_learning_10x10_figures"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_name: str = "auto") -> torch.device:
    device_name = device_name.lower()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "指定了 --device cuda，但当前 PyTorch 没有检测到可用 CUDA。\n"
                "请检查 NVIDIA 驱动与 CUDA 版 PyTorch，或改用 --device cpu。"
            )
        return torch.device("cuda")
    raise ValueError("device 只能是 auto、cpu 或 cuda")


def print_device_info(device: torch.device) -> None:
    print("=== 运行设备 ===")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print(f"当前设备: {device}")
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        print(f"GPU: {torch.cuda.get_device_name(index)}")
        print(f"显存: {props.total_memory / 1024**3:.2f} GB")
        print(f"CUDA runtime: {torch.version.cuda}")


@dataclass
class StepResult:
    next_state: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    intended_action: int
    executed_action: int


class StochasticGridWorld10x10:
    """10x10 随机 GridWorld。

    默认动作随机性：
        80% 执行原动作
        10% 向左偏移
        10% 向右偏移

    ACTIONS 顺序为 上、右、下、左，因此：
        左偏 = (action - 1) % 4
        右偏 = (action + 1) % 4
    """

    ACTIONS = ("上", "右", "下", "左")
    ARROWS = ("↑", "→", "↓", "←")
    DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def __init__(self, device: torch.device, slip_prob: float = 0.20):
        if not 0.0 <= slip_prob < 1.0:
            raise ValueError("slip_prob 必须位于 [0, 1) 区间")

        self.device = device
        self.rows, self.cols = 10, 10
        self.slip_prob = float(slip_prob)

        self.start = (9, 0)
        self.goal = (0, 9)

        # 墙：不能进入。
        self.walls = {
            (0, 3), (0, 4), (0, 8),
            (1, 6),
            (2, 4), (2, 7),
            (3, 7),
            (4, 1), (4, 2), (4, 3), (4, 5), (4, 8),
            (5, 1), (5, 7),
            (6, 5),
            (7, 8),
        }

        # 陷阱：进入后回合终止。
        # 环境仍存在多条可行通路，但需要通过学习避开高风险区域。
        self.traps = {
            (2, 5), (4, 9), (5, 4), (7, 6), (9, 5),
        }

        assert self.start not in self.walls | self.traps
        assert self.goal not in self.walls | self.traps

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

        # 这里保存“实际执行动作”对应的确定性转移。
        # 随机性由 step() 决定 intended_action 最终变成哪个 executed_action。
        self.next_states = torch.empty(
            (self.n_states, self.n_actions), dtype=torch.long, device=self.device
        )
        self.rewards = torch.empty(
            (self.n_states, self.n_actions), dtype=torch.float32, device=self.device
        )
        self.dones = torch.empty(
            (self.n_states, self.n_actions), dtype=torch.bool, device=self.device
        )
        self._build_transition_tables()

        self.state = torch.tensor(
            self.pos_to_state[self.start], dtype=torch.long, device=self.device
        )

    def _build_transition_tables(self) -> None:
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
                    reward = 20.0
                elif new_pos in self.traps:
                    reward = -20.0
                else:
                    # 每走一步都有轻微成本，因此会偏向较短路线。
                    reward = -0.10

                done = new_pos == self.goal or new_pos in self.traps

                self.next_states[state, action] = self.pos_to_state[new_pos]
                self.rewards[state, action] = reward
                self.dones[state, action] = done

    def reset(self) -> torch.Tensor:
        self.state = torch.tensor(
            self.pos_to_state[self.start], dtype=torch.long, device=self.device
        )
        return self.state.clone()

    def _sample_executed_action(
        self,
        intended_action: int,
        generator: torch.Generator,
    ) -> int:
        """根据 slip_prob 把“想执行的动作”变成“实际执行的动作”。"""
        if self.slip_prob <= 0.0:
            return intended_action

        u = torch.rand((), device=self.device, generator=generator).item()
        keep_prob = 1.0 - self.slip_prob

        if u < keep_prob:
            return intended_action
        if u < keep_prob + self.slip_prob / 2.0:
            return (intended_action - 1) % self.n_actions
        return (intended_action + 1) % self.n_actions

    def step(
        self,
        action: int | torch.Tensor,
        generator: torch.Generator,
        stochastic: bool = True,
    ) -> StepResult:
        intended_action = int(torch.as_tensor(action).item())
        executed_action = (
            self._sample_executed_action(intended_action, generator)
            if stochastic
            else intended_action
        )

        executed = torch.tensor(
            executed_action, dtype=torch.long, device=self.device
        )
        next_state = self.next_states[self.state, executed]
        reward = self.rewards[self.state, executed]
        done = self.dones[self.state, executed]

        self.state = next_state.clone()
        return StepResult(
            next_state=next_state.clone(),
            reward=reward.clone(),
            done=done.clone(),
            intended_action=intended_action,
            executed_action=executed_action,
        )


def select_action(
    q_row: torch.Tensor,
    epsilon: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, bool]:
    """ε-greedy：ε 概率探索，否则选择当前最大 Q 值动作。"""
    explore = torch.rand((), device=q_row.device, generator=generator).item() < epsilon

    if explore:
        action = torch.randint(
            q_row.numel(), (), device=q_row.device, generator=generator
        )
        return action, True

    best_actions = torch.where(q_row == q_row.max())[0]
    pick = torch.randint(
        best_actions.numel(), (), device=q_row.device, generator=generator
    )
    return best_actions[pick], False


def train_q_learning(
    env: StochasticGridWorld10x10,
    episodes: int = 2500,
    alpha: float = 0.12,
    gamma: float = 0.97,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.998,
    max_steps: int = 200,
    seed: int = SEED,
):
    """训练表格型 Q-learning。"""
    generator = torch.Generator(device=env.device).manual_seed(seed)

    Q = torch.zeros(
        (env.n_states, env.n_actions), dtype=torch.float32, device=env.device
    )

    returns = torch.empty(episodes, dtype=torch.float32)
    successes = torch.empty(episodes, dtype=torch.bool)
    lengths = torch.empty(episodes, dtype=torch.long)
    epsilons = torch.empty(episodes, dtype=torch.float32)
    explore_ratios = torch.empty(episodes, dtype=torch.float32)
    slip_ratios = torch.empty(episodes, dtype=torch.float32)

    # 保存若干关键训练阶段，用于画“策略如何长出来”。
    snapshot_points = sorted(
        set([0, max(1, episodes // 20), episodes // 5, episodes // 2, episodes])
    )
    snapshots: dict[int, torch.Tensor] = {0: Q.detach().cpu().clone()}

    epsilon = epsilon_start

    with torch.no_grad():
        for episode in range(1, episodes + 1):
            state = env.reset()
            total_reward = 0.0
            success = False
            explore_count = 0
            slip_count = 0

            for step in range(1, max_steps + 1):
                action, explored = select_action(Q[state], epsilon, generator)
                explore_count += int(explored)

                result = env.step(action, generator=generator, stochastic=True)
                slip_count += int(result.executed_action != result.intended_action)

                if result.done.item():
                    next_best = torch.zeros((), device=env.device)
                else:
                    next_best = Q[result.next_state].max()

                # Q-learning TD target
                td_target = result.reward + gamma * next_best
                td_error = td_target - Q[state, action]
                Q[state, action] += alpha * td_error

                total_reward += result.reward.item()
                state = result.next_state

                if result.done.item():
                    success = env.state_to_pos[state.item()] == env.goal
                    break

            returns[episode - 1] = total_reward
            successes[episode - 1] = success
            lengths[episode - 1] = step
            epsilons[episode - 1] = epsilon
            explore_ratios[episode - 1] = explore_count / step
            slip_ratios[episode - 1] = slip_count / step

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            if episode in snapshot_points:
                snapshots[episode] = Q.detach().cpu().clone()

    history = {
        "returns": returns,
        "successes": successes,
        "lengths": lengths,
        "epsilons": epsilons,
        "explore_ratios": explore_ratios,
        "slip_ratios": slip_ratios,
        "snapshots": snapshots,
    }
    return Q, history


def greedy_rollout(
    env: StochasticGridWorld10x10,
    Q: torch.Tensor,
    max_steps: int = 200,
    seed: int = SEED + 1000,
    stochastic: bool = False,
):
    """关闭探索，只执行 argmax Q(s,a)。

    stochastic=False：用于画最终“名义路线”，不加入动作打滑。
    stochastic=True：用于真实随机环境下的一次策略测试。
    """
    generator = torch.Generator(device=env.device).manual_seed(seed)
    state = env.reset()
    route = [env.state_to_pos[state.item()]]
    total_reward = 0.0

    with torch.no_grad():
        for _ in range(max_steps):
            action = Q[state].argmax()
            result = env.step(action, generator=generator, stochastic=stochastic)
            total_reward += result.reward.item()
            state = result.next_state
            route.append(env.state_to_pos[state.item()])
            if result.done.item():
                break

    return route[-1] == env.goal, route, total_reward


def evaluate_policy(
    env: StochasticGridWorld10x10,
    Q: torch.Tensor,
    episodes: int = 300,
    max_steps: int = 250,
    seed: int = SEED + 2000,
):
    """在真实随机环境中重复评估最终贪心策略。"""
    generator = torch.Generator(device=env.device).manual_seed(seed)
    success_count = 0
    rewards = []
    lengths = []

    with torch.no_grad():
        for _ in range(episodes):
            state = env.reset()
            total_reward = 0.0

            for step in range(1, max_steps + 1):
                action = Q[state].argmax()
                result = env.step(action, generator=generator, stochastic=True)
                total_reward += result.reward.item()
                state = result.next_state

                if result.done.item():
                    if env.state_to_pos[state.item()] == env.goal:
                        success_count += 1
                    break

            rewards.append(total_reward)
            lengths.append(step)

    return {
        "success_rate": success_count / episodes,
        "mean_reward": sum(rewards) / episodes,
        "mean_steps": sum(lengths) / episodes,
    }


def moving_average(values: torch.Tensor, window: int = 100):
    values = values.detach().float().cpu()
    if values.numel() < window:
        return torch.arange(1, values.numel() + 1), values

    cumsum = torch.cat([torch.zeros(1), values.cumsum(dim=0)])
    smooth = (cumsum[window:] - cumsum[:-window]) / window
    x = torch.arange(window, values.numel() + 1)
    return x, smooth


def save_figure(fig, filename: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / filename, dpi=220, bbox_inches="tight")


def plot_environment(env: StochasticGridWorld10x10) -> None:
    grid = torch.zeros((env.rows, env.cols), dtype=torch.float32)
    for pos in env.walls:
        grid[pos] = 1.0
    for pos in env.traps:
        grid[pos] = 0.65
    grid[env.goal] = 0.35

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(grid.numpy(), cmap="Greys", vmin=0.0, vmax=1.2)

    for pos in env.walls:
        ax.text(pos[1], pos[0], "W", ha="center", va="center")
    for pos in env.traps:
        ax.text(pos[1], pos[0], "X", ha="center", va="center")
    ax.text(env.start[1], env.start[0], "S", ha="center", va="center", fontweight="bold")
    ax.text(env.goal[1], env.goal[0], "G", ha="center", va="center", fontweight="bold")

    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_xticks([x - 0.5 for x in range(1, env.cols)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, env.rows)], minor=True)
    ax.grid(which="minor", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(f"10x10 Stochastic GridWorld (slip={env.slip_prob:.0%})")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.tight_layout()
    save_figure(fig, "00_environment.png")


def plot_training_dashboard(history: dict, window: int = 100) -> None:
    episodes = torch.arange(1, len(history["returns"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1) 回合奖励
    x, smooth = moving_average(history["returns"], window)
    axes[0, 0].plot(episodes.numpy(), history["returns"].numpy(), alpha=0.18)
    axes[0, 0].plot(x.numpy(), smooth.numpy(), linewidth=2)
    axes[0, 0].set_title("Episode Return")
    axes[0, 0].set_ylabel("Return")

    # 2) 成功率
    x, smooth = moving_average(history["successes"].float(), window)
    axes[0, 1].plot(x.numpy(), smooth.numpy(), linewidth=2)
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].set_title(f"Success Rate (moving {window})")
    axes[0, 1].set_ylabel("Success rate")

    # 3) 回合步数
    x, smooth = moving_average(history["lengths"].float(), window)
    axes[1, 0].plot(episodes.numpy(), history["lengths"].numpy(), alpha=0.18)
    axes[1, 0].plot(x.numpy(), smooth.numpy(), linewidth=2)
    axes[1, 0].set_title("Episode Length")
    axes[1, 0].set_ylabel("Steps")

    # 4) ε + 实际探索比例 + 实际打滑比例
    axes[1, 1].plot(episodes.numpy(), history["epsilons"].numpy(), label="epsilon")
    x, explore = moving_average(history["explore_ratios"], window)
    axes[1, 1].plot(x.numpy(), explore.numpy(), label="actual explore ratio")
    x, slip = moving_average(history["slip_ratios"], window)
    axes[1, 1].plot(x.numpy(), slip.numpy(), label="actual slip ratio")
    axes[1, 1].set_ylim(-0.02, 1.02)
    axes[1, 1].set_title("Exploration and Environment Randomness")
    axes[1, 1].set_ylabel("Ratio")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.2)

    fig.suptitle("Q-learning Training Process: 10x10 Stochastic Environment")
    fig.tight_layout()
    save_figure(fig, "01_training_dashboard.png")


def plot_q_table(Q: torch.Tensor, env: StochasticGridWorld10x10) -> None:
    q_cpu = Q.detach().cpu()
    fig, ax = plt.subplots(figsize=(8, 10))
    image = ax.imshow(q_cpu.numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(env.n_actions))
    ax.set_xticklabels([f"a={i}" for i in range(env.n_actions)])
    ax.set_xlabel("Action")
    ax.set_ylabel("State")
    ax.set_title(f"Final Q Table ({env.n_states} states x {env.n_actions} actions)")
    fig.colorbar(image, ax=ax, label="Q(s,a)")
    fig.tight_layout()
    save_figure(fig, "02_q_table.png")


def draw_policy_arrows(ax, env: StochasticGridWorld10x10, q_table: torch.Tensor) -> None:
    q_cpu = q_table.detach().cpu()
    for state, pos in env.state_to_pos.items():
        if pos == env.goal or pos in env.traps:
            continue
        # 从未真正学习过的全零状态不画箭头，避免把 argmax(0,0,0,0)
        # 误解成已经学会了“向上”。
        if q_cpu[state].abs().max().item() < 1e-10:
            continue
        action = int(q_cpu[state].argmax().item())
        dr, dc = env.DELTAS[action]
        ax.arrow(
            pos[1], pos[0], dc * 0.24, dr * 0.24,
            head_width=0.10, head_length=0.08,
            length_includes_head=True, linewidth=1.1,
        )


def plot_final_policy(
    env: StochasticGridWorld10x10,
    Q: torch.Tensor,
    route: list[tuple[int, int]],
) -> None:
    grid = torch.zeros((env.rows, env.cols), dtype=torch.float32)
    for pos in env.walls:
        grid[pos] = 1.0
    for pos in env.traps:
        grid[pos] = 0.65
    grid[env.goal] = 0.35

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.imshow(grid.numpy(), cmap="Greys", vmin=0.0, vmax=1.2)
    draw_policy_arrows(ax, env, Q)

    xs = [p[1] for p in route]
    ys = [p[0] for p in route]
    ax.plot(xs, ys, marker="o", linewidth=2.5, label="nominal greedy route")

    for pos in env.walls:
        ax.text(pos[1], pos[0], "W", ha="center", va="center")
    for pos in env.traps:
        ax.text(pos[1], pos[0], "X", ha="center", va="center")
    ax.text(env.start[1], env.start[0], "S", ha="center", va="center", fontweight="bold")
    ax.text(env.goal[1], env.goal[0], "G", ha="center", va="center", fontweight="bold")

    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_xticks([x - 0.5 for x in range(1, env.cols)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, env.rows)], minor=True)
    ax.grid(which="minor", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Final Greedy Policy + Nominal Route")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "03_final_policy_route.png")


def plot_policy_evolution(history: dict, env: StochasticGridWorld10x10) -> None:
    snapshots = history["snapshots"]
    points = sorted(snapshots)

    fig, axes = plt.subplots(
        1, len(points), figsize=(3.7 * len(points), 4.2), layout="constrained"
    )
    if len(points) == 1:
        axes = [axes]

    all_values = torch.cat([q.max(dim=1).values for q in snapshots.values()])
    vmin = float(all_values.min().item())
    vmax = float(all_values.max().item())

    for ax, episode in zip(axes, points):
        q = snapshots[episode]
        values = torch.full((env.rows, env.cols), torch.nan)
        for state, pos in env.state_to_pos.items():
            values[pos] = q[state].max()

        image = ax.imshow(values.numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
        draw_policy_arrows(ax, env, q)
        for pos in env.walls:
            ax.text(pos[1], pos[0], "W", ha="center", va="center")
        for pos in env.traps:
            ax.text(pos[1], pos[0], "X", ha="center", va="center")
        ax.text(env.start[1], env.start[0], "S", ha="center", va="center")
        ax.text(env.goal[1], env.goal[0], "G", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Episode {episode}")

    fig.colorbar(image, ax=list(axes), shrink=0.72, pad=0.02, label="max Q(s,a)")
    fig.suptitle("How the Policy Emerges During Training")
    save_figure(fig, "04_policy_evolution.png")


def visualize_results(
    history: dict,
    Q: torch.Tensor,
    env: StochasticGridWorld10x10,
    nominal_route: list[tuple[int, int]],
    show: bool = True,
) -> None:
    plot_environment(env)
    plot_training_dashboard(history)
    plot_q_table(Q, env)
    plot_final_policy(env, Q, nominal_route)
    plot_policy_evolution(history, env)

    print(f"可视化图片已保存到: {FIG_DIR.resolve()}")
    if show:
        plt.show()
    else:
        plt.close("all")


def print_stage_summary(history: dict) -> None:
    n = len(history["returns"])
    block = min(300, max(50, n // 10))

    stages = {
        "训练早期": slice(0, block),
        "训练中期": slice(max(0, n // 2 - block // 2), min(n, n // 2 + block // 2)),
        "训练后期": slice(max(0, n - block), n),
    }

    print("\n=== 训练阶段对比 ===")
    for name, s in stages.items():
        print(
            f"{name}: "
            f"平均奖励={history['returns'][s].mean().item():7.2f} | "
            f"成功率={history['successes'][s].float().mean().item():6.1%} | "
            f"平均步数={history['lengths'][s].float().mean().item():6.1f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="10x10 stochastic tabular Q-learning with CUDA and visualization"
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="auto=有 CUDA 就用 CUDA，否则 CPU",
    )
    parser.add_argument("--episodes", type=int, default=2500, help="训练回合数")
    parser.add_argument(
        "--slip-prob", type=float, default=0.20,
        help="动作随机偏移总概率，例如 0.20 = 10%% 左偏 + 10%% 右偏",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=200,
        help="训练后在随机环境中重复评估的回合数",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="只保存图片，不弹出 Matplotlib 窗口",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(SEED)
    device = choose_device(args.device)
    print_device_info(device)

    env = StochasticGridWorld10x10(device, slip_prob=args.slip_prob)
    print("\n=== 环境信息 ===")
    print(f"网格规模: {env.rows} x {env.cols}")
    print(f"有效状态数: {env.n_states}")
    print(f"动作数: {env.n_actions}")
    print(f"墙数量: {len(env.walls)}")
    print(f"陷阱数量: {len(env.traps)}")
    print(f"动作随机偏移概率: {env.slip_prob:.0%}")
    print(
        f"动作执行规则: {1-env.slip_prob:.0%} 原动作 + "
        f"{env.slip_prob/2:.0%} 左偏 + {env.slip_prob/2:.0%} 右偏"
    )

    Q, history = train_q_learning(env, episodes=args.episodes)

    # 名义路线：关闭环境随机性，展示“学到的策略希望怎么走”。
    nominal_success, nominal_route, nominal_reward = greedy_rollout(
        StochasticGridWorld10x10(device, slip_prob=args.slip_prob),
        Q,
        stochastic=False,
    )

    # 真实测试：保留 20% 动作随机偏移，多次重复评估。
    eval_stats = evaluate_policy(
        StochasticGridWorld10x10(device, slip_prob=args.slip_prob),
        Q,
        episodes=args.eval_episodes,
    )

    print("\n=== Q-learning 训练完成 ===")
    print(f"Q 表形状: {tuple(Q.shape)}")
    print(f"Q 表所在设备: {Q.device}")
    print(f"最后100回合平均奖励: {history['returns'][-100:].mean().item():.2f}")
    print(f"最后100回合成功率: {history['successes'][-100:].float().mean().item():.1%}")
    print(f"最后100回合平均步数: {history['lengths'][-100:].float().mean().item():.1f}")
    print_stage_summary(history)

    print("\n=== 最终策略：无随机偏移的名义路线 ===")
    print(f"是否到达终点: {nominal_success}")
    print(f"名义路线步数: {len(nominal_route)-1}")
    print(f"名义路线奖励: {nominal_reward:.2f}")
    print(f"路线: {nominal_route}")

    print(f"\n=== 最终策略：随机环境重复测试 {args.eval_episodes} 回合 ===")
    print(f"成功率: {eval_stats['success_rate']:.1%}")
    print(f"平均奖励: {eval_stats['mean_reward']:.2f}")
    print(f"平均步数: {eval_stats['mean_steps']:.1f}")

    visualize_results(
        history=history,
        Q=Q,
        env=env,
        nominal_route=nominal_route,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
