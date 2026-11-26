"""Vectorized Recurrent PPO training pipeline for DoomSpatialPolicy.

Supports both single-process (SyncVecEnv) and multi-worker (SubprocVecEnv) rollouts,
truncated BPTT over (gru_h, spatial_map, prev_action), GAE, auxiliary vitals prediction,
and curriculum progression across procedural Tier 1-4 maps.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from submission.model import (
    DoomSpatialPolicy,
    ModelConfig,
    eval_factor_logprob_entropy,
    normalize_vitals,
    preprocess_obs_dict,
    sample_factor_actions,
)
from training.env_wrapper import ShapedDoomEnv, curriculum_map_entries

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "submission" / "weights" / "model.pt"


def _worker(conn, map_entries: list[tuple[str, str]], episode_ticks: int, seed: int):
    env = ShapedDoomEnv(map_entries, episode_ticks=episode_ticks, seed=seed)
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                obs, info = env.reset(seed=data)
                conn.send((obs, info))
            elif cmd == "step":
                obs, rew, term, trunc, info = env.step(data)
                done = term or trunc
                if done:
                    final_info = dict(info)
                    obs, _ = env.reset()
                    final_info["terminal_done"] = True
                    conn.send((obs, rew, True, final_info))
                else:
                    conn.send((obs, rew, False, info))
            elif cmd == "close":
                env.close()
                conn.close()
                break
    except EOFError:
        env.close()


class VecDoomEnv:
    """Lightweight vectorized DOOM runner (same-step autoreset, zero-copy obs dict stacking)."""

    def __init__(self, num_envs: int, map_entries: list[tuple[str, str]], episode_ticks: int, seed: int = 0, use_subproc: bool = True):
        self.num_envs = num_envs
        self.use_subproc = use_subproc and num_envs > 1
        if self.use_subproc:
            ctx = mp.get_context("spawn")
            self.conns = []
            self.procs = []
            for i in range(num_envs):
                parent_conn, child_conn = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(child_conn, map_entries, episode_ticks, seed + i * 997), daemon=True)
                p.start()
                child_conn.close()
                self.conns.append(parent_conn)
                self.procs.append(p)
        else:
            self.envs = [ShapedDoomEnv(map_entries, episode_ticks=episode_ticks, seed=seed + i * 997) for i in range(num_envs)]

    @staticmethod
    def _stack_obs(obs_list: list[dict]) -> dict:
        return {
            "rgb": np.stack([o["rgb"] for o in obs_list], axis=0),
            "vitals": np.stack([o["vitals"] for o in obs_list], axis=0),
            "rays": {
                "distance": np.stack([o["rays"]["distance"] for o in obs_list], axis=0),
                "hit_type": np.stack([o["rays"]["hit_type"] for o in obs_list], axis=0),
            },
            "visible_entities": {
                "features": np.stack([o["visible_entities"]["features"] for o in obs_list], axis=0),
                "entity_type": np.stack([o["visible_entities"]["entity_type"] for o in obs_list], axis=0),
                "mask": np.stack([o["visible_entities"]["mask"] for o in obs_list], axis=0),
            },
        }

    def reset(self, seed: int = 0) -> dict:
        if self.use_subproc:
            for i, c in enumerate(self.conns):
                c.send(("reset", seed + i))
            res = [c.recv() for c in self.conns]
            return self._stack_obs([r[0] for r in res])
        return self._stack_obs([e.reset(seed=seed + i)[0] for i, e in enumerate(self.envs)])

    def step(self, actions: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
        if self.use_subproc:
            for c, a in zip(self.conns, actions):
                c.send(("step", a))
            res = [c.recv() for c in self.conns]
            obs = self._stack_obs([r[0] for r in res])
            rews = np.array([r[1] for r in res], dtype=np.float32)
            dones = np.array([r[2] for r in res], dtype=bool)
            infos = [r[3] for r in res]
            return obs, rews, dones, infos

        obs_list, rews, dones, infos = [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, term, trunc, info = e.step(a)
            done = term or trunc
            if done:
                o, _ = e.reset()
            obs_list.append(o)
            rews.append(r)
            dones.append(done)
            infos.append(info)
        return self._stack_obs(obs_list), np.array(rews, dtype=np.float32), np.array(dones, dtype=bool), infos

    def close(self):
        if self.use_subproc:
            for c in self.conns:
                try:
                    c.send(("close", None))
                except OSError:
                    pass
            for p in self.procs:
                p.join(timeout=2)
        else:
            for e in self.envs:
                e.close()


def mask_state(state: tuple[torch.Tensor, torch.Tensor, torch.Tensor], done_mask: torch.Tensor):
    """Zero recurrent state for environments where an episode just finished (`done_mask` = 1.0)."""
    keep = (1.0 - done_mask).to(state[0].dtype)
    gru_h = state[0] * keep.unsqueeze(-1)
    spatial_map = state[1] * keep.view(-1, 1, 1, 1)
    prev_action = state[2] * keep.unsqueeze(-1)
    return gru_h, spatial_map, prev_action


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """rewards, values, dones: (T, N); next_value: (N,)."""
    t_steps, n_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(n_envs, device=rewards.device)
    for t in reversed(range(t_steps)):
        nv = next_value if t == t_steps - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def save_checkpoint(path: Path, policy: DoomSpatialPolicy, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": policy.state_dict(),
            "config": asdict(policy.cfg),
            "step": step,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu" and args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    print(f"Training on device={device}, stage={args.stage}, num_envs={args.num_envs}, total_steps={args.total_steps}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = ModelConfig()
    policy = DoomSpatialPolicy(cfg).to(device)
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        policy.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
        print(f"Resumed weights from {args.resume}")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, eps=1e-5, weight_decay=1e-4)

    map_entries = curriculum_map_entries(args.stage, map_cache_dir=args.map_dir, maps_per_tier=args.maps_per_tier)
    vec = VecDoomEnv(
        num_envs=args.num_envs,
        map_entries=map_entries,
        episode_ticks=args.episode_ticks,
        seed=args.seed,
        use_subproc=not args.no_subproc,
    )

    obs_np = vec.reset(seed=args.seed)
    state = policy.init_state(args.num_envs, device)

    num_updates = max(1, args.total_steps // (args.rollout_len * args.num_envs))
    t_start = time.perf_counter()
    ep_returns = np.zeros(args.num_envs, dtype=np.float32)
    ep_lens = np.zeros(args.num_envs, dtype=np.int32)
    recent_rets: list[float] = []
    recent_exits: list[float] = []
    recent_kills: list[float] = []
    recent_keys: list[float] = []
    recent_doors: list[float] = []
    recent_cells: list[float] = []
    best_ret = -1e9

    try:
        for update in range(1, num_updates + 1):
            policy.eval()
            roll_obs: list[dict[str, torch.Tensor]] = []
            roll_factor_actions = []
            roll_mb_actions = []
            roll_logprobs = []
            roll_values = []
            roll_rewards = []
            roll_dones = []
            roll_next_vitals = []

            roll_prev_actions = []
            init_state = tuple(s.detach().clone() for s in state)

            for _ in range(args.rollout_len):
                obs_t = preprocess_obs_dict(obs_np, device)
                roll_obs.append({k: v.cpu() for k, v in obs_t.items()})
                roll_prev_actions.append(state[2].cpu())

                with torch.no_grad():
                    logits, value, _, next_state = policy.forward_step(obs_t, state)
                    factor_act, action, logprob = sample_factor_actions(logits, obs_t["vitals"], deterministic=False)

                act_np = action.cpu().numpy().astype(np.int8)
                next_obs_np, rews_np, dones_np, infos = vec.step(act_np)

                next_v = torch.from_numpy(np.ascontiguousarray(next_obs_np["vitals"], dtype=np.float32))
                roll_next_vitals.append(next_v)
                roll_factor_actions.append(factor_act.cpu())
                roll_mb_actions.append(action.cpu())
                roll_logprobs.append(logprob.cpu())
                roll_values.append(value.cpu())
                roll_rewards.append(torch.from_numpy(rews_np))
                roll_dones.append(torch.from_numpy(dones_np.astype(np.float32)))

                ep_returns += rews_np
                ep_lens += 1
                for i, d in enumerate(dones_np):
                    if d:
                        recent_rets.append(float(ep_returns[i]))
                        recent_exits.append(1.0 if infos[i].get("exit_reached") else 0.0)
                        recent_kills.append(float(infos[i].get("kills", 0)))
                        recent_keys.append(float(infos[i].get("keys_held", 0)))
                        recent_doors.append(float(infos[i].get("doors_opened", 0)))
                        recent_cells.append(float(infos[i].get("cells_visited", 0)))
                        ep_returns[i] = 0.0
                        ep_lens[i] = 0

                done_t = torch.from_numpy(dones_np.astype(np.float32)).to(device)
                gru_h, s_map, _ = next_state
                state = mask_state((gru_h, s_map, action), done_t)
                obs_np = next_obs_np

            # Bootstrap last value
            with torch.no_grad():
                last_obs_t = preprocess_obs_dict(obs_np, device)
                _, next_value, _, _ = policy.forward_step(last_obs_t, state)
                next_value = next_value.cpu()

            rewards_t = torch.stack(roll_rewards)  # (T, N)
            values_t = torch.stack(roll_values)
            dones_t = torch.stack(roll_dones)
            factor_actions_t = torch.stack(roll_factor_actions)
            mb_actions_t = torch.stack(roll_mb_actions)
            prev_actions_t = torch.stack(roll_prev_actions)
            old_logprobs_t = torch.stack(roll_logprobs)
            next_vitals_t = torch.stack(roll_next_vitals)
            obs_stacked = {k: torch.stack([step[k] for step in roll_obs], dim=0) for k in roll_obs[0]}

            adv_t, ret_t = compute_gae(rewards_t, values_t, dones_t, next_value, gamma=args.gamma, gae_lambda=args.gae_lambda)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            # Linear entropy annealing down to 45% of ent_coef to avoid late-stage collapse
            frac = 1.0 - (update - 1) / max(1, num_updates)
            cur_ent_coef = args.ent_coef * (0.45 + 0.55 * frac)
            warmup = min(1.0, update / 12.0) if args.resume else 1.0
            cur_lr = args.lr * warmup * (0.35 + 0.65 * frac)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            # PPO update over environment minibatches: vectorize encoders across T*M, loop only recurrence
            policy.train()
            env_perm = np.arange(args.num_envs)
            mb_envs = max(1, args.num_envs // args.num_minibatches)

            pg_losses, vf_losses, ent_losses, aux_losses = [], [], [], []
            stop_early = False
            for _ in range(args.ppo_epochs):
                if stop_early:
                    break
                np.random.shuffle(env_perm)
                for start in range(0, args.num_envs, mb_envs):
                    idx = env_perm[start : start + mb_envs]
                    idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
                    t_len, m_len = args.rollout_len, len(idx)

                    flat_obs = {k: v[:, idx].reshape(t_len * m_len, *v.shape[2:]).to(device) for k, v in obs_stacked.items()}
                    flat_prev_act = prev_actions_t[:, idx].reshape(t_len * m_len, -1).to(device)
                    r_all, v_all, e_all, vis_all, s_all = policy.encode_inputs(flat_obs, flat_prev_act)
                    r_seq = r_all.view(t_len, m_len, -1)
                    v_seq = v_all.view(t_len, m_len, -1)
                    e_seq = e_all.view(t_len, m_len, -1)
                    vis_seq = vis_all.view(t_len, m_len, -1)
                    s_seq = s_all.view(t_len, m_len, *s_all.shape[1:])
                    vit_seq = obs_stacked["vitals"][:, idx].to(device)

                    mb_state = tuple(s[idx_t].clone() for s in init_state)
                    logits_seq, vals_seq, aux_seq = [], [], []
                    for t in range(t_len):
                        lg, vl, ax, next_mb_state = policy.recurrent_step(
                            r_seq[t], v_seq[t], e_seq[t], vis_seq[t], s_seq[t], vit_seq[t], mb_state
                        )
                        logits_seq.append(lg)
                        vals_seq.append(vl)
                        aux_seq.append(ax)
                        done_step = dones_t[t, idx].to(device)
                        act_step = mb_actions_t[t, idx].to(device)
                        mb_state = mask_state((next_mb_state[0], next_mb_state[1], act_step), done_step)

                    lg_all = torch.stack(logits_seq)  # (T, M, 12)
                    vl_all = torch.stack(vals_seq)  # (T, M)
                    ax_all = torch.stack(aux_seq)  # (T, M, 17)

                    mb_fact = factor_actions_t[:, idx].to(device)
                    new_lp, entropy_t = eval_factor_logprob_entropy(lg_all, mb_fact, vit_seq)
                    entropy = entropy_t.mean()

                    mb_old_lp = old_logprobs_t[:, idx].to(device)
                    mb_old_val = values_t[:, idx].to(device)
                    mb_adv = adv_t[:, idx].to(device)
                    mb_ret = ret_t[:, idx].to(device)

                    logratio = new_lp - mb_old_lp
                    ratio = torch.exp(logratio)
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - logratio).mean().item()
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * mb_adv
                    pg_loss = -torch.min(surr1, surr2).mean()
                    v_clipped = mb_old_val + torch.clamp(vl_all - mb_old_val, -args.clip_coef, args.clip_coef)
                    vf_loss = 0.5 * torch.max((vl_all - mb_ret).pow(2), (v_clipped - mb_ret).pow(2)).mean()

                    # Auxiliary loss only on non-terminal transitions
                    target_v = normalize_vitals(next_vitals_t[:, idx].to(device))
                    nonterm = (1.0 - dones_t[:, idx].to(device)).unsqueeze(-1)
                    aux_loss = (F.mse_loss(ax_all, target_v, reduction="none") * nonterm).mean()

                    loss = pg_loss + args.vf_coef * vf_loss - cur_ent_coef * entropy + 0.1 * aux_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                    optimizer.step()

                    pg_losses.append(pg_loss.item())
                    vf_losses.append(vf_loss.item())
                    ent_losses.append(entropy.item())
                    aux_losses.append(aux_loss.item())
                    if args.target_kl > 0 and approx_kl > args.target_kl:
                        stop_early = True
                        break

            global_step = update * args.rollout_len * args.num_envs
            out_path = Path(args.out)
            if update in (num_updates // 3, (2 * num_updates) // 3):
                snap_tag = "s1" if update == num_updates // 3 else "s2"
                save_checkpoint(out_path.with_name(f"{out_path.stem}_{snap_tag}{out_path.suffix}"), policy, global_step)
            if update % args.log_every == 0 or update == num_updates:
                elapsed = time.perf_counter() - t_start
                sps = global_step / max(1e-6, elapsed)
                mean_ret = float(np.mean(recent_rets[-40:])) if recent_rets else float(rewards_t.sum(dim=0).mean())
                exit_rate = float(np.mean(recent_exits[-40:])) if recent_exits else 0.0
                mean_kills = float(np.mean(recent_kills[-40:])) if recent_kills else 0.0
                mean_keys = float(np.mean(recent_keys[-40:])) if recent_keys else 0.0
                mean_doors = float(np.mean(recent_doors[-40:])) if recent_doors else 0.0
                mean_cells = float(np.mean(recent_cells[-40:])) if recent_cells else 0.0
                mean_ent = float(np.mean(ent_losses))
                print(
                    f"[upd {update}/{num_updates} | steps={global_step} | {sps:.0f} sps] "
                    f"ret={mean_ret:.2f} exit={exit_rate:.2f} kills={mean_kills:.2f} keys={mean_keys:.2f} doors={mean_doors:.2f} cells={mean_cells:.1f} "
                    f"pg={np.mean(pg_losses):.3f} vf={np.mean(vf_losses):.3f} ent={mean_ent:.2f}"
                )
                if len(recent_rets) >= 40 and mean_ent >= 1.2 and mean_ret >= best_ret:
                    best_ret = mean_ret
                    save_checkpoint(out_path, policy, global_step)

        final_step = num_updates * args.rollout_len * args.num_envs
        last_path = out_path.with_name(f"{out_path.stem}_last{out_path.suffix}")
        save_checkpoint(last_path, policy, final_step)
        if best_ret < -1e8 or not out_path.is_file():
            save_checkpoint(out_path, policy, final_step)
        print(f"Best checkpoint saved to {out_path} (best_ret={best_ret:.2f}), final saved to {last_path}")
    finally:
        vec.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DoomSpatialPolicy with Recurrent PPO")
    p.add_argument("--stage", type=int, default=2, choices=[1, 2, 3, 4], help="Curriculum stage 1..4")
    p.add_argument("--total-steps", type=int, default=50_000, help="Total environment decision steps")
    p.add_argument("--num-envs", type=int, default=4, help="Parallel DOOM environments")
    p.add_argument("--rollout-len", type=int, default=128, help="Rollout length / BPTT horizon")
    p.add_argument("--episode-ticks", type=int, default=1400, help="Ticks per episode during training")
    p.add_argument("--maps-per-tier", type=int, default=16, help="Procedural maps generated per tier")
    p.add_argument("--map-dir", type=str, default="/tmp/deadlock_maps", help="Directory for cached procedural WADs")
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.012)
    p.add_argument("--target-kl", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--cpu-threads", type=int, default=4, help="PyTorch intra-op CPU threads (4 is optimal on M4)")
    p.add_argument("--no-subproc", action="store_true", help="Run environments synchronously in main process")
    p.add_argument("--resume", type=str, default="", help="Optional checkpoint to resume/fine-tune from")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Output checkpoint path")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
