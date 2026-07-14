"""
Evaluation harness for RL environments.

Standardized across all algorithms so they are compared fairly:
- Each genome is evaluated on a fixed set of seeds per generation
  (we re-evaluate elites each gen to reduce noise).
- We support discrete action spaces (CartPole, MountainCar, Acrobot)
  by argmax over output dim.
- Reward is normalized to [0,1]-ish by env's known bounds for fair
  fitness comparison across environments.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from typing import List, Optional, Callable
from .genome import Genome


# Cached env metadata: max_episode_steps, reward_bounds
_ENV_INFO = {
    'CartPole-v1':       {'max_steps': 500,  'r_min': 0.0, 'r_max': 500.0, 'discrete': True},
    'CartPole-v0':       {'max_steps': 200,  'r_min': 0.0, 'r_max': 200.0, 'discrete': True},
    'MountainCar-v0':    {'max_steps': 200,  'r_min': -200.0, 'r_max': 0.0, 'discrete': True},
    'Acrobot-v1':        {'max_steps': 500,  'r_min': -500.0, 'r_max': 0.0, 'discrete': True},
    'LunarLander-v2':    {'max_steps': 1000, 'r_min': -300.0, 'r_max': 300.0, 'discrete': True},
}


def get_env_info(name: str) -> dict:
    if name not in _ENV_INFO:
        raise ValueError(f"Unknown env: {name}. Add it to _ENV_INFO.")
    return _ENV_INFO[name]


def make_env(name: str):
    return gym.make(name)


def eval_genome(g: Genome, env_name: str, seed: int,
                action_fn: Optional[Callable] = None,
                max_steps: Optional[int] = None,
                collect_states: bool = False) -> float | tuple:
    """Evaluate a single genome on a single episode.
    Returns raw episode reward (or (reward, states) if collect_states=True).
    """
    info = get_env_info(env_name)
    if max_steps is None:
        max_steps = info['max_steps']
    env = gym.make(env_name)
    try:
        obs, _ = env.reset(seed=seed)
        total = 0.0
        states = [np.asarray(obs, dtype=np.float32)] if collect_states else None
        for _ in range(max_steps):
            out = g.forward(list(obs))
            if info['discrete']:
                action = int(np.argmax(out))
            else:
                action = np.array(out, dtype=np.float32)
                # Clip to env bounds (assumes Box with finite bounds)
                if hasattr(env.action_space, 'low'):
                    action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            if collect_states:
                states.append(np.asarray(obs, dtype=np.float32))
            if term or trunc:
                break
        if collect_states:
            return total, np.array(states)
        return total
    finally:
        env.close()


def evaluate_population(pop: List[Genome], env_name: str,
                        seeds: List[int]) -> List[float]:
    """Evaluate every genome on every seed; fitness = mean over seeds.
    Returns the list of mean rewards aligned with `pop`.
    """
    info = get_env_info(env_name)
    fits = []
    for g in pop:
        rs = [eval_genome(g, env_name, s) for s in seeds]
        fits.append(float(np.mean(rs)))
    return fits


def normalize_reward(r: float, env_name: str) -> float:
    """Map raw reward to [0, 1] using env bounds."""
    info = get_env_info(env_name)
    return (r - info['r_min']) / (info['r_max'] - info['r_min'])
