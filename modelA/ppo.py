"""마스크를 유지하며 공유 Actor·Critic에 PPO-Clip 업데이트를 적용합니다."""

import torch
from torch.nn import functional as F

from .networks import masked_distribution


def ppo_update(actor, critic, optimizer, buffer, config, device):
    batch = buffer.tensors(device)
    size = len(batch["actions"])
    alive = batch["alive"]
    advantages = batch["advantage"]
    if alive.any():
        active = advantages[alive]
        advantages = (advantages - active.mean()) / (active.std(unbiased=False) + 1e-8)
    totals = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0,
                  approx_kl=0.0, clip_fraction=0.0)
    policy_count = value_count = 0
    clip = config["clip_ratio"]
    parameters = list(actor.parameters()) + list(critic.parameters())
    for _ in range(config["epochs"]):
        indices = torch.randperm(size, device=device)
        for ids in indices.split(config["minibatch_size"]):
            value_loss = F.mse_loss(critic(batch["features"][ids]), batch["target"][ids])
            loss = config["value_coef"] * value_loss
            policy_ids = ids[alive[ids]]
            if len(policy_ids):
                dist = masked_distribution(actor(batch["features"][policy_ids]),
                                           batch["masks"][policy_ids])
                log_ratio = dist.log_prob(batch["actions"][policy_ids]) - batch["old_log_prob"][policy_ids]
                ratio = log_ratio.exp()
                adv = advantages[policy_ids]
                policy_loss = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
                entropy = dist.entropy().mean()
                loss = loss + policy_loss - config["entropy_coef"] * entropy
                with torch.no_grad():
                    count = len(policy_ids)
                    policy_count += count
                    totals["policy_loss"] += float(policy_loss) * count
                    totals["entropy"] += float(entropy) * count
                    totals["approx_kl"] += float((ratio - 1 - log_ratio).mean()) * count
                    totals["clip_fraction"] += float(((ratio - 1).abs() > clip).float().mean()) * count
            if not torch.isfinite(loss):
                raise FloatingPointError("PPO 손실이 유한하지 않습니다. 보상 크기와 학습률을 확인하세요.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            value_count += len(ids)
            totals["value_loss"] += float(value_loss.detach()) * len(ids)
    return {name: total / max(1, value_count if name == "value_loss" else policy_count)
            for name, total in totals.items()}
