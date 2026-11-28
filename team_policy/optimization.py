from collections import Counter
from time import perf_counter

import torch
from torch.nn.utils import clip_grad_norm_


def batches(items, size):
    for offset in range(0, len(items), size):
        yield items[offset:offset + size]


def synchronize(policy):
    if policy.device.type == "cuda":
        torch.cuda.synchronize(policy.device)


def clipped_objective(ratio, advantage, upper=1.2):
    return torch.minimum(ratio * advantage, ratio.clamp(0.8, upper) * advantage)


def generalized_advantage(rewards, values, terminated, bootstrap, gamma, gae_lambda):
    advantages = torch.empty_like(values)
    future_advantage = torch.zeros_like(bootstrap)
    next_value = bootstrap
    for tick in reversed(range(len(rewards))):
        continuation = ~terminated[tick]
        residual = rewards[tick] + gamma * next_value * continuation - values[tick]
        future_advantage = residual + gamma * gae_lambda * future_advantage * continuation
        advantages[tick] = future_advantage
        next_value = values[tick]
    return advantages


def optimize(policy, optimizer, size, args, rng, log, backward, name, *, lora_scheduler=None):
    """Retry an effective batch with smaller chunks when CUDA cannot hold activations."""
    steps = 0
    first_step_seconds = 0.0
    microbatch = args.microbatch
    sums = Counter()
    start = perf_counter()
    for epoch in range(args.epochs):
        for indices in batches(rng.permutation(size), args.batch_size):
            step_start = perf_counter()
            while True:
                optimizer.zero_grad(set_to_none=True)
                try:
                    batch_stats = backward(indices, microbatch)
                except torch.cuda.OutOfMemoryError:
                    if policy.device.type != "cuda" or microbatch == 1:
                        raise
                else:
                    sums.update(batch_stats)
                    break
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                microbatch = max(1, microbatch // 2)
                log.write("microbatch_reduced", microbatch=microbatch, optimizer_steps=steps)
            grad_norm = clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if lora_scheduler is not None:
                step = lora_scheduler.last_epoch + 1
                if step <= args.lora_lr_warmup_steps:
                    log.write("lora_lr_warmup_step", optimizer_step=step, lora_lr=optimizer.param_groups[-1]["lr"])
                lora_scheduler.step()
            steps += 1
            sums["gradient_norm"] += grad_norm.item()
            if steps == 1:
                synchronize(policy)
                first_step_seconds = perf_counter() - step_start
                log.write(f"first_{name}_step", seconds=first_step_seconds, gradient_norm=grad_norm.item())
        log.write(f"{name}_epoch", epoch=epoch + 1, optimizer_steps=steps, seconds=perf_counter() - start)
    synchronize(policy)
    return sums, {"optimizer_steps": steps, "first_step_seconds": first_step_seconds,
                  "seconds": perf_counter() - start, "gradient_norm": sums["gradient_norm"] / max(1, steps),
                  "microbatch": microbatch}
