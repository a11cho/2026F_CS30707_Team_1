"""순차 공개 범위·좌우 방향·critic 입력과 양 팀의 PPO 갱신을 검사합니다."""
import torch
from model import Action, MESSAGE, SequentialPolicy, arena_for, play_tick, ppo_update, reflect


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    game = dict(height=3, width=10, max_ticks=4, respawn_delay=0)
    opt = dict(hidden=128, entity_hidden=32, alpha=1, gamma=.99, gae_lambda=.95,
               epochs=1, batch_size=32, clip=.2, entropy=.01, value_weight=.5, max_grad_norm=.5)
    policies = [SequentialPolicy(game,opt) for _ in range(2)]
    arena = arena_for(game,7)
    records, rewards, step = play_tick(arena, policies, 1)
    assert step.tick == 1
    for side, (board, context, peers, mask, acts, old, values) in enumerate(records):
        slots = context[:,9:].reshape(3,3,MESSAGE)
        assert torch.equal(context[:,6:9], torch.eye(3))
        for i in range(3):
            assert not slots[i,i:].any()
            for j in range(i):
                assert slots[i,j,:len(Action)].sum() == 1
                assert slots[i,j,acts[j]] == 1
            aid = arena.teams[side][i]
            applied = arena.env.last_decision[-1]["applied"][aid]
            assert int(reflect(Action.from_index(acts[i].item()),side)) == applied
            assert rewards[side][i] == step.area_after[side]/30 + step.credits[aid]
        dist, v = policies[side](board,context,peers,mask)
        torch.testing.assert_close(old, dist.log_prob(acts))
        torch.testing.assert_close(values,v)
        blank = context.clone()
        blank[:,6:] = 0
        torch.testing.assert_close(values,policies[side](board,blank,peers,mask)[1])
    hidden, _, _ = play_tick(arena,policies,1,greedy=True,hide_actions=True)
    assert all(not r[1][:,9:].any() for r in hidden)
    optims = [torch.optim.Adam(p.parameters(),lr=.0003) for p in policies]
    before = [[p.actor[0].weight.detach().clone(),p.critic[0].weight.detach().clone()] for p in policies]
    metrics = ppo_update(game,[8,9],policies,optims,opt)
    for p,(a,c),m in zip(policies,before,metrics):
        assert not torch.equal(a,p.actor[0].weight) and not torch.equal(c,p.critic[0].weight)
        assert all(torch.isfinite(torch.tensor(v)) for v in m.values())
    print("OK: sequential 3v3 PPO")


if __name__ == "__main__":
    main()
