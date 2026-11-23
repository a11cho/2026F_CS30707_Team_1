"""관측 면적과 전이 후 점령 기여에서 학습 보상을 계산합니다."""


def calculate_learning_rewards(before, after, infos, teams, config):
    rewards, components = {}, {}
    for agents in teams.values():
        first = agents[0]
        new = after[first]['areas']
        signal=config.get('signal','gap_delta')
        if signal=='area_share':
            total=int(new[0])+int(new[1])
            if total<=0:
                raise ValueError('전체 맵 면적은 양수여야 합니다.')
            team_reward=config['team_weight']*int(new[0])/total
            component='team_area'
        elif signal=='gap_delta':
            old=before[first]['areas']
            gap_change=(int(new[0])-int(new[1]))-(int(old[0])-int(old[1]))
            team_reward=config['team_weight']*gap_change/len(agents)
            component='team_delta'
        else:
            raise ValueError(f'지원하지 않는 보상 신호: {signal}')
        for agent in agents:
            credit = float(infos[agent]['capture_credit'])
            capture = config['capture_weight'] * credit if config['mode'] == 'A-S' else 0.0
            components[agent] = {component:team_reward,'capture':capture}
            rewards[agent] = team_reward + capture
    return rewards, components
