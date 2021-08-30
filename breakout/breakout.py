from agent import *
import pandas as pd

env = gym.make('Breakout-ram-v0')
agent = Agent(env)


for i in range(10000):
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        env.render()
        state, reward, done = agent.take_action(state)
        score += reward
    print(f'Episode:{i} Score:{score}')
    if ((i + 1) % 100 == 0):
        parameters = agent.q_values.parameters()
        cnt = 0
        for parameter in parameters:
            val = pd.DataFrame(parameter.detach().numpy())
            val.to_csv(f'{cnt}.csv')
            cnt += 1

env.close()
