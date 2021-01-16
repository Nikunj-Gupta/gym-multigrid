import gym
import time
from gym.envs.registration import register
import argparse 
from gym_multigrid.envs.collect_game import CollectGameEnv 


parser = argparse.ArgumentParser(description=None)
parser.add_argument('-s', '--size', default=10, type=int, help="size of grid") 
parser.add_argument('-b', '--balls', nargs="*", default=[5]*3, type=int) 
parser.add_argument('-n', '--nagents', default=3, type=int) 
args = parser.parse_args() 



class MyCollectGame(CollectGameEnv):

    def __init__(self):
        super().__init__(size=args.size, 
        num_balls=args.balls,
        agents_index=list(range(args.nagents)),
        balls_index=list(range(len(args.balls))),
        balls_reward=[1]*len(args.balls),
        zero_sum=True)

def main():
    register(
        id='multigrid-collect-v0',
        entry_point='test_collect_env:MyCollectGame',
    )
    env = gym.make('multigrid-collect-v0') 
    env.reset()
    nb_agents = len(env.agents)
    count=1 
    while True:
        print(count)
        count+=1 
        env.render()
        actions = [env.action_space.sample() for _ in range(nb_agents)]
        obs, rew, done, _ = env.step(actions)
        if done:
            break

if __name__ == "__main__":

    main()