import logging
import numpy as np
from dmmath.math_env.math_env import DMMathEnv
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='perform random search')
parser.add_argument('--data', type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    logging.basicConfig(filename=None, filemode='w', level=logging.INFO)

    env_kwargs = {
        'data_path': args.data,
        'ops_typ': "search_basic",
        # 'ops_typ': "all",
    }
    env = DMMathEnv(**env_kwargs)

    MAX_TRY_NUM = 100000
    search_dict = defaultdict(lambda: [])
    while True:
        obs = env.reset()
        problem = env.problem
        # print(problem.question)
        # gt_actions = problem.parse_program()
        # print(gt_actions)
        # if gt_actions == None:
        #     continue

        is_ok = False
        try_idx = 0
        while try_idx < MAX_TRY_NUM:
            # while try_idx < len(gt_actions):
            sample_action = np.random.choice(env.get_available_actions())
            # sample_action = env.action_space.action_id(gt_actions[try_idx])
            # print(env.action_space.action_desc(sample_action),
            #       [env.action_space.action_desc(x) for x in env.get_available_actions()])
            assert sample_action in env.get_available_actions()

            obs, reward, done, info = env.step(sample_action)
            if len(env.math_engine.curr_actions_desc) > 20 or reward == env.rewards.ans_right:
                done = True
            if done:
                logging.info(f"try_idx: {try_idx} reward: {reward}")
                if reward == env.rewards.ans_right:
                    is_ok = True
                    break
                else:
                    obs = env.reset(problem=problem)
            try_idx += 1
        if is_ok:
            actions = env.math_engine.curr_actions_desc
            print(f"{problem.question}\n{problem.answer}\n{actions}\n")
            search_dict[tuple(problem.question)].append(actions)
        else:
            print(f"{problem.question}\n{problem.answer}\n{'None'}\n")
