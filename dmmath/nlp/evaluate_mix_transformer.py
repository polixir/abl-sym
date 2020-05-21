import torch
import argparse
from tqdm import tqdm
import logging
from dmmath.math_env import DMMathEnv
from dmmath.nlp.nlp_utils import DMMathTokenizer
from dmmath.utils import Problem

torch.manual_seed(1)


def run_program(pred_path, save_path):
    '''
    execute the program of prediction to see if it produces the correct answer.
    :param pred_path: prediction file
    :param save_path: file with the result of program execution
    :return:
    '''
    prog_f = open(save_path, 'w')

    env = DMMathEnv(data_path=pred_path, ops_typ='search_basic')
    tokenizer = DMMathTokenizer()

    with open(pred_path, 'r') as f:
        lines = f.readlines()
    for idx in tqdm(range(len(lines) // 3), 'program...', total=len(lines) // 3):
        ques = lines[3 * idx].strip()
        ans, prog = lines[3 * idx + 1].strip().split('###')
        p_ans, p_prog = lines[3 * idx + 2].strip().split('###')

        if p_prog == 'extra#None':
            run_status = 'fail'
            print(ques, file=prog_f)
            print(f'{ans}###{prog}', file=prog_f)
            print(f'{p_ans}###{p_prog}###{run_status}', file=prog_f)
            continue

        question = tokenizer.tokenize(ques)

        gt_actions = []
        for x in p_prog.split():
            k, v = x.split('#')
            if k == 'pos':
                k = 'position'
                if int(v) >= len(question):
                    continue
            gt_actions.append((k, v))

        problem = Problem(ques, question, ans, gt_actions)
        env.reset(problem=problem)
        gt_actions = problem.program

        for gt_action in gt_actions:
            sample_action = env.action_space.action_id(gt_action)
            obs, reward, done, info = env.step(sample_action)
            if done or len(env.math_engine.curr_actions_desc) > 30:
                break
        if done and reward == env.rewards.ans_right:
            run_status = 'right'
        elif done and reward == env.rewards.run_success:
            run_status = 'success'
        else:
            run_status = 'fail'
        print(ques, file=prog_f)
        print(f'{ans}###{prog}', file=prog_f)
        print(f'{p_ans}###{p_prog}###{run_status}', file=prog_f)
        # print(ques)
        # print(f'{ans}###{prog}')
        # print(f'{p_ans}###{p_prog}###{run_status}')
        # break


def evaluate(result):
    '''
    performs evaluation
    :param result: file with question, answer, program, execute_state
    :return:
    '''
    cnt = 0
    ans_corr = 0
    prog_corr = 0
    pred_corr = 0
    with open(result, 'r') as f:
        lines = f.readlines()
    for idx in tqdm(range(len(lines) // 3), 'stats...'):
        ques = lines[3 * idx].strip()
        ans, prog = lines[3 * idx + 1].strip().split('###')
        p_ans, p_prog, run_status = lines[3 * idx + 2].strip().split('###')

        if p_ans == ans:
            ans_corr += 1
        if p_prog == prog:
            prog_corr += 1
        if run_status == 'right' or (run_status == 'fail' and p_ans == ans):
            pred_corr += 1
        cnt += 1

    ans_acc = ans_corr * 1.0 / cnt
    prog_acc = prog_corr * 1.0 / cnt
    pred_acc = pred_corr * 1.0 / cnt
    logging.info(f'ans_acc = {ans_acc}')
    logging.info(f'prog_acc = {prog_acc}')
    logging.info(f'pred_acc = {pred_acc}')


parser = argparse.ArgumentParser()
parser.add_argument('--result', type=str, help='prediction result file path for evaluation')

if __name__ == "__main__":
    args = parser.parse_args()

    level = logging.INFO
    format = '  %(message)s'
    log_path = args.result + '.acc'
    handlers = [logging.FileHandler(log_path, mode='w'), logging.StreamHandler()]
    logging.basicConfig(level=level, format=format, handlers=handlers)

    program_path = f'{args.result}.program'
    run_program(args.result, program_path)
    evaluate(program_path)
