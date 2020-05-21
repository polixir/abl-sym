import logging
from dmmath.math_env.math_env import DMMathEnv
from dmmath.utils import ConstParams, parse_dataset
from dmmath.nlp.nlp_utils import DMMathDatasetReader
from dmmath.nlp.transformer import MyTransformer
from dmmath.utils import Problem
from allennlp.predictors import Seq2SeqPredictor
import torch
import argparse
from collections import defaultdict


def get_q2progs(predictor, data_path):
    '''
    get candidate programs for compositional problems based on simple problems
    '''
    problems = parse_dataset(data_path)
    part_text_lst = []
    q2part_idxes = defaultdict(lambda: [])
    cnt = 0
    for problem in problems:
        lst = problem.text.split('. ')
        for idx in range(len(lst)):
            if idx == len(lst) - 1:
                part_text = lst[idx]
            else:
                part_text = lst[idx] + '.'
            part_text = part_text.strip()
            instance = predictor._json_to_instance({"source": part_text})
            l = instance["source_tokens"].sequence_length()
            part_text_lst.append((part_text, l))
            q2part_idxes[problem.text].append(cnt)
            cnt += 1
    for text, part_idxes in q2part_idxes.items():
        composed_text = ' '.join([part_text_lst[x][0] for x in part_idxes])
        assert text == composed_text

    bsize = 512
    res_all = []
    for i in range(len(part_text_lst) // bsize + 1):
        bdata = [x[0] for x in part_text_lst[bsize * i:bsize * i + bsize]]
        if len(bdata) == 0:
            continue
        batch_data = [{"source": x} for x in bdata]
        res = predictor.predict_batch_json(batch_data)
        print(res)
        res_all.extend([x["predicted_tokens"] for x in res])

    q2progs = {}
    for pb, part_idxes in q2part_idxes.items():
        lb = 0
        lst = []
        for idx in part_idxes:
            part_text, l = part_text_lst[idx]
            res = res_all[idx]
            res_new = []
            for idy, api in enumerate(res):
                if api == 'extra#@end@':
                    continue
                k, v = api.split('#')
                if 'pos' == k:
                    if int(v) >= l:
                        if idy == 0:
                            continue
                        else:
                            break
                    v = int(v) + lb
                    newapi = f'pos#{v}'
                else:
                    newapi = f'{k}#{v}'
                res_new.append(newapi)
            lb = lb + l
            lst.append(res_new)
        progs = []
        for x in generate_candidate_programs_all(lst):
            if is_proper_program(x):
                progs.append(x + ['extra#@end@'])
        q2progs[pb] = progs
    return q2progs


def generate_candidate_programs(prog):
    if prog == []:
        return [[]]
    else:
        api = prog[0]
        k, v = api.split('#')
        proper_apis = []
        if k == 'argc':
            proper_apis.extend([['argc#1'], ['argc#2'], []])
        elif k == 'api':
            proper_apis.extend([[api], []])
        else:
            proper_apis.extend([[api]])

        progs = []
        for x in proper_apis:
            for y in generate_candidate_programs(prog[1:]):
                z = x + y
                progs.append(z)
        return progs


def generate_candidate_programs_all(prog_lst):
    if prog_lst == []:
        return [[]]
    else:
        prog = prog_lst[0]
        candidate_progs = generate_candidate_programs(prog)
        z = [x + y for x in candidate_progs for y in generate_candidate_programs_all(prog_lst[1:])]
        return z


def is_proper_program(prog):
    is_ok = True
    for i, api in enumerate(prog):
        k, v = api.split('#')
        if k == 'argc':
            if i == len(prog) - 1 or 'api' not in prog[i + 1]:
                is_ok = False
        if k == 'api':
            if i == 0 or 'argc' not in prog[i - 1]:
                is_ok = False
    return is_ok


def prog_txt2lst(prog_txt):
    program_seq = []
    for x in prog_txt.split():
        api, name = x.split('#')
        if api == 'pos':
            api = 'position'
        program_seq.append((api, name))

    return program_seq


parser = argparse.ArgumentParser(description='perform search for compositional problems')
parser.add_argument('--model', type=str, help='model(.tar.gz format) path')
parser.add_argument('--data', type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    predictor = Seq2SeqPredictor.from_path(args.model, predictor_name='seq2seq',
                                           cuda_device=0)
    q2progs = get_q2progs(predictor, args.data)

    logging.basicConfig(filename=None, filemode='w', level=logging.INFO)

    env_kwargs = {
        'data_path': args.data,
        'ops_typ': 'search_basic',
    }

    env = DMMathEnv(**env_kwargs)

    while True:
        obs = env.reset()
        problem = env.problem
        if problem.answer in ['True', 'False']:
            continue
        if {'digit', 'round', 'rounded', 'sort'} & set(problem.text.lower().split()) or '/' in problem.text:
            continue
        progs = q2progs[problem.text]
        print(problem.text)
        # print(len(progs))
        is_ok = False
        for prog in progs:
            env.reset(problem=problem)
            for action in prog:
                k, v = action.split('#')
                if k == 'pos':
                    k = 'position'
                action = (k, v)
                sample_action = env.action_space.action_id(action)
                obs, reward, done, info = env.step(sample_action)
            if done and reward == env.rewards.ans_right:
                is_ok = True
                print(f"{problem.answer}###{prog}")
                break
        if not is_ok:
            print(f"{problem.answer}###extra#None")
