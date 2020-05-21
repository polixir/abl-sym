import gym
from gym import spaces

from dmmath.math_env.ops import Actions
from dmmath.math_env.math_engine import MathEngine
from dmmath.utils import Problem, Rewards, RunStatus, Observation
from dmmath.nlp.nlp_utils import DMMathTokenizer


class DMMathEnv(gym.Env):
    def __init__(self, data_path, ops_typ='all', rewards=None):
        super(DMMathEnv, self).__init__()
        self.action_space = Actions(ops_typ=ops_typ)
        self.observation_space = spaces.Box(0, 1, shape=(1,))
        self.setup(data_path)
        self.problem: Problem = None
        self.math_engine = MathEngine(self.action_space, self.problem)
        if rewards is None:
            self.rewards = Rewards(ans_right=1.0, run_success=0.0, run_fail=-10.0)
        else:
            self.rewards = Rewards(ans_right=rewards["right"], run_success=rewards["success"], run_fail=rewards["fail"])
        self._obs = None

    def _get_init_avail_actions(self):
        avail_action_descs = []
        for op in self.action_space.action_descs:
            if op in self.action_space.useless_actions:
                continue
            s1, s2 = op
            if s1 == "position" and int(s2) < len(self.problem.question):
                if self.problem.question[int(s2)].is_expr:
                    avail_action_descs.append(op)
            if s1 != "position":
                avail_action_descs.append(op)
        return avail_action_descs

    def get_available_actions(self, using_restrict_policy=True):
        avail_action_descs = self._init_avail_actions
        if using_restrict_policy:
            used_actions = self.math_engine.curr_actions_desc
            # print('used actions', used_actions)
            stack_states = self.math_engine._stack
            # print('stack states', stack_states)
            avail_action_descs = list(
                filter(lambda x: not (x[0] == 'argc' and int(x[1]) > len(stack_states)), avail_action_descs))
            # avail_action_descs = list(
            #     filter(lambda x: not (x == ('api', 'eval') and len(exec_states) < 2), avail_action_descs))
            if len(used_actions) == 0:
                avail_action_descs = list(
                    filter(lambda x: not (x[0] in ['api', 'extra']), avail_action_descs))
            if len(used_actions) > 0 and used_actions[-1][0] == 'argc':
                argc = int(used_actions[-1][1])
                avail_action_descs = self.action_space.argc2ops[argc]

            if len(used_actions) > 0 and used_actions[-1][0] != 'argc':
                avail_action_descs = list(filter(lambda x: not (x[0] in ['api']), avail_action_descs))

            lst = []
            for x in avail_action_descs:
                if x[0] == 'position':
                    if used_actions.count(x) <= 0:
                        lst.append(x)
                else:
                    lst.append(x)
            avail_action_descs = lst

            if len([x for x in used_actions if x[0] == 'api']) == 0:
                avail_action_descs = [x for x in avail_action_descs if x != ('extra', '@end@')]

        avail_actions = [self.action_space.action_id(op) for op in avail_action_descs]
        return avail_actions

    def setup(self, data_path):
        tokenizer = DMMathTokenizer()
        self.data_gen = self._read(data_path, tokenizer)

    def reset(self, problem=None):
        if problem is None:
            self.problem = self.data_gen.__next__()
        else:
            self.problem = problem
        assert self.problem is not None
        self.math_engine.reset(self.problem)
        self._init_avail_actions = self._get_init_avail_actions()
        obs = self._get_obs(using_cache=False)
        return obs

    def step(self, action):
        self.math_engine.push(action)
        status, reward = self._get_reward()
        done = False
        if reward == self.rewards.run_fail or self.action_space.action_desc(action) == self.action_space.end_symbol:
            done = True

        obs = self._get_obs()
        info = self._get_info()
        info['status'] = status
        return obs, reward, done, info

    def _get_obs(self, using_cache=True):
        avail_actions = self.get_available_actions()
        if not using_cache:
            self._obs = Observation(self.problem, avail_actions, self.math_engine.stack_state(),
                                    self.math_engine.exec_state())
        else:
            self._obs = self._obs._replace(avail_actions=avail_actions,
                                           stack_state=self.math_engine.stack_state(),
                                           exec_state=self.math_engine.exec_state())
        return self._obs

    def _get_info(self):
        info = {
            "problem": self.problem
        }
        return info

    def _get_reward(self):
        status = self.math_engine.run()
        if status == RunStatus.Fail:
            reward = self.rewards.run_fail
        else:
            if status == RunStatus.Success:
                reward = self.rewards.run_success
            else:
                reward = self.rewards.ans_right
        return status, reward

    def _read(self, filepath, tokenizer):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        for idx in range(len(lines) // 2):
            text = lines[2 * idx].strip()
            if text.startswith('%'):
                continue
            question = tokenizer.tokenize(text)
            anno = lines[2 * idx + 1].strip()
            if '###' in anno:
                answer, program = [x.strip() for x in anno.split('###')]
            else:
                answer, program = anno, None

            problem = Problem(text, question, answer, program)
            yield problem
