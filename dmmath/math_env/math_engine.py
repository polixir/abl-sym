from dmmath.utils import Problem, RunStatus, StackStatus, ExecStatus
from dmmath.math_env.ops import Actions
from typing import Iterable
import numpy as np
import timeout_decorator
import re
from sympy.core import numbers
from dmmath.math_env.defined_ops import expr2func, eq_transform, assign, myexpand
from sympy.parsing.sympy_parser import parse_expr
from sympy import N

nths = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']
nths_dict = dict(zip(nths, range(1, len(nths) + 1)))
nths_dict['square'] = 2
nths_dict['cube'] = 3

func_pattern = re.compile('\w\([^,]+\)\s?=\s?\S+')


def is_eq(expr):
    if '=' in expr:
        return True
    else:
        return False


def _is_func_expr(expr):
    if func_pattern.match(str(expr)):
        return True
    else:
        return False


class MathEngine:
    def __init__(self, action_spaces: Actions, problem: Problem = None):
        self.action_spaces = action_spaces
        self._init_ctx()
        self.curr_actions_desc = []
        self.problem = problem
        self._tv = 0

    def run(self):
        lang, api_name = self.curr_actions_desc[-1]

        is_fail = False
        if lang == 'position':
            token = self.problem.question[int(api_name)]
            expr = token.text
            if expr in nths_dict:
                expr = str(nths_dict[expr])
            if _is_func_expr(expr):
                expr = expr2func(expr)
            elif not is_eq(expr):
                if '/' in expr:
                    expr = f'subs(parse_expr({self._warp_expr_with_quota(expr)}), globals())'
            try:
                if isinstance(parse_expr(expr), (float, numbers.Float)):
                    expr = f"N({expr}, 20)"
            except:
                pass
            self._stack.append((lang, expr))
        elif lang in ['argc']:
            self._stack.append((lang, api_name))
        elif lang in ['api'] and api_name != '(extra, @end@)':
            _, v = self._stack.pop()
            assert v.isdigit(), 'argc must be digit'
            argc = int(v)
            assert len(self._stack) >= argc
            args = []
            for _ in range(argc):
                arg_typ, arg = self._stack.pop()
                args.append((arg_typ, arg))
            if api_name in ['expr2func', 'assign']:
                args = [x[1] for x in args]
                try:
                    cmd_str = eval(f'{api_name}(*{args})')
                except Exception as e:
                    cmd_str = 'wrong cmd'
                    pass
            else:
                lst = []
                for arg_typ, arg in args:
                    if arg_typ == 'position' and isinstance(arg, str):
                        if is_eq(arg):
                            arg = eq_transform(arg)
                    lst.append(arg)
                cmd_str = '''{}({})'''.format(api_name, ','.join(lst))
            self._stack.append(('value', cmd_str))
            try:
                tv = self.exec()
                if isinstance(tv, dict):
                    for k, v in tv.items():
                        self._stack.append(('value', f'{k} = {v}'))
            except Exception as e:
                pass
                # print(e)
        else:
            assert api_name == '@end@'
            try:
                self._tv = self.exec()
                # print(self._format_result(self._tv), self.problem.answer)
            except Exception as e:
                # print(e)
                is_fail = True
        # if len(self._stack) > 8:
        #     is_fail = True
        #     raise ('_stack is too long!')
        self._tv = self._format_result(self._tv)
        if is_fail:
            result = None
        else:
            result = self._tv
        status = self.get_run_status(result, self.problem.answer)
        return status

    @timeout_decorator.timeout(10, use_signals=False)
    def exec(self):
        cmd_strs = [x[1] for x in self._stack]
        cmd_str = '\n'.join(cmd_strs[:-1]) + '\n' + f'tv = {cmd_strs[-1]}'
        subs_str = f'tv = subs(tv, globals())'
        cmd_str = f'{self._env_cmd_str}\n{cmd_str}\n{subs_str}'
        # print(cmd_str)
        exec(cmd_str, globals(), globals())
        tv = globals().get('tv')
        return tv

    def _format_result(self, tv):
        try:
            if tv.is_polynomial() and ('api', 'factor') not in self.curr_actions_desc:
                tv = myexpand(tv)
        except:
            pass
        if isinstance(tv, Iterable) and not isinstance(tv, str):
            tv = ', '.join([str(x) for x in tv])
        elif isinstance(tv, (float, numbers.Float)):
            tv = str(N(tv, 20))
        else:
            # print(tv)
            tv = str(tv)
        return tv

    def _init_ctx(self):
        env_str = \
            '''
from sympy.abc import *
from sympy import *
from dmmath.math_env.defined_ops import *
from sympy.parsing.sympy_parser import parse_expr
            '''
        self._env_cmd_str = env_str
        self._stack = []

    def _warp_expr_with_quota(self, expr):
        return f"'{expr}'"

    def float_result_check(self, result, answer):
        lst = []
        if isinstance(result, str) and isinstance(answer, str):
            if ',' in result:
                res_list = result.split(',')
            else:
                res_list = [result]
            if ',' in answer:
                ans_list = answer.split(',')
            else:
                ans_list = [answer]
            if len(res_list) == len(ans_list):
                for res, ans in zip(res_list, ans_list):
                    if res.strip('+').strip('-').replace('.', '', 1).isdigit() and ans.strip('-').strip('+').replace(
                            '.', '', 1).isdigit():
                        if abs(float(res) - float(ans)) < 1e-5:
                            lst.append(1)
                        else:
                            lst.append(0)
        if len(lst) > 0 and (np.array(lst) == 1).all():
            return True
        else:
            return False

    def get_run_status(self, result, answer):
        float_res_right = self.float_result_check(result, answer)

        if result is None:
            status = RunStatus.Fail
        elif not float_res_right and result != answer:
            status = RunStatus.Success
        else:
            status = RunStatus.Right
        return status

    def push(self, action):
        k, v = self.action_spaces.action_desc(action)
        self.curr_actions_desc.append((k, v))

    def reset(self, problem):
        self._init_ctx()
        self.problem = problem
        self.curr_actions_desc = []
        self._tv = 0

    def stack_state(self):
        sz = min(len(self._stack), len(StackStatus) - 1)
        return StackStatus(sz)

    def exec_state(self):
        sz = 0
        return ExecStatus(sz)
