from dmmath.math_env.sympy_helper import get_sympy_module_op_dict
from dmmath.math_env.defined_ops import defined_ops
from gym.spaces import Discrete
from dmmath.utils import OP_END_SYMBOL, OP_START_SYMBOL, OP_PADDING_SYMBOL
from sympy.abc import *
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
import pickle as pkl
from dmmath.math_env.defined_ops import *


class MathOperators:
    sympy_modules = ['sympy.core', 'sympy.polys', 'sympy.functions', 'sympy.ntheory', 'sympy.simplify', 'sympy.solvers',
                     'sympy.calculus', 'sympy.algebras']
    defined_ops = "define"
    variable_op_module = "variable"
    constant_op_module = "constant"
    args_op_module = "args_op"
    position_op_module = "position_op"
    extra_basic_modules = [defined_ops, constant_op_module, args_op_module, position_op_module]
    no_const_basic_modules = [defined_ops, args_op_module, position_op_module]
    all_modules = sympy_modules + extra_basic_modules + [variable_op_module]

    def __init__(self, chosen_modules=None):
        if chosen_modules is None:
            self.chosen_modules = MathOperators.all_modules
        else:
            self.chosen_modules = chosen_modules
        self.module_ops_dict = {}
        self._setup()

    def _setup(self, args_num=3, position_num=100):
        self._get_sympy_api_dict()
        self._get_defined_ops()
        self._get_variable_ops()
        self._get_constant_ops()
        self._get_argc_ops(args_num)
        self._get_postion_ops(position_num)

    def _get_sympy_api_dict(self):
        for module in MathOperators.sympy_modules:
            self.module_ops_dict[module] = [('api', x) for x in get_sympy_module_op_dict(module)]
        self.module_ops_dict['sympy.core'].extend([('api', 'parse_expr')])

    def _get_defined_ops(self):
        ops = defined_ops
        self.module_ops_dict[MathOperators.defined_ops] = ops

    def _get_variable_ops(self):
        self.module_ops_dict[MathOperators.variable_op_module] = [('var', str(chr(x))) for x in
                                                                  range(ord('a'), ord('z') + 1)]

    def _get_constant_ops(self):
        # ops = [('const', '-1'), ('const', '0'), ('const', '1'), ('const', '2'), ('const', '3'), ('const', '10')]
        ops = []
        self.module_ops_dict[MathOperators.constant_op_module] = ops

    def _get_argc_ops(self, max_num):
        self.module_ops_dict[MathOperators.args_op_module] = [('argc', str(i)) for i in range(1, max_num + 1)]

    def _get_postion_ops(self, position_num):
        self.module_ops_dict[MathOperators.position_op_module] = [('position', str(i)) for i in range(position_num)]

    def get_ops(self, typ):
        if typ == None or typ == 'all':
            chosen_modules = MathOperators.all_modules
            ops = self._get_modules_op(chosen_modules)
            print(ops)
        elif typ == 'search_basic':
            chosen_modules = MathOperators.no_const_basic_modules
            ops = self._get_modules_op(chosen_modules)
            for v in self._get_sympy_manu_purified_ops().values():
                ops.update(v)
        else:
            chosen_modules = MathOperators.extra_basic_modules + typ.split(',')
            ops = self._get_modules_op(chosen_modules)
            for v in self._get_sympy_purified_ops().values():
                ops.update(v)

        return list(ops)

    def _get_modules_op(self, modules):
        ops = set([])
        for m in modules:
            ops.update(self.module_ops_dict[m])
        return ops

    def _get_sympy_purified_ops(self):
        return {
            "sympy.core":
                [('api', 'prod'), ('api', 'ilcm'), ('api', 'igcd'),
                 ('api', 'expand'), ('api', 'comp'), ('api', 'expand_log'), ('api', 'integer_log'),
                 ('api', 'expand_func'), ('api', 'factor_terms'), ('api', 'integer_nthroot'), ('api', 'sympify'),
                 ('api', 'factor_nc'), ('api', 'gcd_terms'), ('api', 'Pow'), ('api', 'expand_mul'), ('api', 'Subs'),
                 ('api', 'parse_expr'), ('api', 'Minus'), ('api', 'Divide'), ('api', 'Ge'), ('api', 'Gt'),
                 ('api', 'Le'), ('api', 'Lt'), ('api', 'Mod')],
            "sympy.polys":
                [('api', 'gcd'), ('api', 'count_roots'), ('api', 'poly'), ('api', 'total_degree'),
                 ('api', 'decompose'), ('api', 'factor'), ('api', 'compose'), ('api', 'gcd_list'),
                 ('api', 'real_roots'),
                 ('api', 'poly_from_expr'), ('api', 'terms_gcd'), ('api', 'pdiv'),
                 ('api', 'cofactors'), ('api', 'nth_power_roots_poly'), ('api', 'roots'),
                 ('api', 'minimal_polynomial'), ('api', 'ground_roots'), ('api', 'lcm'), ('api', 'monic'),
                 ('api', 'lcm_list'),
                 ('api', 'factor_list'), ('api', 'nroots'), ('api', 'rem'),
                 ('api', 'LM'), ('api', 'prem'), ('api', 'Monomial'), ('api', 'exquo'), ('api', 'degree'),
                 ('api', 'discriminant'),
                 ('api', 'resultant'), ('api', 'apart_list'), ('api', 'apart'), ('api', 'invert'),
                 ('api', 'LT'), ('api', 'content'), ('api', 'LC'), ('api', 'together'), ('api', 'div')],
            "sympy.functions":
                [('api', 'sec'), ('api', 'floor'), ('api', 'real_root'),
                 ('api', 'log'), ('api', 'ln'), ('api', 'sqrt'), ('api', 'frac'),
                 ('api', 'root'), ('api', 'sin'), ('api', 'sign'),
                 ('api', 'ceiling'), ('api', 'exp'), ('api', 'Abs'), ('api', 'cos'), ('api', 'tan')],
            "sympy.ntheory":
                [('api', 'prime'), ('api', 'divisor_count'), ('api', 'myisprime'), ('api', 'primitive_root'),
                 ('api', 'composite'),
                 ('api', 'divisors'), ('api', 'factorint'), ('api', 'primefactors'), ('api', 'nextprime')],
            "sympy.simplify":
                [('api', 'ratsimp'), ('api', 'simplify'), ('api', 'denom'), ('api', 'hypersimilar'),
                 ('api', 'combsimp'), ('api', 'radsimp'), ('api', 'fraction'), ('api', 'collect_const'),
                 ('api', 'rcollect'),
                 ('api', 'hypersimp'), ('api', 'hyperexpand'), ('api', 'collect'), ('api', 'bottom_up'),
                 ('api', 'nsimplify'),
                 ('api', 'numer'), ('api', 'posify')],
            "sympy.solvers":
            # [('api', 'homogeneous_order'), ('api', 'linsolve'), ('api', 'solve'), ('api', 'solve_poly_system'),
            # ('api', 'solve_linear'), ('api', 'nsolve'), ('api', 'solveset')],
                [],
            "sympy.calculus":
                [],
            "sympy.algebras":
                []
        }

    def _get_sympy_manu_purified_ops(self):
        return {
            "sympy.core":
                [('api', 'expand'),
                 ('api', 'integer_nthroot'),
                 ('api', 'Pow'), ('api', 'Add'), ('api', 'Mul'),
                 ('api', 'Ge'), ('api', 'Gt'),
                 ('api', 'Le'), ('api', 'Lt'), ('api', 'Mod'), ('api', 'parse_expr')
                 ],
            "sympy.polys":
                [('api', 'gcd'), ('api', 'lcm'), ('api', 'factor')
                 ],
            "sympy.functions":
                [('api', 'floor'),
                 ('api', 'sqrt'),
                 ('api', 'sign'),
                 ('api', 'ceiling'), ('api', 'Abs'), ('api', 'diff'), ('api', 'root')],
            "sympy.ntheory":
                [('api', 'divisors'), ('api', 'primefactors'), ('api', 'denom')],
            "sympy.simplify":
                [('api', 'simplify'),
                 ('api', 'collect'), ('api', 'collect_const')],
            "sympy.solvers":
                [],
            "sympy.calculus":
                [],
            "sympy.algebras":
                [],
        }

    def __str__(self):
        s = ""
        for m in self.module_ops_dict:
            s += f"{m} num: {len(self.module_ops_dict[m])}\n"
            s += f"{self.module_ops_dict[m]}\n"
        return s


class Actions(Discrete):
    def __init__(self, ops_typ='all'):
        all_ops = get_ops(ops_typ)
        self.padding_symbol = OP_PADDING_SYMBOL
        self.start_symbol = OP_START_SYMBOL
        self.end_symbol = OP_END_SYMBOL
        self.extra_ops = [self.padding_symbol, self.start_symbol, self.end_symbol]
        all_ops = self.extra_ops + list(set(all_ops) - set(self.extra_ops))
        self.all_ops = all_ops
        super(Actions, self).__init__(len(self.all_ops))
        self.op2id = dict(zip(self.all_ops, range(len(self.all_ops))))
        self.argc2ops = self._get_argc2ops()

    @property
    def useless_actions(self):
        return [self.padding_symbol, self.start_symbol]

    def __getitem__(self, opid):
        assert self.contains(opid)
        return self.all_ops[opid]

    def sample(self):
        NotImplemented()

    def action_desc(self, opid):
        return self.all_ops[opid]

    def action_id(self, action_desc):
        return self.op2id[action_desc]

    @property
    def action_descs(self):
        return list(self.all_ops)

    @property
    def actions(self):
        return list(range(self.n))

    def _get_argc2ops(self):
        argc2ops = get_argc2apis(self.all_ops)
        return argc2ops


def get_ops(typ):
    m = MathOperators()
    return m.get_ops(typ)


def expr_pool():
    d = {
        "const": [str(x) for x in [-1, 0, 1, 2, 0.1]],
        "poly": ['x', 'y'],
        "arith": ["1/2", "11"],
    }
    vs = []
    for k, v in d.items():
        vs += v
        vs += [parse_expr(x) for x in v]
    vs += [(1, 2), {1: '1', 2: '2'}]
    vs += ['f(x)=x']
    return vs


def get_argc2apis(ops):
    pkl_path = 'data/api_info.pkl'
    import os
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            argc2apis = pkl.load(f)
        return argc2apis

    ops = list(filter(lambda x: x[0] != 'position', ops))

    vals = expr_pool()
    argc_argvs_dict = {
        1: [(v,) for v in vals],
        2: [(v1, v2) for v1 in vals for v2 in vals],
        3: [(v1, v2, v3) for v1 in vals for v2 in vals for v3 in vals],
    }

    argc2apis = {}
    for op in ops:
        if op[0] == 'api':
            is_ok = False
            if op[1] == 'primefactors':
                argcs = [1]
            elif op[1] in ['Mul', 'Add']:
                argcs = [2, 3]
            else:
                argcs = [1, 2, 3]
            for argc in argcs:
                argvs = argc_argvs_dict[argc]
                for argv in argvs:
                    try:
                        cmd_str = f'''{op[1]}(*{argv})'''
                        exec(cmd_str)
                        is_ok = True
                        argc = len(argv)
                        if argc in argc2apis:
                            argc2apis[argc].add(op)
                        else:
                            argc2apis[argc] = {op}
                        # print(op, argc)
                        break
                    except Exception as e:
                        continue
            if is_ok == False:
                print(f'{op} is fail!!!!!!!!!')
    with open(pkl_path, 'wb') as f:
        pkl.dump(argc2apis, f)
    return argc2apis
