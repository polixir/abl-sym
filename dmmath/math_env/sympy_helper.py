import importlib
import inspect
import sympy
import typing
import re
from sympy.abc import x
from collections import OrderedDict


def op_filter(ops):
    def is_match(pattern, word):
        if '*' in pattern:
            if re.match(pattern, word):
                return True
        else:
            return pattern == word

    ops = filter(lambda x: hasattr(sympy, x), ops)
    not_allow_words_pattern = ['.*arg.*', '.*symbol.*', '.*check.*', '.*err', '.*error.*', '.*Error.*', '.*[fF]ailed.*',
                               'is_.*', 'cancel', 'empty', '.*[Pp]ython.*', 'Symbol', 'Basic',
                               'Function', 'S', '.*[fF]ield', '.*[rR]ing', 'Lambda', '.*Expr.*',
                               "FF_.*", "ZZ_.*", "QQ_.*", "GF", "FF", "ZZ", "QQ", "RR", "CC", "EX",
                               'Wild', 'Options', 'AlgebraicNumber', 'Tuple', 'Dummy', 'Integer', 'Piecewise',
                               'RealNumber', 'Id', 'SingularityFunction', 'Li', 'Number', 'DiracDelta', 'Float',
                               'Domain', 'Unequality',
                               'Dict', 'Rational', 'WildFunction', '.*[fF]actorial', '.*finite.*', '.*inequal.*',
                               'erf.*',
                               '.*pde.*', '.*gamma']
    filtered_ops = set()
    for op in ops:
        func = getattr(sympy, op)
        try:
            inspect.signature(func)
        except:
            continue
        if not any(is_match(x, op) for x in not_allow_words_pattern) and isinstance(func, typing.Callable):
            filtered_ops.add(op)
    # should_add_ops = ['E', 'I', 'nan', 'oo', 'pi'] + ['parse_expr']
    # filtered_ops.update(should_add_ops)
    return filtered_ops


def get_op_info_dict(op):
    func = getattr(sympy, op)
    doc_str = func.__doc__
    if isinstance(doc_str, str) and doc_str.strip():
        doc_str = doc_str.strip().split('\n\n')[0]
        sep_words = ['Examples', 'Parameters', 'See Also', '=======', '>>>', 'Notes', '------', 'Background',
                     'for example', 'For example']
        # sep_words = []
        if sep_words:
            pos = min(map(lambda x: doc_str.find(x) if x in doc_str else len(doc_str), sep_words))
            if pos == 0:
                pos = len(doc_str)
            doc_str = doc_str[:pos]
    else:
        if op == 'Mul':
            doc_str = 'Multiply, to find the product of by multiplication. for example, multiply 2, 4, i.e. 2*4'
        elif op == 'denom':
            doc_str = '''Denominator, in math, a denominator can be defined as the bottom number in a fraction that shows the number of equal parts an item is divided into. 
    It is the divisor of a fraction. for example, denom of 1/2 is 2'''
        elif op == 'numer':
            doc_str = 'Numerator, the part of a fraction that is above the line and signifies the number to be divided by the denominator, for example, numer of 1/2 is 1'
        elif op == 'Add':
            doc_str = '''Addition (often signified by the plus symbol "+") is one of the four basic operations of arithmetic; the others are subtraction,
             multiplication and division. The addition of two whole numbers is the total amount of those values combined. For example, "3 + 2 = 5"  i.e., 3 add 2 is equal to 5.'''
        else:
            doc_str = ''

    if isinstance(func, typing.Callable):
        args = []
        kwargs = {}
        for k, v in inspect.signature(func).parameters.items():
            if str(v).startswith('*'):
                if not str(v).startswith('**'):
                    args.append('*args')
                else:
                    kwargs['kwargs'] = '**kwargs'
            else:
                if v.default == inspect.Parameter.empty:
                    args.append(k)
                else:
                    kwargs[k] = v.default
        info_dict = {
            'type': 'callable',
            'func': func,
            'op': op,
            'args': args,
            'kwargs': kwargs,
            'doc': doc_str
        }
    else:
        info_dict = {
            'type': 'constant',
            'func': func,
            'op': op,
            'doc': doc_str
        }
    return info_dict


def get_sympy_module_op_dict(module_name):
    sympy_modules = ['sympy.core', 'sympy.polys', 'sympy.functions', 'sympy.ntheory', 'sympy.simplify', 'sympy.solvers',
                     'sympy.calculus', 'sympy.algebras']
    assert module_name in sympy_modules
    m = importlib.import_module(module_name)
    ops = set(m.__dict__.keys())
    ops = op_filter(ops)
    op_info_dicts = OrderedDict()
    for op in ops:
        op_info_dict = get_op_info_dict(op)
        if not op_info_dict['doc']:
            continue

        if op_info_dict['type'] != 'callable':
            op_info_dicts[op] = op_info_dict
            continue
        func = op_info_dict['func']
        available_args_list = [(2,), (2, 4), (2, 4, 8), ([2, 4, 8],), (x, 1), (x, 2), (x ** 2, 2), (x, 1, 2), (x, 2, 1),
                               (x,), (x, x),
                               (x, x, x), ([x, x, x],), ((sympy.Matrix([[1]]), sympy.Matrix([0])), [x])]
        is_input_right = False
        for arg in available_args_list:
            try:
                result = func(*arg)
                is_input_right = True
                op_info_dicts[op] = op_info_dict
                # print(f'{op} result type is {type(result)}')
                break
            except:
                continue
        if not is_input_right:
            # print(f'{op} input is wrong!')
            pass

    return op_info_dicts


def get_all_op2info():
    sympy_modules = ["core", "polys", "functions", "ntheory", "simplify", "solvers", "calculus", "algebras"]
    op2info = OrderedDict()
    for module in sympy_modules:
        op_info_dicts = get_sympy_module_op_dict('sympy.' + module)
        op2info.update(op_info_dicts)
    return op2info

