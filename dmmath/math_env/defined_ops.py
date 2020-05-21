import re
from sympy.parsing.sympy_parser import parse_expr
from sympy import Mul, Pow, Poly
from sympy import expand
from sympy import S, Symbol

defined_ops = [('api', x) for x in
               ['expr2func', 'assign', 'mysolve', 'toseq',
                'min', 'max', 'intersect', 'Minus', 'Divide']]


def subs(expr, ref_dict):
    if hasattr(expr, 'atoms'):
        all_symbols = [str(x) for x in expr.atoms(Symbol)]
        sub_vals = {x: ref_dict[x] for x in all_symbols if x in ref_dict}
        expr = expr.subs(sub_vals)
    return expr


def eq_transform(expr):
    expr = str(expr)
    assert '=' in expr
    l, r = expr.split('=')
    res = l + f'-({r})'
    return res


def myexpand(v):
    v = expand(v)
    v = str(v)
    v = poly_descent(v)
    return v


def toseq(*args):
    return args


def mysolve(expr, var=None):
    from sympy import solve
    dict_res = solve(expr, dict=True)
    if len(dict_res) == 1:
        res = dict_res[0]
    elif len(dict_res) > 1:
        res = solve(expr, dict=False)
        var = list(dict_res[0].keys())[0]
        res = {var: res}
    return res


def intersect(a, b):
    return list(set(a) & set(b))


def is_func_expr(expr):
    expr = str(expr)
    assert '=' in expr
    l, r = [x.strip() for x in expr.split('=')]
    if l[0] == '(' and l[-1] == ')' and r[0] == '(' and r[-1] == ')':
        l = l[1:-1]
        r = r[1:-1]

    pattern = r'(\w+)\((\w+)\)'
    m = re.match(pattern, l)
    return m is not None


def expr2func(expr1, expr2=None):
    expr1 = str(expr1)
    if expr2 is None:
        assert is_func_expr(expr1)
        expr = expr1
    else:
        expr2 = str(expr2)
        expr = f'{expr1} = {expr2}'
    l, r = [x.strip() for x in expr.split('=')]
    if l[0] == '(' and l[-1] == ')' and r[0] == '(' and r[-1] == ')':
        l = l[1:-1]
        r = r[1:-1]
    cmd_str = f'def {l}: return {r}'

    return cmd_str


def assign(x_str, y_str):
    x_str = str(x_str)
    y_str = str(y_str)
    cmd_str = f'{x_str} = {y_str}'
    if is_func_expr(cmd_str):
        return _warp_expr_with_quota(cmd_str)
    else:
        return cmd_str


def Minus(a, b):
    if isinstance(a, str):
        a = parse_expr(a)
    if isinstance(b, str):
        b = parse_expr(b)
    return a - b


def Divide(a, b):
    if isinstance(a, str):
        a = parse_expr(a)
    if isinstance(b, str):
        b = parse_expr(b)
    return Mul(a, Pow(b, -1))


def _warp_expr_with_quota(expr):
    return f"'{expr}'"


def poly_descent(expr):
    if expr[0] == '-' and expr[1:].isdigit() or expr.isdigit():
        return expr
    p = Poly(parse_expr(str(expr)))
    vars = set([])
    for c in str(expr):
        if c.isalpha():
            vars.add(c)
    assert len(vars) > 0
    if len(vars) > 1:
        return expr
    var = vars.pop()
    s = ''
    lst = sorted(p.all_terms(), key=lambda x: x[0], reverse=True)
    for d, c in lst:
        d = d[0]
        if d == 0:
            if c > 0:
                s += f' + {c}'
            elif c < 0:
                s += f' - {abs(c)}'
        elif d == 1:
            if c > 0 and c != 1:
                s += f' + {c}*{var}'
            elif c == 1:
                s += f' + {var}'
            elif c < 0 and c != -1:
                s += f' - {abs(c)}*{var}'
            elif c == -1:
                s += f' - {var}'

        else:
            if c > 0 and c != 1:
                s += f' + {c}*{var}**{d}'
            elif c == 1:
                s += f' + {var}**{d}'
            elif c < 0 and c != -1:
                s += f' - {abs(c)}*{var}**{d}'
            elif c == -1:
                s += f' - {var}**{d}'
    if s[:3] == ' - ':
        s = '-' + s[3:]
    if s[:3] == ' + ':
        s = s[3:]

    return s
