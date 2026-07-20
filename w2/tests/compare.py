import sys
import argparse
import math

def main():
    parser = argparse.ArgumentParser(description='Compare conv2d test outputs')
    parser.add_argument('expected')
    parser.add_argument('actual')
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--atol', type=float, default=1e-6)
    args = parser.parse_args()

    def load_tests(path):
        tests = []
        with open(path) as f:
            lines = [l.rstrip('\n') for l in f]
        cur = {}
        for line in lines:
            if not line.strip():
                continue
            if line.startswith('INPUT_SHAPE:'):
                if cur:
                    tests.append(cur)
                    cur = {}
                key, val = line.split(':', 1)
                cur[key.strip()] = val.strip()
            else:
                if ':' in line:
                    key, val = line.split(':', 1)
                    cur[key.strip()] = val.strip()
                else:
                    # ignore malformed
                    pass
        if cur:
            tests.append(cur)
        return tests

    def parse_shape(s):
        return [int(x) for x in s.split(',') if x != '']

    def parse_floats(s):
        if s == '':
            return []
        return [float(x) for x in s.split(',') if x != '']

    exp_tests = load_tests(args.expected)
    act_tests = load_tests(args.actual)

    ok = True
    if len(exp_tests) != len(act_tests):
        print(f'Number of tests differ: expected {len(exp_tests)} vs actual {len(act_tests)}')
        ok = False

    def compare_vals(key, e, a):
        nonlocal ok
        # Shapes
        if key.endswith('_SHAPE'):
            es = parse_shape(e)
            as_ = parse_shape(a)
            if es != as_:
                print(f'MISMATCH {key}: expected {es} actual {as_}')
                ok = False
            return
        # Numeric lists
        if key.endswith('_VALUES'):
            ev = parse_floats(e)
            av = parse_floats(a)
            if len(ev) != len(av):
                print(f'MISMATCH {key} length: expected {len(ev)} actual {len(av)}')
                ok = False
                return
            for i, (evv, avv) in enumerate(zip(ev, av)):
                if not math.isclose(evv, avv, rel_tol=args.rtol, abs_tol=args.atol):
                    print(f'MISMATCH {key}[{i}]: expected {evv} actual {avv}')
                    ok = False
            return
        # STRIDE / DILATION may be numeric pairs
        if key in ('STRIDE', 'DILATION'):
            try:
                es = parse_shape(e)
                as_ = parse_shape(a)
                if es != as_:
                    print(f'MISMATCH {key}: expected {es} actual {as_}')
                    ok = False
            except Exception:
                if e != a:
                    print(f'MISMATCH {key}: expected {e} actual {a}')
                    ok = False
            return
        # Other fields: direct compare
        if e != a:
            print(f'MISMATCH {key}: expected "{e}" actual "{a}"')
            ok = False

    n = min(len(exp_tests), len(act_tests))
    for i in range(n):
        print(f'Comparing test {i}')
        et = exp_tests[i]
        at = act_tests[i]
        keys = set(et.keys()) | set(at.keys())
        for k in sorted(keys):
            ev = et.get(k)
            av = at.get(k)
            if ev is None:
                print(f'MISSING in expected: {k}')
                ok = False
                continue
            if av is None:
                print(f'MISSING in actual: {k}')
                ok = False
                continue
            compare_vals(k, ev, av)

    if ok:
        print('ALL MATCH')
        sys.exit(0)
    else:
        print('DIFFERENCES FOUND')
        sys.exit(2)


if __name__ == "__main__":
    main()
