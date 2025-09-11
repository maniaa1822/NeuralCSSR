#!/usr/bin/env python3
import argparse
from pathlib import Path
import math


def load_tokens(path: Path):
    s = path.read_text().strip()
    return [int(c) for c in s if c in '01']


def compute_parity_before(tokens):
    # 0 = E, 1 = O before emitting tokens[i]
    s = 0
    out = []
    for t in tokens:
        out.append(s)
        if t == 1:
            s = 1 - s
        else:
            s = 0
    return out


def h2(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return - (p * math.log(p, 2) + (1 - p) * math.log(1 - p, 2))


def main():
    ap = argparse.ArgumentParser(description='Quick Even Process dataset check')
    ap.add_argument('--data', type=Path, required=True)
    args = ap.parse_args()

    tokens = load_tokens(args.data)
    n = len(tokens)
    if n < 10:
        raise SystemExit('Too few tokens')

    parity = compute_parity_before(tokens)

    cnt_E = sum(1 for s in parity)
    cnt_O = 0
    ones_E = 0
    zeros_E = 0
    zeros_O = 0
    ones_O = 0
    total_zero = 0

    for i in range(n):
        if tokens[i] == 0:
            total_zero += 1
        if parity[i] == 0:
            if tokens[i] == 1:
                ones_E += 1
            else:
                zeros_E += 1
        else:
            cnt_O += 1
            if tokens[i] == 1:
                ones_O += 1
            else:
                zeros_O += 1

    cnt_E = n - cnt_O
    p1_E = ones_E / max(1, cnt_E)
    p0_O = zeros_O / max(1, cnt_O)
    frac_O = cnt_O / n
    frac_zero = total_zero / n

    # Theoretical relations for Even Process with parameter p = P(1|E)
    p = p1_E
    frac_O_theory = p / (1 + p)
    frac_zero_theory = (1 - p) / (1 + p)
    H_rate_bits = h2(p) / (1 + p)

    print('Dataset:', args.data)
    print(f'Length: {n}')
    print(f'p(1|E)≈{p1_E:.6f}, p(0|O)≈{p0_O:.6f}')
    print(f'frac_O≈{frac_O:.6f} (theory≈{frac_O_theory:.6f})')
    print(f'zero_rate≈{frac_zero:.6f} (theory≈{frac_zero_theory:.6f})')
    print(f'Entropy rate (bits/sym) theory≈{H_rate_bits:.6f}')


if __name__ == '__main__':
    main()


