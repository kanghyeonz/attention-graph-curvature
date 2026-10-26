"""solution text, token spans of the steps and step labels"""

import numpy as np


def assemble(problem, steps):
    text = problem.strip() + "\n\n"
    spans = []
    for s in steps:
        start = len(text)
        text += s.strip() + "\n"
        spans.append((start, len(text)))
    return text, spans


def token_spans(offsets, char_spans):
    out = []
    for cs, ce in char_spans:
        idx = [t for t, (a, b) in enumerate(offsets)
               if b > cs and a < ce and not (a == 0 and b == 0)]
        out.append([min(idx), max(idx) + 1] if idx else [0, 0])
    return out


def step_labels(label, n_steps):
    # label -1: all correct. label k: steps<k correct (1), k error (0), >k unknown (-1, not used)
    y = np.ones(n_steps, dtype=np.int64)
    if label >= 0:
        y[label] = 0
        y[label + 1:] = -1
    return y
