from numpy import arange

__all__ = [
    'build_time', 'build_seq_list', 'standardise_activities'
]


def build_time(subs, win_len, fs):
    win = arange(win_len, dtype=float) / fs
    inc = win_len / fs
    t = []
    prev_sub = subs[0]
    for curr_sub in subs:
        if curr_sub != prev_sub:
            win = arange(win_len, dtype=float) / fs
        t.extend(win)
        win += inc
        prev_sub = curr_sub
    return t


def build_seq_list(subs, win_len):
    seq = []
    si = 0
    last_sub = subs[0]
    for prev_sub in subs:
        if prev_sub != last_sub:
            si += 1
        seq.extend([si] * win_len)
        last_sub = prev_sub
    return seq


def standardise_activities(lookup, activities):
    return [lookup[act] for act in activities]
