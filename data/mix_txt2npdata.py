import sys
import numpy as np
from tqdm import tqdm

fil = sys.argv[1]
vocab_dir = sys.argv[2]

with open(vocab_dir + '/source_tokens.txt', 'r') as f:
    src_vocab_token2index = dict({x.strip('\n'): idx + 1 for idx, x in enumerate(f.readlines())})
    src_vocab_token2index['@@PADDING@@'] = 0
target_namespaces = ["answer", "program"]
dst_vocab_token2index = {}
for tn in target_namespaces:
    with open(vocab_dir + f'/{tn}.txt', 'r') as f:
        dst_vocab_token2index[tn] = dict({x.strip('\n'): idx + 1 for idx, x in enumerate(f.readlines())})
        dst_vocab_token2index[tn]['@@PADDING@@'] = 0

with open(fil, 'r') as f:
    lines = f.readlines()

QUES_MAX_LEN = 160 + 2
ANS_MAX_LEN = 30 + 3
PROG_MAX_LEN = 30 + 3
N = len(lines) // 2
Q = np.zeros((N, QUES_MAX_LEN + 1), dtype=np.ubyte)
A = np.zeros((N, ANS_MAX_LEN + 1), dtype=np.ubyte)
P = np.zeros((N, PROG_MAX_LEN + 1), dtype=np.ubyte)
get_src_idx = lambda x: src_vocab_token2index[x]
get_dst_idx = {k: dst_vocab_token2index[k].__getitem__ for k in dst_vocab_token2index}

for idx in tqdm(range(len(lines) // 2)):
    ques = lines[2 * idx].strip()
    ans, prog = lines[2 * idx + 1].strip().split('###')
    prog = prog.split()
    assert len(ques) <= QUES_MAX_LEN
    assert len(ans) <= ANS_MAX_LEN
    assert len(prog) <= PROG_MAX_LEN
    Q[idx][:len(ques)] = list(map(get_src_idx, ques))
    Q[idx][-1] = len(ques)
    A[idx][:len(ans) + 2] = [dst_vocab_token2index["answer"]["@start@"]] + list(map(get_dst_idx["answer"], ans)) + [
        dst_vocab_token2index["answer"]["@end@"]]
    A[idx][-1] = len(ans) + 2
    P[idx][:len(prog) + 2] = [dst_vocab_token2index["program"]["@start@"]] + list(map(get_dst_idx['program'], prog)) + [
        dst_vocab_token2index["program"]["@end@"]]
    P[idx][-1] = len(prog) + 2

data = {"Q": Q, "A": A, "P": P}

if Q.shape[0] > 10 ** 7:
    format = 'npz'
else:
    format = 'npy'

if fil.endswith('.txt'):
    save_path = fil[:-4] + '.' + format
else:
    save_path = f'{fil}.{format}'
if format == 'npy':
    np.save(save_path, data)
else:
    np.savez(save_path, Q=Q, A=A, P=P)
