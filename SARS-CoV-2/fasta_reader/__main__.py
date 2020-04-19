#
import sys
from _ast import If

def lcs(seq_a, seq_b, matchwidth):
    # make sure _a is longer than or equal
    if len(seq_a) < len(seq_b) :
        seq_a, seq_b = seq_b, seq_a
#    dpt = [[0 for c in range(0, len(seq_a) + 1)] for r in range(0,4)]
    dpt = dict()
    print('go.', file=sys.stderr)
    for r in range(1, len(seq_b) + 1):
        if (r % 2000) == 0 :
            print('.', file=sys.stderr, flush=True, end='')
        for c in range(max(1, r - matchwidth), 
                       min(len(seq_a) - (len(seq_b) - matchwidth) + r, len(seq_a) + 1)):
            if seq_a[c-1] == seq_b[r-1] :
                #dpt[r & 3][c] = dpt[(r-1) & 3][c-1] + 1
                if (r-1, c-1) not in dpt :
                    dpt[(r,c)] = 1
                else:
                    dpt[(r,c)] = dpt[(r-1,c-1)] + 1
            else:
                #dpt[r & 3].append(max(dpt[(r-1) & 3][c], dpt[r & 3][c-1], dpt[r-1][c-1]))
                if (r-1, c) not in dpt:
                    r_1_c = 0
                else:
                    r_1_c = dpt[(r-1,c)]
                if (r, c-1) not in dpt:
                    r_c_1 = 0
                else:
                    r_c_1 = dpt[(r,c-1)]
                if max(r_1_c, r_c_1) != 0 :
                    dpt[(r,c)] = max(r_1_c, r_c_1)
#               dpt[(r,c)] = max(dpt[(r-1)&3][c], dpt[r&3][c-1]) #, dpt[(r-1)&3][c-1])
#   return ( dpt[len(seq_b) & 3][len(seq_a)], len(seq_a), len(seq_b) )
    print()
    return ( dpt[(len(seq_b), len(seq_a))], len(seq_a), len(seq_b) )


if len(sys.argv) == 4 :
    keys = sys.argv[1:3]
    filename = sys.argv[3]
    print(keys, filename)
else:
    print('usage: key1, key2, filename', file=sys.stderr)
    exit()

print('reading', end='', file=sys.stderr, flush=True)
seqdb = list()
min_seqlen = 19000
lcount = 0
with open(filename) as infile:
    a_tuple = list()
    strbuf = ''
    while True:
        a_line = infile.readline()
        if len(a_line) == 0 or a_line[0] == '>':
            if len(a_tuple) != 0 :
                a_tuple.append(strbuf.upper())
                seqdb.append(tuple([a_tuple[1],a_tuple[2],a_tuple[0],a_tuple[3]]))
                lcount += 1
                if (lcount % 2000) == 0 :
                    print('.', end='', file=sys.stderr, flush=True)
            if len(a_line) == 0 :
                break
            a_tuple = a_line.strip()[1:].split('|')
            strbuf = ''
        else:
            strbuf += a_line.strip()

print(' done.', file=sys.stderr)
print('got ' + str(lcount) + ' sequences.', file=sys.stderr)

#
#for a_record in seqdb:
#    print('{0},{1},{2},{3}'.format(a_record[0], a_record[1], a_record[2], a_record[3]))
#print('finished.')

seqdict = dict()
for i in range(0, len(seqdb)):
#    if len(seqdb[i][3]) < 29900 * 0.97 :
#        continue
    if seqdb[i][0] in keys :
        if seqdb[i][0] not in seqdict :
            seqdict[seqdb[i][0]] = i

seqs = [seqdb[i][3] for i in seqdict.values()]
matchwidth = int(min(len(seqs[0]), len(seqs[1])) * 0.005)
lcslen, len_longer, len_shorter = lcs(seqs[0], seqs[1], matchwidth)

print('the length of longest common subsequence = ', lcslen)
print(str(seqdict.keys()) + ' similarity =', float(lcslen)/len_shorter)
