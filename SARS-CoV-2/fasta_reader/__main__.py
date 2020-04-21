#
import sys

def lcsgap(seq_a, seq_b, gaplimit = 1000, alignment = False):
    # make sure _a is longer than or equal
    if len(seq_a) < len(seq_b) :
        seq_a, seq_b = seq_b, seq_a
#    dpt = [[0 for c in range(0, len(seq_a) + 1)] for r in range(0,4)]
    dpt = dict()
    #print('go.', file=sys.stderr)
    for r in range(1, len(seq_b) + 1):
        #if (r % 2000) == 0 :
        #    print('.', file=sys.stderr, flush=True, end='')
        for c in range(max(1, r - gaplimit), 
                       min(len(seq_a) - (len(seq_b) - gaplimit) + r, len(seq_a) + 1)):
            if seq_a[c-1] == seq_b[r-1] :
                #dpt[r & 3][c] = dpt[(r-1) & 3][c-1] + 1
                dpt[(r,c)] = dpt.get((r-1,c-1), 0) + 1
            else:
                #dpt[r & 3].append(max(dpt[(r-1) & 3][c], dpt[r & 3][c-1], dpt[r-1][c-1]))
                r_1_c = dpt.get((r-1,c), 0); t_c_1 = dpt.get((r,c-1), 0)
                if max(r_1_c, t_c_1) != 0 :
                    dpt[(r,c)] = max(r_1_c, t_c_1)
#               dpt[(r,c)] = max(dpt[(r-1)&3][c], dpt[r&3][c-1]) #, dpt[(r-1)&3][c-1])
#   return ( dpt[len(seq_b) & 3][len(seq_a)], len(seq_a), len(seq_b) )
    if not alignment :
        return dpt[(len(seq_b), len(seq_a))]

    #back tracking
    align = ''; r = len(seq_b); c = len(seq_a)
    while r > 0 and c > 0 :
        #print(r,c,dpt[(r,c)],  flush = True)
        if (r,c) not in dpt :
            align = ('-'*r ) + align
            break
        if dpt.get((r-1,c-1), 0) == dpt[(r,c)] :
            align = '-' + align
            r -= 1; c -= 1
        elif dpt.get((r-1,c), 0) == dpt[(r,c)] :
            r -= 1
        elif dpt.get((r,c-1), 0) == dpt[(r,c)] :
            align = '-' + align
            c -= 1
        else:
            if dpt.get((r-1,c-1), 0) + 1 == dpt[(r,c)]:
                align = seq_a[c-1] + align
                r -= 1; c -= 1
            else:
                print('align error at ({0},{1})'.format(r,c), file=sys.stderr)
    return (dpt[(len(seq_b), len(seq_a))], align)


def read_fasta(filename, key = None, longerthan = 28000):
    print('reading', end='', file=sys.stderr, flush=True)
    seqdb = list()
    lcount = 0
    passflag = True
    with open(filename) as infile:
        a_tuple = list()
        strbuf = ''
        while True:
            a_line = infile.readline()
            if key is None or key in a_line :
                passflag = False
            if len(a_line) == 0 or a_line[0] == '>':
                if len(a_tuple) != 0 and len(strbuf) > longerthan and not passflag:
                    a_tuple.append(strbuf.upper())
                    seqdb.append(tuple([a_tuple[1],a_tuple[2],a_tuple[0],a_tuple[3]]))
                    passflag = True
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
    return seqdb


if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    print('usage: filename key1 key2', file=sys.stderr)
    exit()
 
seqdb = read_fasta(fname, key='Japan', longerthan = 28000)
#
#for a_record in seqdb:
#    print('{0},{1},{2},{3}'.format(a_record[0], a_record[1], a_record[2], a_record[3]))
#print('finished.')
for i in range(0, len(seqdb)) :
    for j in range(i+1, len(seqdb)):
        seq_1 = seqdb[i]; seq_2 = seqdb[j]
        maxlen = max(len(seq_1[3]), len(seq_2[3]))
        #lcslen = lcs(seqs[0], seqs[1], gaplimit)
        print(seq_1[0], end=',')
        print(seq_2[0], end=',')
        gaplen, lcsstr = lcsgap(seq_1[3], seq_2[3], gaplimit = int(maxlen*0.1), alignment = True)
        print(float(gaplen)/maxlen,  end=',')
        print(lcsstr)
        print('{0} ({1}), {2} ({3}): {4}'.format(i, len(seq_1[3]), j, len(seq_2[3]), float(gaplen)/maxlen), file=sys.stderr)
        #print('the length of longest common subsequence = ', lcslen)
        #print(str(seqdict.keys()) + ' similarity =', float(lcslen)/min(len(seqs[0]), len(seqs[1])))
