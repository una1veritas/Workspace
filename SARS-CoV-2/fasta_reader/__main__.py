#
import sys

filename = sys.argv[1]

seqdb = list()
lcount = 0
with open(filename) as infile:
    a_tuple = list()
    strbuf = ''
    for a_line in infile:
        if a_line[0] == '>' :
            if len(a_tuple) != 0 :
                a_tuple.append(strbuf)
                seqdb.append(a_tuple)
                lcount += 1
                if (lcount % 1000) == 0 :
                    print(str(lcount)+'...')
            a_tuple = a_line.strip()[1:].split('|')
            strbuf = ''
        else:
            strbuf += a_line.strip()
    else:
        if len(a_tuple) != 0 :
            a_tuple.append(strbuf)
            seqdb.append(a_tuple)
            lcount += 1

print('reading done.')
print('got ' + str(lcount) + ' sequences.')
cnt = 0
for a_record in sorted(seqdb, key=lambda r:r[1]):
    cnt += 1
    if cnt > 10 and cnt < 9170:
        continue
    print(a_record[0]+','+a_record[1]+','+a_record[2]+','+a_record[3][:24])
print('finished.')
