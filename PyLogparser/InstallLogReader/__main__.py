#
#
import sys
import re

params = dict()
if sys.argv[1][0] == '-' :
    [attr, val] = sys.argv[1][1:].split('=')
    params[attr] = val

if not 'file' in params:
    print(params)
    exit()

date_pattern = re.compile('\d\d\d\d-\d\d-\d\d')
time_pattern = re.compile('\d\d:\d\d:\d\d')
dtdict = dict()
with open(params['file'], 'r', encoding='utf-8') as file:
    for a_line in file:
        a_line = a_line.strip().split(' ')
        if len(a_line) < 2 :
            continue
        m_date = date_pattern.match(a_line[0])
        m_time = time_pattern.match(a_line[1])
        if m_date != None and m_time != None :
            t_date = m_date.group()
            t_time = m_time.group()
            if not t_date in dtdict:
                dtdict[t_date] = [ t_time ]
            else:
                if len(dtdict[t_date]) == 1:
                    if dtdict[t_date][0] < t_time :
                        dtdict[t_date].append(t_time)
                    elif dtdict[t_date][0] > t_time :
                        dtdict[t_date] = [t_time].append(dtdict[t_date])
                elif dtdict[t_date][0] > t_time :
                    dtdict[t_date][0] = t_time
                elif dtdict[t_date][1] < t_time :
                    dtdict[t_date][1] = t_time
for d in dtdict:
    print(d, dtdict[d])
print('finished.')
