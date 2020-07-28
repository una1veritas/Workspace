#!/usr/local/bin/python3
# encoding: utf-8
'''
install_log_reader.__main__ 
'''

import sys
import re

params = dict()
print(sys.argv)
params['file'] = sys.argv[1]

date_pattern = re.compile('\d\d\d\d-\d\d-\d\d')
time_pattern = re.compile('\d\d:\d\d:\d\d\+\d\d')

times24 = dict()
with open(params['file'], 'r', encoding='utf-8') as file:
    for a_line in file:
        t = a_line.strip()
        if len(t) == 0 :
            continue
        d_matched = date_pattern.search(t)
        t_matched = time_pattern.search(t)
        if d_matched != None and t_matched != None:
            d_str = d_matched.group()
            t_str = t_matched.group()
            if t_str.endswith('+09') :
                t_str = t_str.split('+')[0]
            else:
                raise ValueError('Time Zone Error!')
            if d_str in times24 :
                if t_str < times24[d_str][0] :
                    if len(times24[d_str]) == 2 :
                        times24[d_str][0] = t_str
                    else:
                        times24[d_str].insert(0,t_str)
                elif t_str > times24[d_str][-1] :
                    if len(times24[d_str]) == 2 :
                        times24[d_str][1] = t_str
                    else:
                        times24[d_str].append(t_str)                
            else: #d_str not in times24
                times24[d_str] = [t_str]
        #2020-06-10 12:28:05+09 
# if __name__ == "__main__":
#     sys.exit(main())

for each in times24:
    print(each, times24[each])
