#
# -*- coding: utf-8 -*-
import sys

count_one = dict()
for i in range(0, 2**4) :
    bin_str = format(i, '04b')
    cnt = bin_str.count('1')
    if cnt in count_one :
        count_one[cnt].append(i)
    else:
        count_one[cnt] = [i]

for cnt in range(0, 4+1):
    print([format(each, '04b') for each in count_one[cnt] ] )