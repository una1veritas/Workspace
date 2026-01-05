'''
Created on 2026/01/04

@author: sin
'''
import random

def shuffle16(val16, shuffled = [11, 1, 14, 9, 5, 10, 8, 3, 0, 12, 15, 7, 2, 6, 13, 4]): 
    val16 = val16^0x49d6 + 0x3ba9
    res16 = 0
    for i in shuffled :
        res16 <<= 1
        res16 |= (val16>>shuffled[i]) & 1
    return res16 & 0xffff


if __name__ == '__main__':
    occurred = set(range(0x10000))
    #shuffle_seq = list(range(16))
    #random.shuffle(shuffle_seq)
    for i in range(0, 0x10000) :
        shuffled = shuffle16(i)
        occurred.add(shuffled)
        print(f'{shuffled:04x}')
    print(len(occurred))