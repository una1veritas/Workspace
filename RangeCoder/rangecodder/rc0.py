'''
Created on 2020/10/28

@author: Sin Shimozono
'''
# coding: utf-8
#
# rc0.py : レンジコーダ
#
#          Copyright (C) 2007 Makoto Hiroi
#
import time, sys, getopt, os.path
from rangecoder import *

# 出現頻度表
class Freq:
    def __init__(self, count):
        size = len(count)
        self.size = size
        m = max(count)
        # 2 バイトに収める
        if m > 0xffff:
            self.count = [0] * size
            n = 0
            while m > 0xffff:
                m >>= 1
                n += 1
            for x in xrange(size):
                if count[x] != 0:
                    self.count[x] = (count[x] >> n) | 1
        else:
            self.count = count[:]
        self.count_sum = [0] * (size + 1)
        # 累積度数表
        for x in xrange(size):
            self.count_sum[x + 1] = self.count_sum[x] + self.count[x]

    #
    def write_count_table(self, fout):
        for x in self.count:
            putc(fout, x >> 8)
            putc(fout, x & 0xff)
    
    # 符号化
    def encode(self, rc, c):
        temp = rc.range / self.count_sum[self.size]
        rc.low += self.count_sum[c] * temp
        rc.range = self.count[c] * temp
        rc.encode_normalize()

    # 復号
    def decode(self, rc):
        # 記号の探索
        def search_code(value):
            i = 0
            j = self.size - 1
            while i < j:
                k = (i + j) / 2
                if self.count_sum[k + 1] <= value:
                    i = k + 1
                else:
                    j = k
            return i
        #
        temp = rc.range / self.count_sum[self.size]
        c = search_code(rc.low / temp)
        rc.low -= temp * self.count_sum[c]
        rc.range = temp * self.count[c]
        rc.decode_normalize()
        return c

# ファイルの読み込み
def read_file(fin):
    while True:
        c = getc(fin)
        if c is None: break
        yield c

# レンジコーダによる符号化
def encode(fin, fout):
    count = [0] * 256
    for x in read_file(fin):
        count[x] += 1
    rc = RangeCoder(fout, ENCODE)
    freq = Freq(count)
    freq.write_count_table(fout)
    fin.seek(0)
    for x in read_file(fin):
        freq.encode(rc, x)
    rc.finish()

# 出現頻度表の読み込み
def read_count_table(fin):
    count = [0] * 256
    for x in xrange(256):
        count[x] = (getc(fin) << 8) + getc(fin)
    return count

# レンジコーダによる復号
def decode(fin, fout, size):
    freq = Freq(read_count_table(fin))
    rc = RangeCoder(fin, DECODE)
    for _ in xrange(size):
        putc(fout, freq.decode(rc))

# 符号化
def encode_file(name1, name2):
    size = os.path.getsize(name1)
    infile = open(name1, "rb")
    outfile = open(name2, "wb")
    putc(outfile, (size >> 24) & 0xff)
    putc(outfile, (size >> 16) & 0xff)
    putc(outfile, (size >> 8) & 0xff)
    putc(outfile, size & 0xff)
    if size > 0: encode(infile, outfile)
    infile.close()
    outfile.close()

# 復号
def decode_file(name1, name2):
    infile = open(name1, "rb")
    outfile = open(name2, "wb")
    size = 0
    for _ in xrange(4):
        size = (size << 8) + getc(infile)
    if size > 0: decode(infile, outfile, size)
    infile.close()
    outfile.close()

#
def main():
    eflag = False
    dflag = False
    opts, args = getopt.getopt(sys.argv[1:], 'ed')
    for x, y in opts:
        if x == '-e' or x == '-E':
            eflag = True
        elif x == '-d' or x == '-D':
            dflag = True
    if eflag and dflag:
        print 'option error'
    elif eflag:
        encode_file(args[0], args[1])
    elif dflag:
        decode_file(args[0], args[1])
    else:
        print 'option error'

#
s = time.clock()
main()
e = time.clock()
print "%.3f" % (e - s)
