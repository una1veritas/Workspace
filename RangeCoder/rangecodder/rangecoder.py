'''
Created on 2020/10/28

@author: Sin Shimozono
'''
##http://www.nct9.ne.jp/m_hiroi/light/pyalgo36.html
# coding: utf-8
#
# rangecoder.py : レンジコーダ (Range Coder)
#
#                 Copyright (C) 2007 Makoto Hiroi
#

# 定数
ENCODE = "encode"
DECODE = "decode"
MAX_RANGE = 0x100000000
MIN_RANGE = 0x1000000
MASK      = 0xffffffff
SHIFT     = 24

# バイト単位の入出力
def getc(f):
    c = f.read(1)
    if c == '': return None
    return ord(c)

def putc(f, x):
    f.write(chr(x & 0xff))

#
class RangeCoder:
    def __init__(self, file, mode):
        self.file = file
        self.range = MAX_RANGE
        self.buff = 0
        self.cnt = 0
        if mode == ENCODE:
            self.low = 0
        elif mode == DECODE:
            # buff の初期値 (0) を読み捨てる
            getc(self.file)
            # 4 byte read
            self.low = getc(self.file)
            self.low = (self.low << 8) + getc(self.file)
            self.low = (self.low << 8) + getc(self.file)
            self.low = (self.low << 8) + getc(self.file)
        else:
            raise "RangeCoder mode error"

    # 符号化の正規化
    def encode_normalize(self):
        if self.low >= MAX_RANGE:
            # 桁上がり
            self.buff += 1
            self.low &= MASK
            if self.cnt > 0:
                putc(self.file, self.buff)
                for _ in xrange(self.cnt - 1): putc(self.file, 0)
                self.buff = 0
                self.cnt = 0
        while self.range < MIN_RANGE:
            if self.low < (0xff << SHIFT):
                putc(self.file, self.buff)
                for _ in xrange(self.cnt): putc(self.file, 0xff)
                self.buff = (self.low >> SHIFT) & 0xff
                self.cnt = 0
            else:
                self.cnt += 1
            self.low = (self.low << 8) & MASK
            self.range <<= 8

    # 復号の正規化
    def decode_normalize(self):
        while self.range < MIN_RANGE:
            self.range <<= 8
            self.low = ((self.low << 8) + getc(self.file)) & MASK

    # 終了
    def finish(self):
        c = 0xff
        if self.low >= MAX_RANGE:
            # 桁上がり
            self.buff += 1
            c = 0
        putc(self.file, self.buff)
        for _ in xrange(self.cnt): putc(self.file, c)
        #
        putc(self.file, (self.low >> 24) & 0xff)
        putc(self.file, (self.low >> 16) & 0xff)
        putc(self.file, (self.low >> 8) & 0xff)
        putc(self.file, self.low & 0xff)
