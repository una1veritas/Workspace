# coding: utf-8
'''
Created on 2020/10/28

@author: Sin Shimozono
'''

import sys, math, os, copy
from bitstream import BitStream

def nibblestream(fp, buffer_size = 32):    
    buff = fp.read(buffer_size)
    while buff :
        for ch in buff:
            for i in range(2):
                yield (0xf0 & (ch<<(4*i)))>>4
        buff = fp.read(buffer_size)
    
def bytestream(fp, buffer_size = 32) :
    buff = fp.read(buffer_size)
    while buff :
        for ch in buff:
            yield ch
        buff = fp.read(buffer_size)

def commonhighbits(l, r, w): 
    bmask = 1<<(w-1)
    bdigits = ''
    while bmask :
        if (l & bmask) != (r & bmask) :
            break
        bdigits += '1' if (l & bmask) != 0 else '0' 
        bmask >>= 1 
    return bdigits

def histgram(hist, block, enhance = True):
    if enhance :
        for c in block:
            hist[c] += 1
        return hist
    for c in block:
        hist[c] += 1
    for i in range(len(hist)):
        if hist[i] > 1 :
            hist[i] >>= 1
    if sum(hist) > len(block) :
        i = 0
        while sum(hist) > len(block) :
            if hist[i] > 1 :
                hist[i] -= 1
            i = (i + 1) % len(hist)
    elif sum(hist) < len(block) :
        i = 0
        while sum(hist) < len(block) :
            if hist[i] > 1 :
                hist[i] += 1
            i = (i + 1) % len(hist)
    #print(sum(hist))
    return hist
        
def subbitseq(val, bitdepth, index_begin, index_end):
    mask = 1 << (bitdepth - 1)
    result = ''
    for i in range(bitdepth):
        if index_end <= i :
            break
        if index_begin <= i :
            if (val & mask) :
                result += '1'
            else:
                result += '0'
        mask >>= 1
    return result
    
def encode_block(block, bitsize, sections, left = 0, right = 1, bits = 0):
    codebytes = b''
    bitbuffer = ''
    for i in range(len(block)) :
        ch = block[i]
        l, r = sections[ch], sections[ch+1]
        width = right - left
        right = (left<<bitsize) + (width * r)
        left = (left<<bitsize) + (width * l)
        bits += bitsize
    # print( '[{}, {}), '.format(left/(1<<bits), right/(1<<bits)))
        hbits = commonhighbits(left, right, bits)
        if len(hbits) :
            mask = (1<<bits) - 1
            mask >>= len(hbits)
            left &= mask
            right &= mask
            bits -= len(hbits)
            bitbuffer += hbits
            #print('{} '.format(hbits),end='')
            while (left & 1 == 0) and (right & 1 == 0) :
                left >>= 1
                right >>= 1
                bits -= 1
            if len(bitbuffer) // 8 > 0:
                bytenum = len(bitbuffer) // 8
                codebytes += int(bitbuffer[:bytenum*8],2).to_bytes(bytenum, byteorder='big')
                bitbuffer = bitbuffer[bytenum*8:]
        #print('[{},{})'.format(format(left,'b')[:8], format(right,'b')[:8]), end='')
        
    #print()
    #print('last char {}, remained {}, {} bits'.format(hex(ch), bitbuffer, bits), subbitseq(left, bits, 0, 8), subbitseq(right, bits, 0, 8))
    if len(bitbuffer) :
        bitbuffer = bitbuffer.ljust(8, '0')
        #print(bitbuffer)
        codebytes += int(bitbuffer,2).to_bytes(bytenum, byteorder='big')
    return codebytes

def decode_block(block, bitsize, sections, left = 0, right = 1, bits = 0):
    codebytes = b''
    bitbuffer = ''
    for i in range(len(block)) :
        ch = block[i]
        l, r = sections[ch], sections[ch+1]
        width = right - left
        right = (left<<bitsize) + (width * r)
        left = (left<<bitsize) + (width * l)
        bits += bitsize

def decoder(infile, outfile = None):
    byte_bitsize = 8
    print(infile)
    hist = [1 for i in range(1<<byte_bitsize)]
    try:
        block_size_bits = 8
        block_size_bits_max = 16
        with open(infile, "rb") as ifp:
            while True :
                buff = ifp.read(1<<block_size_bits)
                if not len(buff) :
                    break
                code_range = [0]
                range_width = sum(hist)
                subsum = 0
                for i in range(len(hist)):
                    subsum += hist[i]
                    code_range.append(subsum)
                bitstream = ''
                value = 0
                bitdepth = 0
                bpos = 0
                left = 0
                while bpos < (len(buff)<<3) :
                    abit = 0x01 & ((buff[bpos>>3])>>(7-(bpos&0x07)))
                    bitstream += str(abit)
                    value <<= 1
                    value |= abit
                    bitdepth += 1
                    while True :
                        if not (value < code_range[left]) :
                            break
                        left += 1
                    right = left
                    while value >= code_range[right] :
                        right += 1
                    if left + 1 == right :
                        print(value)
                        left = 0
                print(len(bitstream))
                    
                if len(buff) != (1<<block_size_bits) :
                    break
                else:
                    block_size_bits += 1
                break
    except IOError:
        print('file "' + infile + '" open failed.')
        exit()
    
def encoder(infile, outfile = None):
    byte_bitsize = 8
    print(infile)
    hist = list()
    # the initial histogram
    for i in range(1<<byte_bitsize):
        hist.append(1)
    codebytes = b''
    try:
        block_size_bits = 8
        block_size_bits_max = 16

        if outfile :
            #let outfile be the empty binary file.
            with open(outfile, "wb") as ofp:
                pass
        
        with open(infile, "rb") as fp:
            while True :
                buff = fp.read(1<<block_size_bits)
                # print('block_size = {}'.format(block_size) )
                # setup the code_range
                subsum = 0
                code_range = [0]
                for i in range(len(hist)):
                    subsum += hist[i]
                    code_range.append(subsum)
                # encode
                #print('encoding')
                block_codebytes = encode_block(buff, block_size_bits, code_range, left = 0, right = 1, bits = 0)
                if outfile :
                    with open(outfile, "ab") as ofp:
                        ofp.write(block_codebytes)
                else:
                    codebytes += block_codebytes
                #print(block_codebytes)
                print('block code bytes '+str(len(block_codebytes)), 'block size '+str(1<<block_size_bits))
                
                # if the block is full (not the final block)
                if len(buff) == (1<<block_size_bits) :
                    if block_size_bits < block_size_bits_max :
                        histgram(hist, buff, enhance = True)
                        print('histogram enhanced to '+str(sum(hist)))
                        block_size_bits += 1
                    else:
                        histgram(hist, buff, enhance = False)
                        print('histogram modified.')
                else:
                    # termination condition
                    print(sum(hist), 1<<block_size_bits)
                    break
    except IOError:
        print('file "' + infile + '" open failed.')
        exit()
    print('hist_total = {}'.format(sum(hist)))
    for i in range(1<<byte_bitsize):
        if hist[i] > 1 :
            print( (chr(i) if chr(i).isprintable() else hex(i) ), hist[i])
    
    print("input  file size: {}".format(os.path.getsize(infile)))
    if outfile :
        print("output file size: {}".format(os.path.getsize(outfile)))
    else:
        print('output array size: {}'.format(len(codebytes)))
    print('finished.')
    
if __name__ == "__main__" :
    infile = sys.argv[1]
    outfile = None
    if len(sys.argv) >= 3 :
        outfile = sys.argv[2]
    encoder(infile, outfile)