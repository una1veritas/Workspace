'''
Created on 2020/10/28

@author: Sin Shimozono
'''

import sys, math, os

def bitstream(fp, buffer_size = 32):    
    buff = fp.read(buffer_size)
    while buff :
        for ch in buff:
            for bp in range(0,8):
                yield 1 if (ch & 0x80) else 0
                ch <<= 1
        buff = fp.read(buffer_size)
    
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

def samehighbits(l, r, w): 
    bmask = 1<<(w-1)
    bdigits = ''
    while bmask :
        if (l & bmask) != (r & bmask) :
            break
        bdigits += '1' if (l & bmask) != 0 else '0' 
        bmask >>= 1 
    return bdigits
   
def update_histgram(hist, block, enhance = True):
    for c in block:
        hist[c] += 1
    if enhance :
        return hist
    carry = 0
    for i in range(len(hist)):
        if hist[i] > 1 :
            hist[i] >>= 1
    while sum(hist) != len(block) :
        i = 0
        while sum(hist) > len(block):
            if hist[i] > 1:
                hist[i] -= 1
            i += 1
            if not i < len(hist) :
                break
        i = 0
        while sum(hist) < len(block):
            if hist[i] > 1:
                hist[i] += 1
            i += 1
            if not i < len(hist) :
                break
    print('sum hist {}'.format(sum(hist)))
#    print('carry = ' + str(carry))
    return hist

def encode_block(block, bitsize, sections, left = 0, right = 1, bits = 0):
    codestr = ''
    for i in range(len(block)) :
        ch = block[i]
        l, r = sections[ch], sections[ch+1]
        width = right - left
        right = (left<<bitsize) + (width * r)
        left = (left<<bitsize) + (width * l)
        bits += bitsize
    # print( '[{}, {}), '.format(left/(1<<bits), right/(1<<bits)))
        hbits = samehighbits(left, right, bits)
        if len(hbits) :
            mask = (1<<bits) - 1
            mask >>= len(hbits)
    # print(mask)
            left &= mask
            right &= mask
            bits -= len(hbits)
            codestr += hbits
    #    print('{}:[{:b}, {:b}) ({})'.format(hbits, left, right, bits))
    #                         print(encodedstr)
    # #                     if (left & 0xff == 0) and (right & 0xff == 0) :
    # #                         left >>= 8
    # #                         right >>= 8
    # #                         bits -= 8
    codestr += '1'
    return codestr

def main(argv = None):
    infile = argv[1]
    if len(argv) >= 3 :
        outfile = argv[2]
    else:
        outfile = 'test.brc'
    byte_bitsize = 8
    print(infile)
    hist = list()
    # the initial histogram
    for i in range(1<<byte_bitsize):
        hist.append(1)
    
    try:
        block_size_bits = 8
        block_size_bits_max = 13
        
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
                encodedstr = encode_block(buff, block_size_bits, code_range, left = 0, right = 1, bits = 0)
                with open(outfile, "ab") as ofp:
                    for i in range(len(encodedstr)>>3):
                        binstr = encodedstr[i*8:(i+1)*8]
                        ofp.write(int(binstr,2).to_bytes(1, 'little'))
                print(len(encodedstr), 1<<block_size_bits)
                
                # if the block is full (not the final block)
                if len(buff) == (1<<block_size_bits) :
                    if block_size_bits < block_size_bits_max :
                        update_histgram(hist, buff, enhance = True)
                        print('enhanced histogram size to '+str(sum(hist)))
                        block_size_bits += 1
                    else:
                        update_histgram(hist, buff, enhance = False)
                else:
                    # termination condition
                    print(sum(hist), 1<<block_size_bits)
                    break
    except IOError:
        print('file "' + infile + '" open failed.')
        exit()
    print('hist_total = {}'.format(sum(hist)))
    for i in range(1<<byte_bitsize):
        print( (chr(i) if chr(i).isprintable() else hex(i) ), hist[i])
    exit()
        
    ranges = list()
    subsum = 0
    for i in range(0,len(hist)):
        ranges.append( (subsum, subsum + hist[i]) )
        subsum += hist[i]
#     for ch in range(256):
#         (l, r) = ranges[ch]
#          if l != r :
#              print( chr(ch) if chr(ch).isprintable() else hex(ch) , r - l)
    
    with open(infile, "rb") as fp:
        code = list()
        for a_byte in bytestream(fp, 256):
            #print(ranges[a_byte])
            code.append(ranges[a_byte])
    code.append(ranges[0])
    print('file size = '+str(len(code)))
    left = 0
    right = 1
    bits = 0
    encodedstr = ''
    for (l, r) in code[:4096] :
#        print('---–\n{0}\n{1}\n'.format(format(left, 'b').zfill(bits), format(right, 'b').zfill(bits)))
        hbits = samehighbits(left, right, bits)
        if len(hbits) :
            mask = (1<<bits) - 1
            mask >>= len(hbits)
            left &= mask
            right &= mask
            bits -= len(hbits)
#             print('{}.{}\n{}.{}\n({}:{})\n'.format(hbits, format(left,'b').zfill(bits), hbits, format(right, 'b').zfill(bits), len(hbits), bits))
            encodedstr += hbits
        if (left & 0xff == 0) and (right & 0xff == 0) :
            left >>= 8
            right >>= 8
            bits -= 8
    else:
        encodedstr += '1'

    #print('{}\n{}'.format(bin(right), len(bin(right))))
    print(encodedstr)
    print(len(encodedstr))
    print('finished.')
    
if __name__ == "__main__" :
    main(sys.argv)