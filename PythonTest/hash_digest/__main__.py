# encoding: utf-8
'''
@author:     user_name

@copyright:  2020 organization_name. All rights reserved.

@license:    license

@contact:    user_email
@deffield    updated: Updated
'''

import sys
import hashlib

def hashstr(sh):
    strbuff = ''
    for i in range(len(sh.hexdigest())>>1) :
        ch = int(sh.hexdigest()[2*i:2*(i+1)],16)
        if chr(ch).isprintable() and chr(ch).isascii() :
            strbuff += chr(ch)
        else:
            strbuff += format(ch,'X')
    return strbuff

def main(argv=None): # IGNORE:C0111
    '''Command line options.'''
    keyword = argv[1]
    sh = hashlib.sha256(keyword.encode())
    print(hashstr(sh))
    sh = hashlib.md5(keyword.encode())
    print(hashstr(sh))
    
if __name__ == "__main__":
    sys.exit(main(sys.argv))