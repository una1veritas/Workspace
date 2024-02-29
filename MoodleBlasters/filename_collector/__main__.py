'''
Created on 2024/02/28

@author: Sin Shimozono
'''
import sys, os, glob

if __name__ == '__main__':
    path = sys.argv[1]
    if len(sys.argv) < 2 :
        print("No directory path.")
        exit(1)
    #print(path)
    if not os.path.isdir(path) :
        print("Not a directory.")
        exit(1)
    sizedict = dict()
    for f in glob.glob(os.path.join(path,"*")):
        fbase = os.path.basename(f)
        ext = fbase.split('.')[-1]
        fnamelist = fbase.split('_')
        fid = fnamelist[0]
        fsize = os.path.getsize(f)
        if fid in sizedict:
            sizedict[fid] = max(sizedict[fid], fsize)
        else:
            sizedict[fid] = fsize
    for sid in sorted(sizedict.keys()):
        print(sid)