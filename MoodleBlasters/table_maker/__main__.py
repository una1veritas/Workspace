import io,os,sys
import glob

path = '/Users/Sin/Dropbox/アルゴリズム設計Ａ/202104/data/C11L202-2021-0101-1121010101-'

namelistfilename = path+'namelist.txt'
if not os.path.isfile(namelistfilename) :
    print('Not found name list file.')
    exit()
dirlist = list()
wldcard='*'
for p in glob.glob(path+wldcard):
    if os.path.isdir(p):
        dirlist.append(p)

namelist = list()
with open(namelistfilename) as f:
    for l in f.readlines():
        namelist.append(l.strip().split('\t'))
namelist[1:].sort(key=(lambda x: x[0]))
#print(namelist)

presence = dict()
for dirname in dirlist:
    dt = dirname.split('/')[-1].split('-')[-1]
    dt = dirname.split('/')[-1].split('-')[1] + '-' + dt[:2] + '-' + dt[2:]
    presence[dt] = list()
    for subm in os.listdir(dirname):
        sid = subm.split('_')[0]
        presence[dt].append(sid)

table = dict()
columns = list([namelist[0][0]])
for stu in namelist[1:]:
    table[stu[0]] = list()

#print(columns)
#print(table)
    
for dt in sorted(presence.keys()):
    columns.append(dt)
    for student in namelist[1:]:
        sid = student[0]
        if sid in presence[dt]:
            table[sid].append(u'出')
        else:
            table[sid].append(u'欠')

with open(path+'table.txt', mode='w') as outf:
    outf.write('\t'.join(columns)+'\n')
    for ea in sorted(table.items()):
        outf.write(ea[0]+'\t')
        outf.write('\t'.join(ea[1]))
        outf.write('\n')

print('Written to ' + path+'table.txt')