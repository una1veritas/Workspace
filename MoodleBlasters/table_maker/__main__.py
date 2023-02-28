import os, glob, sys

path = ""
namelist = ""
if len(sys.argv) <= 2 :
    print('arguments: [base directory] [csv name list file]', file=sys.stderr)
    exit(1)
else:
    if os.path.isdir(sys.argv[1]) :
        path = sys.argv[1]
    else:
        path = os.path.dirname(sys.argv[1])
    if os.path.isfile(sys.argv[2]):
        namelist = sys.argv[2]
    else:
        print("file "+sys.argv[2]+ " does not exist.")
        exit(1)
print("path = " + path)
print("name list file = " + namelist)

print('Reading the list of registered students from '+namelist)
db_registered = list()
COMMENTESCAPE = '#'
separator = ','
with open(namelist, encoding = "utf-8-sig") as f:
    for l in f.readlines():
        l = l.strip('"')
        if l[0] == COMMENTESCAPE : continue
        fields = l.split(separator)
        # (sid, namestr, dep, grade)
        db_registered.append( (fields[2], fields[1], fields[4], fields[6]) )
        #print(fields)
print()

db_registered.sort(key=(lambda x: x[0]))
#for i in db_registered:
#    print(i)
print('Found ' + str(len(db_registered))+' names of students.')
for row in db_registered[:5] :
    print(row)
else:
    print("...", end="\n\n")

print('Collecting assignment folders...')
db_folders = list()
for fdname in glob.glob(os.path.join(path, '*')):
    if  os.path.isdir(fdname) : 
        db_folders.append( fdname  )
db_folders.sort()
if len(db_folders) :
    for folder in db_folders[:5]:
        print(folder)
    else:
        print("...")
    print('Found ' + str(len(db_folders)) + ' folders.', end="\n\n")
else:
    print('No assignment folders. stop.')
    exit()

print('Building attendance table...')
att_table = dict()
for folder in db_folders:
    submissions = [d for d in glob.glob(os.path.join(folder,"*"))]
    print(os.path.basename(folder), "submissions = ", len(submissions))
    for f_or_d in submissions:
        folder_sum = 0
        folder_count = 0
        if os.path.isdir(f_or_d) :
            with os.scandir(f_or_d) as it:
                for entry in it:
                    if not entry.name.startswith('.') and entry.is_file() :
                        folder_count += 1
                        folder_sum += entry.stat().st_size
            print(folder_sum)
            if folder_sum < 16*1024 :
                print(f_or_d+'/'+entry.name)
                print('warning! folder sum is '+str(round(folder_sum/1024,1))+', less than 16kB.')
            sid = os.path.basename(f_or_d).split('_')[0]
            att_table[(os.path.basename(folder),sid)] = folder_sum if folder_count > 0 else 'x'

print(att_table.keys())    

presence = dict()
presence[' sid'] = list()
for (sid, namestr, dep, grade) in db_registered:
    presence[sid] = list()
for dt in db_folders:
    datename = os.path.basename(dt)
    presence[' sid'].append(datename)
    for (sid, namestr, dep, grade) in db_registered:
        print(datename, sid)
        if (datename, sid) in att_table :
            sz = str(att_table[(datename, sid)]/1024)
        else:
            sz = 0
        presence[sid].append(str(sz))

with open(os.path.join(path, 'attendance.csv'), mode='w', encoding='utf-8-sig') as outputf:
    for sid in presence.keys():
        outputf.write(sid)
        outputf.write(',')
        outputf.write(','.join(presence[sid]))
        outputf.write('\n')

exit()

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
