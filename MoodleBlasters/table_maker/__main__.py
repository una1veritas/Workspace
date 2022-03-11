import os, glob
import io,sys

path = u'/Users/sin/Dropbox/離散数学Ⅱ/離散数学Ⅱ-2021q4/quiz'
if path[-1] != '/' : path += '/'
print('Looking for name list file in '+path+'.')
if len(glob.glob(path+'seiseki*.csv')) == 1 :
    namelistf = glob.glob(path+'seiseki*.csv')[0]
    print('Found "'+namelistf+'" for registered students names.')
else:
    print('Not found name flie "namelist*.csv" at '+path+'. ', file=sys.stderr)

print('Reading the list of registered students.')
db_registered = list()
COMMENTESCAPE = '#'
separator = ','
with open(namelistf, encoding = "utf-8-sig") as f:
    for l in f.readlines():
        if l[0] == COMMENTESCAPE : continue
        fields = l.split(separator)
        # (sid, namestr, dep, grade)
        db_registered.append( (fields[2], fields[1], fields[4], fields[6]) )
        #print(fields)
db_registered.sort(key=(lambda x: x[0]))
#for i in db_registered:
#    print(i)
print('Found ' + str(len(db_registered))+' names of students.')
for row in db_registered[:5] :
    print(row)

print('Collecting folder names.')
db_folders = list()
for fdname in glob.glob(path+'*'):
    if  os.path.isdir(fdname) : 
        db_folders.append( ( os.path.basename(fdname) ) )
db_folders.sort()
if len(db_folders) :
    for folder in db_folders[:5]:
        print(folder)
    print('Found ' + str(len(db_folders)) + ' folders for reports.')
else:
    print('No folders. stop.')
    exit()

print('Building attendance table...')
att_table = dict()
for folder in db_folders:
    submissiondirs = [d for d in glob.glob(path+folder+'/*')]
    for report_folder in submissiondirs:
        folder_sum = 0
        folder_count = 0
        with os.scandir(report_folder) as it:
            for entry in it:
                if not entry.name.startswith('.') and entry.is_file() :
                    folder_count += 1
                    folder_sum += entry.stat().st_size
        if folder_sum < 16*1024 :
            print(report_folder+'/'+entry.name)
            print('warning! folder sum is '+str(round(folder_sum/1024,1))+', less than 16kB.')
        sid = os.path.basename(report_folder).split('_')[0]
        att_table[(folder,sid)] = folder_sum if folder_count > 0 else 'x'

#print(att_table.keys())    
#exit()

presence = dict()
presence[' sid'] = list()
for (sid, namestr, dep, grade) in db_registered:
    presence[sid] = list()
for dt in db_folders:
    presence[' sid'].append(dt)
    for (sid, namestr, dep, grade) in db_registered:
        if (dt, sid) in att_table :
            sz = str(att_table[(dt, sid)]/1024)
        else:
            sz = 0
        presence[sid].append(str(sz))

with open(path+'attendance.csv', mode='w', encoding='utf-8-sig') as outputf:
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
