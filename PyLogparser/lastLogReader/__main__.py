'''
Created on 2018/08/01

@author: sin
'''

import sys
import datetime
import subprocess

if __name__ == '__main__':
    pass
else:
    exit()

month_dict = { 'Jan':1, 'Feb':2, 'Mar': 3, 'Apr': 4, 
             'May': 5, 'Jun':6, 'Jul': 7, 'Aug': 8,
             'Sep': 9, 'Oct': 10, 'Nov':11, 'Dec': 12, }

cmd_list = ['last']
try:
    res = subprocess.run(
        cmd_list, 
        shell=True, 
        stdout=subprocess.PIPE,
        universal_newlines=True,
        )
except subprocess.CalledProcessError:
    print('error on '+str(cmd_list))
    exit()

date_today = datetime.date.today()
active_period = dict()

for each_line in res.stdout.splitlines():
    items = each_line.split()
    if len(items) < 4:
        continue
    evnt_year = int(date_today.year)
    evnt_mon = month_dict[items[3]]
    evnt_date = int(items[4])
    evnt_dt = datetime.date(evnt_year, evnt_mon, evnt_date)
    hm = items[5].split(':')
    time_1 = datetime.time(hour=int(hm[0]), minute=int(hm[1]))
    time_2 = time_1
    if len(items) > 7 :
        hm = items[7].split(':')
        if len(hm) == 2:
            time_2 = datetime.time(hour=int(hm[0]), minute=int(hm[1]))
    if not evnt_dt in active_period :
        active_period[evnt_dt] = [time_1, time_2]
    else:
        if time_1 < active_period[evnt_dt][0] :
            active_period[evnt_dt][0] = time_1
        if time_2 > active_period[evnt_dt][1] :
            active_period[evnt_dt][1] = time_2

for key in sorted(active_period) :
    print(key, active_period[key][0], '-', active_period[key][1])
    
exit()

# fname = sys.argv[1]
# print('filename: '+fname)
# bseq = [ ]
# with open(fname, "r") as ifile:
#     for a_line in ifile:
#         data = a_line.split()
#         if len(data) == 0:
#             break
#         if data[1][0:4] == 'ttys' :
#             continue
#         dcstr = data[0]
#         if not (dcstr == 'reboot' or dcstr == 'shutdown') :
#             dcstr = 'user'
#         datestr = str(monthdic[data[3]]).zfill(2)+'/'+data[4].zfill(2)
#         timestr = data[5]
#         bseq.append([datestr, timestr, dcstr])
# 
# bseq.reverse()
# #bseq.sort(key=lambda x:(x[0],x[1]))

date = ''
time_from = ''
time_to = ''
for tp in bseq:
    if date != tp[0] :
        if date != '' :
            print(date+'\t'+time_from+'\t'+time_to)
        date = tp[0]
        time_from = ''
        time_to = ''
    else:
        if time_from == '' or tp[1] < time_from :
            time_from = tp[1]
        elif tp[1] > time_to :
            time_to = tp[1]
print(date+',\t'+time_from+',\t'+time_to)

print('finished.')