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

print(datetime.date.today())
active_period = dict()

for each_line in res.stdout.splitlines():
    item_list = each_line.split()
    print(item_list[2:6])
#     if len(item_list) >= 5:
#         line_date = str(month_dict[item_list[3]])+'/'+item_list[4].zfill(2)
#         line_day = item_list[2]
#         line_time = item_list[5]
#         if item_list[0] == 'reboot' or item_list[0] == 'shutdown':
#             if not (line_date, line_day) in event_log : 
#                 event_log[(line_date, line_day)] = [(line_time, item_list[0])]
#             else:
#                 event_log[(line_date, line_day)].append( (line_time, item_list[0]) )
    
    
# for key in sorted(event_log.keys()):
#     print(key, sorted(event_log[key]))
exit()
    

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