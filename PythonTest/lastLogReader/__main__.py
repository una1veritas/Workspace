'''
Created on 2018/08/01

@author: sin
'''

import sys
import datetime
import subprocess
import math

if __name__ == '__main__':
    pass
else:
    exit()

month = 0
if len(sys.argv) > 1 :
    month = int(sys.argv[1])
if month == 0 or month > 12:
    print('format: command month')
    exit()

month_dict = { 'Jan':1, 'Feb':2, 'Mar': 3, 'Apr': 4, 
             'May': 5, 'Jun':6, 'Jul': 7, 'Aug': 8,
             'Sep': 9, 'Oct': 10, 'Nov':11, 'Dec': 12, }

dayname_dict = { 0:u'月', 1:u'火', 2:u'水', 3:u'木', 4:u'金', 5:u'土', 6:u'日',  }

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

todaydate = datetime.date.today()
keydate = todaydate - datetime.timedelta(days = 30)
keydate = datetime.date(keydate.year, keydate.month, 1)
while keydate != todaydate :
    if keydate in active_period :
        time1 = active_period[keydate][0]
        time2 = active_period[keydate][1]
        time1 = datetime.time(time1.hour, int(time1.minute/5)*5)
        if time2.minute > 55 :
            time2 = datetime.time(time2.hour+1, 0)
        else:
            time2 = datetime.time(time2.hour, math.ceil(time2.minute/5)*5)
        print(keydate, dayname_dict[keydate.weekday()], time1.strftime("%H:%M"), time2.strftime("%H:%M"), sep='\t' )
    else:
        print(keydate, dayname_dict[keydate.weekday()], sep='\t')
    keydate = keydate + datetime.timedelta(days=1)
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
