'''
Created on 2023/11/20

@author: sin
'''

import sys
import datetime

TERMINAL = {
    28 : ('2101', '1'), 
    29 : ('2101', '2'), 
    }

LECTURE_DATES = {
    '2023-10-04', '2023-10-11', '2023-10-13', 
    '2023-10-18', '2023-10-20', '2023-10-27',
    '2023-11-01', '2023-11-08', '2023-11-10',
    '2023-11-15', '2023-11-17',
    }

if __name__ != '__main__':
    exit(0)

print(sys.argv)

reader_master_db = dict()
for filename in sys.argv[1:] :
    with open(filename, 'r') as f :

        for linestr in f :
            #read the first header line
            rowitems = linestr.strip().split(',')
            if rowitems == ['区分', '学生番号', '枝番', '日付', '時刻', '端末番号'] :
                #print('header')
                pass
            else:
                rowitems = linestr.strip().split(',')
                #print(rowitems)
                terminal_id = int(rowitems[5], 10)
                scanned_datetime = datetime.datetime(int(rowitems[3][0:4]), int(rowitems[3][4:6]), int(rowitems[3][6:8]),
                                            int(rowitems[4][0:2]), int(rowitems[4][2:4]), int(rowitems[4][4:6]))
                scanned_stampdate = scanned_datetime.strftime('%Y-%m-%d')
                scanned_stamptime = scanned_datetime.strftime('%H:%M')
                sid_issue = rowitems[2]
                sid_number = rowitems[1]
                card_class = rowitems[0]
                (room_name, reader_name) = TERMINAL[terminal_id]
                if not (room_name, scanned_stampdate) in reader_master_db :
                     reader_master_db[ (room_name, scanned_stampdate) ] = list()
                reader_master_db[ (room_name, scanned_stampdate) ].append( (scanned_stamptime, sid_number) )
                
print('reading data finished.')
print('imported '+str(len(reader_master_db))+ ' data.')

student_ids = set()
attendance_sheet = dict()
for key, value in reader_master_db.items() :
    room_name, date = key
    if room_name != '2101' :
        continue 
    if not date in LECTURE_DATES :
        continue
    for scanned in value :
        timestamp, sid = scanned
        if '12:40' <= timestamp <= '13:35' : 
            if not date in attendance_sheet :
                attendance_sheet[date] = dict()
            if not sid in student_ids :
                student_ids.add(sid)
            attendance_sheet[date][sid] = timestamp
    
for key in attendance_sheet:
    print(key)
    print(attendance_sheet[key])

#make an output attendance sheet.
dates = list(attendance_sheet.keys())
sids = sorted(student_ids)
print(dates)
with open('attendance.csv', mode='w') as f :
    f.write('date,')
    f.write(dates[0])
    for date in dates[1:] :
        f.write(',')
        f.write(date)
    print(file=f)
    for sid in sids :
        f.write(sid)
        for date in dates :
            f.write(',')
            if sid in attendance_sheet[date] :
                f.write(attendance_sheet[date][sid])
        print(file=f)