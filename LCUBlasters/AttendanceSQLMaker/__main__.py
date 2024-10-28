

for i in range(1,16) :
	print('create temporary table tmp{0} as select * from attendance where date = (select date from dates where No = {0});'.format(i))
print()

sql = 'select tmp1.subjcode, tmp1.sid, tmp1.name, tmp1.attendance '
for i in range(2, 16) :
	sql += ', tmp'+str(i) + '.attendance'
sql += "from tmp1"
for i in range(2, 16) :
	sql += ', tmp'+str(i) 
sql += 'where '
for i in range(2, 16) :
	sql += 'tmp'+str(i-1)+'.sid = tmp'+str(i)+'.sid'
	if i < 15 :
		sql += ' and '
print(sql)