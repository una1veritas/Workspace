import sys

# 引数に数字 8 文字を渡すと、8 桁の整数で書かれたグレゴリオ歴（1815年10月15日以降）
# の日付 YYYYMMDD とみなし、その Julian day number ユリウス通日を出力する

def JulianDate(yyyy, mm, dd) :
    if mm < 3 :
        mm += 12
        yyyy -= 1
    if yyyy*10000+mm*100+dd >= 15821015 :
        a = int(yyyy/100)
        b = 2-a+int(a/4)
    else:
        b = 0
    return int(365.25 * (yyyy + 4716)) + int(30.6001 * (mm+1)) + dd + b - 1524.5;

def DayOfWeek(jd):
    day_name = { 1:'Mon', 2:'Tue', 3:'Wed', 4:'Thu', 5:'Fri', 6:'Sat', 0:'Sun'}
    return day_name[int(jd + 1.5) % 7]

if len(sys.argv) == 1 :
    exit(1) # 入力がない、エラー終了

yyyymmdd = float(sys.argv[1])
print(yyyymmdd)
yyyy = int(yyyymmdd) // 10000
mm = int(yyyymmdd) // 100 % 100
dd = yyyymmdd % 100

print("yyyy = {}, mm = {}, dd = {}".format(yyyy, mm, dd))
jd = JulianDate(yyyy, mm, dd)
print(jd, DayOfWeek(jd))
