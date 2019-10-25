import sys

# 引数に数字 8 文字を渡すと、8 桁の整数で書かれたグレゴリオ歴（1815年10月15日以降）
# の日付 YYYYMMDD とみなし、その Julian day number ユリウス通日を出力する

def JulianDate(yyyy, mm, dd) :
    if mm < 3 :
        mm += 12
        yyyy -= 1
    a = int(yyyy/100)
    b = 2-a+int(a/4)
    return int(365.25 * yyyy) + int(30.6001 * (mm+1)) + dd + b + 1720994.5;

def DayOfWeek(jd):
    day_name = { 0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
    return day_name[int(jd + 1) % 7]

if len(sys.argv) == 1 :
    exit(1) # 入力がない、エラー終了

yyyymmdd = float(sys.argv[1])
yyyy = int(yyyymmdd) // 10000
mm = int(yyyymmdd) // 100 % 100
dd = yyyymmdd % 100

print(yyyy, mm, dd)
jd = JulianDate(yyyy, mm, dd)
print(jd, DayOfWeek(jd))
