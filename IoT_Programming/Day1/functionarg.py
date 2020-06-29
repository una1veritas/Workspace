from datetime import datetime

def dt(form = 0):
    if form == 0:
        return datetime.now().strftime("%m/%d/%y %H:%M:%S")
    else:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


print(dt())
print(dt(1))
