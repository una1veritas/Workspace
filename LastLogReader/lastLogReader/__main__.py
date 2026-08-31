import subprocess
import os, sys
from datetime import datetime, timedelta

def get_boot_shutdown_history():
    print("macOSのシステム履歴から起動/終了時刻を抽出中...")

    # last コマンドで reboot（起動）と shutdown（終了）の履歴を取得
    cmd = ["last", "-y", "reboot", "shutdown", "console"]
    curr_env = os.environ.copy()
    curr_env["LC_ALL"] = "C"
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=curr_env)
        lines = result.stdout.strip().split("\n")

        #print("\n--- 起動・終了履歴 (新しい順) ---")
        count = 0
        logdict = dict()

        for line in lines:
            if not line.strip():
                continue
            
            items = line.strip().split()
            if items[0] == 'wtmp':
                continue
            
            count += 1
            #print(items)
            dt = datetime.strptime(' '.join(items[3:7]), "%b %d %Y %H:%M")
            #print(dt.date())
            if items[0] in ('reboot', 'shutdown') :
                #events
                if dt.date() not in logdict :
                    logdict[dt.date()] = list()
                logdict[dt.date()].append( (items[0], str(dt)) )
                # print(logdict[dt.date()][-1])
            elif ('sin', 'console') == (items[0], items[1]) :
                #login-out
                print(items[0], items[6:11])
                if items[7] == 'still' :
                    dur = datetime.now() - dt
                else:
                    dur = items[9].strip('()').split('+')
                    hh, mm = dur[-1].split(':')
                    dd = 0
                    if len(dur) != 1 :
                        dd = int(dur[0])
                    dur = timedelta(days=dd, hours=int(hh), minutes=int(mm))
                    #print(dur)
                if dt.date() not in logdict :
                    logdict[dt.date()] = list()
                logdict[dt.date()].append( (items[1], str(dt), str(dt+dur)) )
            
        for e in sorted(logdict.keys()):
            print(e, sorted(logdict[e], key=lambda x: x[1:]))
        
        if count == 0:
            print("履歴が見つかりませんでした。")

    except subprocess.CalledProcessError as e:
        print(f"エラー: 履歴の取得に失敗しました。: {e}")
        sys.exit(1)
    
if __name__ == "__main__":
    get_boot_shutdown_history()
