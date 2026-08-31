import subprocess
import sys
from datetime import datetime

def get_boot_shutdown_history():
    print("macOSのシステム履歴から起動/終了時刻を抽出中...")

    # last コマンドで reboot（起動）と shutdown（終了）の履歴を取得
    cmd = ["last", "reboot", "shutdown"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split("\n")

        print("\n--- 起動・終了履歴 (新しい順) ---")
        count = 0

        for line in lines:
            if not line.strip():
                continue

            # 起動ログの整形
            if "reboot" in line:
                # 例: reboot   ~                         Mon Aug 31 09:30
                parts = line.split()
                # 曜日 月 日 時刻 を抽出
                # print(parts)
                dow, mon_name = parts[2], parts[3]
                date = int(parts[4])
                hour = int(parts[5].split(':')[0])
                minute = int(parts[5].split(':')[1])
                year = 2026  # 年は適宜設定してください
                # print(f"{dow} {date} {mon_name} {year} {parts[5]}")
                dt = datetime.strptime(f"{dow} {date} {mon_name} {year} {parts[5]}", "%a %d %b %Y %H:%M")
                print(f"{dt.strftime('%Y-%m-%d %H:%M')} 🟢 起動 (Boot) ")
                count += 1

            # 終了ログの整形
            elif "shutdown" in line:
                # 例: shutdown ~                         Mon Aug 31 00:15
                parts = line.split()
                # 曜日 月 日 時刻 を抽出
                # print(parts)
                dow, mon_name = parts[2], parts[3]
                date = int(parts[4])
                hour = int(parts[5].split(':')[0])
                minute = int(parts[5].split(':')[1])
                year = 2026  # 年は適宜設定してください
                # print(f"{dow} {date} {mon_name} {year} {parts[5]}")
                dt = datetime.strptime(f"{dow} {date} {mon_name} {year} {parts[5]}", "%a %d %b %Y %H:%M")
                print(f"{dt.strftime('%Y-%m-%d %H:%M')} 🔴 終了 (Shutdown): ")
                count += 1

        if count == 0:
            print("履歴が見つかりませんでした。")

    except subprocess.CalledProcessError as e:
        print(f"エラー: 履歴の取得に失敗しました。: {e}")
        sys.exit(1)


if __name__ == "__main__":
    get_boot_shutdown_history()
