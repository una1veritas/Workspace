import subprocess
import sys


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
                time_str = " ".join(parts[3:7])
                print(f"🟢 起動 (Boot)   : {time_str}")
                count += 1

            # 終了ログの整形
            elif "shutdown" in line:
                # 例: shutdown ~                         Mon Aug 31 00:15
                parts = line.split()
                time_str = " ".join(parts[3:7])
                print(f"🔴 終了 (Shutdown): {time_str}")
                count += 1

        if count == 0:
            print("履歴が見つかりませんでした。")

    except subprocess.CalledProcessError as e:
        print(f"エラー: 履歴の取得に失敗しました。: {e}")
        sys.exit(1)


if __name__ == "__main__":
    get_boot_shutdown_history()
