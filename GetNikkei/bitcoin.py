# coding: UTF-8
import lxml.html
from datetime import datetime
import csv, time, codecs


INTERVAL = 60 * 2 #待機時間の長さ設定。

#抜き出した行から複数列を抜き出す
def get_item(tr):
    exchange = None

    if len(tr[0]) > 0:
        exchange = tr[0][0].text
    else:
        return None

    return (exchange, tr[1].text, tr[2].text)

#ウェブサイトから複数行を抜き出す
def do_snapshot(url, format):
    snapshot = list()
    retult = None
    parse_html = lxml.html.parse(url)
    tr_array = parse_html.xpath(format)
    for tr in tr_array:
        retult = get_item(tr)
        if retult:
            snapshot.append(retult)
    return snapshot


#取得したデータからリスト作成
def flush_handle(writer):
    # 相場データ(0:取引所 1:買板 2:売板)
    result = do_snapshot('http://xn--eck3a9bu7cul981xhp9b.com/',
                         '//div[@class="panel panel-info"]//table[@class="table table-bordered"]//tbody//tr')

    # ページ下部の方の24時間の取引高(0:取引所 1:BTC 2:円換算)を取得する場合
    # do_snapshot('http://xn--eck3a9bu7cul981xhp9b.com/','//div[@class="panel panel-warning"]//table[@class="table table-bordered"]//tbody//tr')

    time_ = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    for item in result:
        # item[0] は「取引所」の名前です。
        writer.writerow([time_, item[1], item[2]])

    print('APPEND:' + time_)


def do_idle():
    # You can do something
    pass

#時間間隔を守りながらファイル書き込み操作
def main():
    #bitcoin.csvというファイルを作り、開く。第２引数のaは追記モード。
    with open('bitcoin.csv', 'a') as f:
        #自動改行の設定
        writer = csv.writer(f, lineterminator='\n')
        #１行目に書き込む 
        writer.writerow(['time', 'sell', 'buy'])

        try:
            check = 0
            # 永久に実行させます
            while True:
                now = time.time()
                #待機時間を超えた際
                if now - check > INTERVAL:
                    flush_handle(writer)
                    writer.writerow(['___', '___', '___'])  # 削除できます(テスト用)
                    f.flush()
                    check = time.time()
                #待機時間
                else:
                    do_idle()
        #処理が中止された際の動作
        except KeyboardInterrupt:
            print('KeyboardInterrupt')


# Program entry
if __name__ == '__main__':
    main()
