#!/usr/local/bin/python2.7
# encoding: utf-8
# coding: UTF-8
import urllib2
from bs4 import BeautifulSoup
from datetime import datetime
import csv
import time

time_flag = True

# 永久に実行させます
while True:
    # 時間が59分以外の場合は58秒間時間を待機する
    if datetime.now().second <= 30:
        time.sleep(28)
        continue

    # csvを追記モードで開きます→ここでcsvを開くのはファイルが大きくなった時にcsvを開くのに時間がかかるためです
    #f = open('nikkei_heikin.csv', 'a')
    #writer = csv.writer(f, lineterminator='\n')

    # 59分になりましたが正確な時間に測定をするために秒間隔で59秒になるまで抜け出せません
    while datetime.now().second != 59:
            # 00秒ではないので1秒待機
            time.sleep(1)
    # 処理が早く終わり二回繰り返してしまうのでここで一秒間待機します
    time.sleep(1)

    # csvに記述するレコードを作成します
    csv_list = []

    # 現在の時刻を年、月、日、時、分、秒で取得します
    time_ = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    # 1カラム目に時間を挿入します
    csv_list.append(time_)

    # アクセスするURL
    #url = "http://www.nikkei.com/markets/kabu/"
    url = "https://stocks.finance.yahoo.co.jp/stocks/detail/?code=998407.O"
    
    # URLにアクセスする htmlが帰ってくる → <html><head><title>経済、株価、ビジネス、政治のニュース:日経電子版</title></head><body....
    html = urllib2.urlopen(url)

    # htmlをBeautifulSoupで扱う
    soup = BeautifulSoup(html, "html.parser")

    # span要素全てを摘出する→全てのspan要素が配列に入ってかえされます→[<span class="m-wficon triDown"></span>, <span class="l-h...
    #span = soup.find_all("span")
    contents = soup.find_all("table");

    # print時のエラーとならないように最初に宣言しておきます。
    nikkei_heikin = ""
    # for分で全てのspan要素の中からClass="mkc-stock_prices"となっている物を探します
    for table in contents:
        # classの設定がされていない要素は、tag.get("class").pop(0)を行うことのできないでエラーとなるため、tryでエラーを回避する
        try:
            # tagの中からclass="n"のnの文字列を摘出します。複数classが設定されている場合があるので
            # get関数では配列で帰ってくる。そのため配列の関数pop(0)により、配列の一番最初を摘出する
            # <span class="hoge" class="foo">  →   ["hoge","foo"]  →   hoge
            string_ = table.get("class").pop(0)

            # 摘出したclassの文字列にmkc-stock_pricesと設定されているかを調べます
            if string_ in "stocksTable":
                # mkc-stock_pricesが設定されているのでtagで囲まれた文字列を.stringであぶり出します
                nikkei_heikin = table.string
                # 摘出が完了したのでfor分を抜けます
                break
        except:
            # パス→何も処理を行わない
            pass

    # 摘出した日経平均株価を時間とともに出力します。
    print time_, nikkei_heikin
    # 2カラム目に日経平均を記録します
    csv_list.append(nikkei_heikin)
    # csvに追記敷きます
    #writer.writerow(csv_list)
    # ファイル破損防止のために閉じます
    #f.close()
    