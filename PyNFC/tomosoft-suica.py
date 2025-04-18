#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
#sys.path.insert(1, os.path.split(sys.path[0])[0])
 
import binascii
import nfc
   
service_code = 0x090f
# 学生証のサービスコード
service_code = 0x300B

def connected(tag):
  # タグのIDなどを出力する
  print(tag)
   
  if isinstance(tag, nfc.tag.tt3.Type3Tag):
    try:
      # 内容を16進数で出力する
      print('  ' + '\n  '.join(tag.dump()))
    except Exception as e:
      print("error: %s" % e)
  else:
    print("error: tag isn't Type3Tag")
   
# タッチ時のハンドラを設定して待機する
clf = nfc.ContactlessFrontend('usb')
clf.connect(rdwr={'on-connect': connected})
