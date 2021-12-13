import binascii
import nfc
import os

class MyCardReader(object):
    def on_connect(self, tag):
        #タッチ時の処理
        print("【 Touched 】")

        #タグ情報を全て表示
        print(tag)

        #IDmのみ取得して表示
        self.idm = binascii.hexlify(tag._nfcid)
        print("IDm : " + str(self.idm))

        #特定のIDmだった場合のアクション
        if self.idm == "00000000000000":
            print("【 登録されたIDです 】")

        return True

    def read_id(self):
        clf = nfc.ContactlessFrontend('usb')
        try:
            clf.connect(rdwr={'on-connect': self.on_connect})
        finally:
            clf.close()

if __name__ == '__main__':
    cr = MyCardReader()
    while True:
        #最初に表示
        print("Please Touch")

        #タッチ待ち
        cr.read_id()

        #リリース時の処理
        print("【 Released 】")

