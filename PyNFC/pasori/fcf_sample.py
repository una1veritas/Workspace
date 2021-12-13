
import binascii
import nfc
import time

# 学生証のサービスコード
service_code = 0x300B

def on_connect_nfc(tag):
    # タグのIDなどを出力する
    # print tag
    
    if isinstance(tag, nfc.tag.tt3.Type3Tag):
        try:
            # sc = nfc.tag.tt3.ServiceCode(service_code >> 6 ,service_code & 0x3f)
            # bc = nfc.tag.tt3.BlockCode(0,service=0)
            # data = tag.read_without_encryption([sc],[bc])
            # sid =  "s" + str(data[4:11])
            # print(sid)
            # 内容を16進数で出力する
            print(binascii.hexlify(tag._nfcid)) #convert byte array to hex str
            for item in tag.dump():
                print(item)
            #print('  ' + '\n  '.join(tag.dump()))
        except Exception as e:
            print("error: %s" % e)
        else:
            print("error: tag isn't Type3Tag")
        time.sleep(0.5)
            
def main():
    clf = nfc.ContactlessFrontend('usb')
#    while True:
    clf.connect(rdwr={'on-connect': on_connect_nfc})
#        time.sleep(3)
        
if __name__ == "__main__":
    main()

    
