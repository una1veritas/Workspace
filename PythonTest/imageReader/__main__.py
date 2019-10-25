#pixel image file data reader
import os
import sys

from PIL import Image
import numpy as np

workpath = ''
if len(sys.argv) > 1 :
    workpath = sys.argv[1]
    if not os.path.isabs(workpath) :
        workpath = os.path.abspath(workpath)
else:
    workpath = os.getcwd()
print('working dir = '+workpath)
print(os.listdir(workpath))

#(R',G',B') = 逆ガンマ補正(R,G,B)
#V' = 0.2126*R' + 0.7152*G' + 0.0722*B'
#V = ガンマ補正(V')
#
#def RGBtoGrey(pixval):
#    return int(0.2126 * pixval[0] + 0.7152 * pixval[1] + 0.0722 * pixval[2])

np_data = []

for file in os.listdir(workpath) :
    filext = file.split('.')[-1].lower()
    print(filext, workpath + '/' + file)
    if filext == 'jpg' or filext == 'png':
        img = Image.open(workpath + '/' + file)
        print(img.mode, img.size[0], img.size[1])
        img_width, img_height = img.size
        img_classname = file.split('_')[0]
        if img.mode == 'RGB' :
            img = img.convert('L')
        if img_height == 32 and img_width == 32 :
            pixdata = np.array([[img.getpixel((i,j)) for j in range(32)] for i in range(32)])
            np_data.append( (img_classname, pixdata) )
        elif img_height >= 32 and img_width >= 32:
            # ぎりぎりまで左右上下にずらして例を作成
            for offset_x in range(0, img_width - 32 + 1) :
                for offset_y in range(0, img_width - 32 + 1):
                    img_cropped = img.crop(box=(offset_x, offset_y, offset_x + 32, offset_y+32 ))
                    pixdata = np.array([[img_cropped.getpixel((i,j)) for j in range(32)] for i in range(32)])
                    np_data.append( (img_classname, pixdata) )
                    # 大きさにゆとりがあるなら反時計回り，時計回りにかたむける．とりあえず 2 度ぐらいでいいやろ
                    if img_height > 36 and img_width > 36:
                        img_rot = img.rotate(-2)
                        if img_rot != img :
                            pixdata = np.array([[img_rot.getpixel((i,j)) for j in range(32)] for i in range(32)])
                            np_data.append( (img_classname, pixdata) )
                        img_rot = img.rotate(2)
                        if img_rot != img :
                            pixdata = np.array([[img_rot.getpixel((i,j)) for j in range(32)] for i in range(32)])
                            np_data.append( (img_classname, pixdata) )
    else:
        print('skip ' + file)

print(len(np_data))
print(np_data[:30])
#python3 imageDataReader ./files
