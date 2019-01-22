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
            for offset_x in range(0, img_width - 32 + 1) :
                for offset_y in range(0, img_width - 32 + 1):
                    img_cropped = img.crop(box=(offset_x, offset_y, offset_x + 32, offset_y+32 ))
                    pixdata = np.array([[img_cropped.getpixel((i,j)) for j in range(32)] for i in range(32)])
                    np_data.append( (img_classname, pixdata) )
                    if img_height > 36 and img_width > 36:
                        img_rotcc = img.rotate(-2)
                        pixdata = np.array([[img_rotcc.getpixel((i,j)) for j in range(32)] for i in range(32)])
                        np_data.append( (img_classname, pixdata) )
                        img_rotcc = img.rotate(2)
                        pixdata = np.array([[img_rotcc.getpixel((i,j)) for j in range(32)] for i in range(32)])
                        np_data.append( (img_classname, pixdata) )                
    else:
        print('skip ' + file)

print(len(np_data))
print(np_data[:30])
#python3 imageDataReader ./files
# working dir = /Users/sin/Documents/Workspace/PythonTest/imageDataReader/files
# ['4gray.png', '4gray.jpg', '3col.jpg', '4col.jpg', '8col.jpg']
# png /Users/sin/Documents/Workspace/PythonTest/imageDataReader/files/4gray.png
# L 32 32
# [[ 24  13  40 ... 224  63  12]
#  [ 22  10  40 ... 221  62  12]
#  [ 19  11  40 ... 222  61  14]
#  ...
#  [ 24  17 139 ... 114  18  33]
#  [ 23  16 142 ... 102  14  29]
#  [ 16  15 145 ...  94  12  25]]
# jpg /Users/sin/Documents/Workspace/PythonTest/imageDataReader/files/4gray.jpg
# L 32 32
# [[ 24  13  39 ... 224  61  12]
#  [ 20  12  39 ... 223  61  13]
#  [ 20  11  41 ... 222  61  14]
#  ...
#  [ 24  16 139 ... 112  19  32]
#  [ 21  16 144 ... 102  15  29]
#  [ 18  14 145 ...  95  10  24]]
# jpg /Users/sin/Documents/Workspace/PythonTest/imageDataReader/files/3col.jpg
# RGB 32 32
# [[ 13 114 241 ...  10  55  65]
#  [  9 116 244 ...  33  20  19]
#  [ 10 123 248 ...  32  32  34]
#  ...
#  [130 230 215 ...  43  28  81]
#  [133 229 214 ...  45  29  87]
#  [136 228 212 ...  45  34  84]]
# jpg /Users/sin/Documents/Workspace/PythonTest/imageDataReader/files/4col.jpg
# RGB 32 32
# [[ 50  26  70 ... 232  97  24]
#  [ 40  21  68 ... 232  97  25]
#  [ 39  23  72 ... 231  97  27]
#  ...
#  [ 49  34 169 ... 145  32  59]
#  [ 43  34 173 ... 138  30  56]
#  [ 33  30 177 ... 131  23  46]]
# jpg /Users/sin/Documents/Workspace/PythonTest/imageDataReader/files/8col.jpg
# RGB 32 32
# [[ 46  35  42 ... 181  86  39]
#  [ 45  33  41 ... 181  79  37]
#  [ 47  33  41 ... 181  74  39]
#  ...
#  [ 46  15 117 ... 125  55  40]
#  [ 38  10 120 ... 117  45  33]
#  [ 47  18 129 ... 113  40  28]]
