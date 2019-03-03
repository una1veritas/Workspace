import os
import sys

from PIL import Image
import numpy as np

'''workpath = ''
if len(sys.argv) > 1:
    workpath = sys.argv[1]
    if not os.path.isabs(workpath):
        workpath = os.path.abspath(workpath)
else:
    workpath = os.getcwd()
print('working dir = '+workpath)
print(os.listdir(workpath))

'''
workpath = "dataset_before"
exportpath = "dataset_after"
np_data = []

for file in os.listdir(workpath):
    filext = file.split('.')[-1].lower()
    print(filext, workpath + '/' + file)
    if filext == 'jpg' or filext == 'png':
        img = Image.open(workpath + '/' + file)
        print(img.mode, img.size[0], img.size[1])
        img_width, img_height = img.size
        img_classname = file.split('_')[0]
        if img.mode == 'RGB':
            img = img.convert('L')
        if img_height == 32 and img_width == 32:
            pixdata = np.array([[img.getpixel((i,j)) for j in range(32)] for i in range(32)])
            np_data.append((img_classname, pixdata))
        elif img_height > 32 or img_width > 32:
            for offset_x in range(0, img_width - 32 + 1):
                for offset_y in range(0, img_height - 32 + 1):
                    img_cropped = img.crop(box=(offset_x, offset_y, offset_x + 32, offset_y + 32))
                    img_cropped.save(exportpath + '/' + file.split('.')[0] + str(offset_x) + '_' + str(offset_y) + '.' + file.split('.')[-1].lower(), 'PNG')
                    pixdata = np.array([[img_cropped.getpixel((i,j)) for j in range(32)] for i in range(32)])
                    np_data.append((img_classname, pixdata))

                    '''if img_height > 35 and img_width > 35:
                        img_rot = img.rotate(-2)
                        if img_rot != img:
                            img_rot.save(exportpath + '/' + file.split('.')[0] + str(offset_x) + '_' + str(offset_y) + '_rn.' + file.split('.')[-1].lower(), 'PNG')
                            pixdata = np.array([[img_rot.getpixel((i, j)) for j in range(32)] for i in range(32)])
                            np_data.append((img_classname, pixdata))

                        img_rot = img.rotate(2)
                        if img_rot != img:
                            img_rot.save(exportpath + '/' + file.split('.')[0] + str(offset_x) + '_' + str(offset_y) + '_rp.' + file.split('.')[-1].lower(), 'PNG')
                            pixdata = np.array([[img_rot.getpixel((i, j)) for j in range(32)] for i in range(32)])
                            np_data.append((img_classname, pixdata))'''

    else:
        print('skip'+file)

print(len(np_data))