from PIL import Image
import numpy as np

# 元となる画像の読み込み
img = Image.open('cloudyandsunny.png')
#オリジナル画像の幅と高さを取得
width, height = img.size

img.show()
