# -*- coding: utf-8 -*-
"""点列教室（合成データ）から、入力例（点列の画像2枚と正解）を作る。

    python make_example_images.py

examples/ に、例ごとに次の3つを書き出す。
    <例>_P.png             1枚目の点列の画像（白地に黒い点。1つの点が1人）
    <例>_Q.png             2枚目の点列の画像
    <例>_ground_truth.txt  正しい人数と、2枚を重ねる正しい変換

点列教室：1つの座席配置から、2枚の視野を切り出す。2枚目は横に大きく（視野の半分ほど）、縦に少し（1行以内）
ずれていて、倍率・せん断・ロールも少し入る。各点には検出のずれ（標準偏差 delta px）を2枚で独立に足す。
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import synthetic_classroom as G

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "examples")
RADIUS = 3.0          # 点の半径 [px]

# (名前, 点列教室の設定, 乱数の種)。種は手法の値を決めたときにも確かめたときにも使っていないもの
EXAMPLES = [
    ("example1_standard", dict(), 501),
    ("example2_vertical_shift_one_row", dict(ty=70.0), 502),
    ("example3_nearly_full", dict(occ=0.85, W=700, tx=350.0), 503),
    ("example4_sparse", dict(occ=0.35), 504),
    ("example5_large_detection_error", dict(delta=6.0), 505),
]


def render(points, W, H, r=RADIUS):
    """白地に黒い円を描く。縁はなめらかにする（点の中心を画素より細かく読み取れるように）。"""
    img = np.ones((H, W))
    for x, y in points:
        x0, x1 = max(int(np.floor(x - r - 1)), 0), min(int(np.ceil(x + r + 1)), W)
        y0, y1 = max(int(np.floor(y - r - 1)), 0), min(int(np.ceil(y + r + 1)), H)
        if x0 >= x1 or y0 >= y1:
            continue
        jj, ii = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        d = np.hypot(jj + 0.5 - x, ii + 0.5 - y)          # 画素 (ii, jj) の中心は (jj + 0.5, ii + 0.5)
        cover = np.clip(r + 0.5 - d, 0.0, 1.0)
        img[y0:y1, x0:x1] = np.minimum(img[y0:y1, x0:x1], 1.0 - cover)
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, prm, seed in EXAMPLES:
        aff = G.draw_affine(seed)
        g = G.generate(dict(prm, **aff), seed)
        W, H = g["prm"]["W"], g["prm"]["H"]
        plt.imsave(os.path.join(OUT, name + "_P.png"), render(g["P"], W, H), cmap="gray", vmin=0, vmax=1)
        plt.imsave(os.path.join(OUT, name + "_Q.png"), render(g["Q"], W, H), cmap="gray", vmin=0, vmax=1)
        A, b = g["A_true"], g["b_true"]
        with open(os.path.join(OUT, name + "_ground_truth.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("正しい人数        %d 人\n" % g["N_true"])
            f.write("P の点            %d 点\n" % g["a"])
            f.write("Q の点            %d 点\n" % g["b"])
            f.write("両方に写る人      %d 人\n" % len(g["pairs"]))
            f.write("画像の大きさ      %d x %d px\n" % (W, H))
            sg = lambda v, fmt: ("− " if v < 0 else "+ ") + fmt % abs(v)
            first = lambda v: ("−" if v < 0 else "") + "%.4f" % abs(v)
            f.write("正しい変換        P の点 (x, y) は、Q では (x', y') に写る（count_people.py の出力と同じ書き方）\n")
            f.write("                  x' = %s x %s y %s\n" % (first(A[0, 0]), sg(A[0, 1], "%.4f"), sg(b[0], "%.1f")))
            f.write("                  y' = %s x %s y %s\n" % (first(A[1, 0]), sg(A[1, 1], "%.4f"), sg(b[1], "%.1f")))
            f.write("点列教室の設定    倍率 %.3f、せん断 %.3f、ロール %.2f 度、検出のずれ %.0f px、着席率 %.0f%%\n"
                    % (aff["scale"], aff["shear"], aff["roll"], g["prm"]["delta"], 100 * g["prm"]["occ"]))
        print("%s：正しい人数 %d（P %d 点、Q %d 点、両方 %d 人）" % (name, g["N_true"], g["a"], g["b"], len(g["pairs"])))
    print("入力例を %s に書き出した" % OUT)


if __name__ == "__main__":
    main()
