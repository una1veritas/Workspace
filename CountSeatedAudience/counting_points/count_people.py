# -*- coding: utf-8 -*-
"""点列の画像2枚から、写っている人数を数える。

    python count_people.py

入力  白地に黒い点を打った画像2枚（1つの点が1人）。パスは下の IMAGE_P、IMAGE_Q に書く。
      同じ教室を、横に大きく・縦に少しずらして撮った2枚にあたる
出力  画面に人数と途中の値。確認用の図 <1枚目の名前>_result.png（1枚目の画像と同じフォルダ）

手法（counting_method.py）
    段1・段2  2枚の点列を横と縦に射影したヒストグラムの形を比べ、重ね方（変換 T）の候補を探す
    段3      ビン幅を半分ずつ狭めて T を詰め、仮の組から最小二乗で当て直す
    段4      T で重ねて、ε 以内の点どうしを同じ人として組にする（最大マッチング）
    段5      一致率で候補を選び、人数 = 1枚目の点の数 + 2枚目の点の数 − 組の数
"""
import os
import sys

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import counting_method as M

# ------------------------------------------------------------------ 入力
# 点列の画像2枚のパス。別の画像で試すときは、ここを書き換える
# （count_people.py のあるフォルダから見たパス。C:/... のような絶対パスでもよい）
IMAGE_P = "examples/example1_standard_P.png"    # 1枚目（白地に黒い点）
IMAGE_Q = "examples/example1_standard_Q.png"    # 2枚目（1枚目と同じ大きさ）

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


# ------------------------------------------------------------------ 読み取り
def read_points(path):
    """白地に黒い点の画像から、点の中心を読む（濃さで重みをつけた重心。画素より細かく読める）。
    返り値 (点の座標 (n, 2)。x は右向き、y は下向き、画像の幅、画像の高さ、大きすぎる点の数)。"""
    img = plt.imread(path).astype(float)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    if img.max() > 1.0:
        img = img / 255.0
    dark = 1.0 - img
    lab, n = ndimage.label(dark > 0.1)
    H, W = img.shape
    if n == 0:
        return np.zeros((0, 2)), W, H, 0
    idx = np.arange(1, n + 1)
    com = np.array(ndimage.center_of_mass(dark, lab, idx))
    pts = com[:, ::-1] + 0.5                 # (行, 列) → (x, y)。画素 (行 i, 列 j) の中心は (j + 0.5, i + 0.5)
    area = ndimage.sum(np.ones_like(dark), lab, idx)
    big = int((area > 2.0 * np.median(area)).sum())   # 2つの点が重なって1つに見えているかもしれない点
    return pts, W, H, big


# ------------------------------------------------------------------ 表示
def pick_font():
    """日本語を描けるフォントを探す。なければ None（図の見出しを英語にする）。"""
    names = {f.name for f in font_manager.fontManager.ttflist}
    for c in ("Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic", "Meiryo", "Noto Sans CJK JP",
              "IPAexGothic", "TakaoGothic"):
        if c in names:
            return c
    return None


def draw(res, P, Q, W, H, dst, title):
    """上の段に P と Q、下の段に重ね合わせ（P を T で Q の座標へ写し、組を線で結ぶ）を描いて dst に保存する。"""
    jp = pick_font()
    if jp:
        plt.rcParams["font.family"] = jp
    t = (lambda ja, en: ja) if jp else (lambda ja, en: en)
    fig = plt.figure(figsize=(13, 2.0 + 11.0 * H / W))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.5])
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]

    def frame(a, label):
        a.plot([0, W, W, 0, 0], [0, 0, H, H, 0], color="0.6", lw=1)
        a.set_aspect("equal")
        a.invert_yaxis()
        a.set_title(label, fontsize=11)

    ax[0].scatter(P[:, 0], P[:, 1], s=12, c="k")
    frame(ax[0], t("1枚目 P（%d 点）" % len(P), "image P (%d points)" % len(P)))
    ax[1].scatter(Q[:, 0], Q[:, 1], s=12, c="k")
    frame(ax[1], t("2枚目 Q（%d 点）" % len(Q), "image Q (%d points)" % len(Q)))

    a = ax[2]
    if "top" in res:
        top = res["top"]
        TP = P @ top["A"].T + top["b"]
        corners = np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]], float) @ top["A"].T + top["b"]
        a.plot(corners[:, 0], corners[:, 1], color="tab:blue", lw=1, ls="--",
               label=t("P の視野を T で写した範囲", "view of P mapped by T"))
        pi = [i for i, _ in top["pairs"]]
        qj = [j for _, j in top["pairs"]]
        onlyP = np.setdiff1d(np.arange(len(P)), pi)
        onlyQ = np.setdiff1d(np.arange(len(Q)), qj)
        for i, j in top["pairs"]:
            a.plot([TP[i, 0], Q[j, 0]], [TP[i, 1], Q[j, 1]], color="tab:green", lw=1)
        a.scatter(Q[qj, 0], Q[qj, 1], s=22, facecolors="none", edgecolors="tab:green",
                  label=t("組になった人（%d）" % len(qj), "matched (%d)" % len(qj)))
        a.scatter(TP[pi, 0], TP[pi, 1], s=10, c="tab:green", marker="x")
        a.scatter(TP[onlyP, 0], TP[onlyP, 1], s=14, c="tab:blue", marker="x",
                  label=t("組にならなかった P の点（%d）" % len(onlyP), "unmatched P (%d)" % len(onlyP)))
        a.scatter(Q[onlyQ, 0], Q[onlyQ, 1], s=22, facecolors="none", edgecolors="tab:orange",
                  label=t("組にならなかった Q の点（%d）" % len(onlyQ), "unmatched Q (%d)" % len(onlyQ)))
        closed = np.where(res["closed"])[0]
        if len(closed):
            a.scatter(Q[closed, 0], Q[closed, 1], s=90, facecolors="none", edgecolors="0.5", marker="s",
                      label=t("解像できない Q の点（%d）" % len(closed), "unresolvable Q (%d)" % len(closed)))
        frame(a, t("重ね合わせ：P を T で Q の座標へ写した（× が P、○ が Q、灰色の枠が Q の視野）",
                   "overlay: P mapped by T onto Q (x = P, o = Q, gray frame = view of Q)"))
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize=9, frameon=False)
    else:
        a.axis("off")
        a.text(0.5, 0.5, t("重ね方を決められなかった", "no overlay"), ha="center", va="center",
               transform=a.transAxes)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(dst, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ 本体
def result_name(path):
    """確認用の図の名前。example1_standard_P.png → example1_standard_result.png"""
    base = os.path.splitext(path)[0]
    return (base[:-2] if base.endswith("_P") else base) + "_result.png"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    path_P = os.path.normpath(os.path.join(here, IMAGE_P))
    path_Q = os.path.normpath(os.path.join(here, IMAGE_Q))
    for name, path in (("IMAGE_P", path_P), ("IMAGE_Q", path_Q)):
        if not os.path.isfile(path):
            sys.exit("画像が見つからない：%s（count_people.py の %s を確かめてください）" % (path, name))

    P, W, H, bigP = read_points(path_P)
    Q, Wq, Hq, bigQ = read_points(path_Q)
    if (W, H) != (Wq, Hq):
        sys.exit("2枚の画像の大きさが違う（%d x %d と %d x %d）。同じ大きさの画像を渡してください" % (W, H, Wq, Hq))
    print("入力      P %s、Q %s" % (IMAGE_P, IMAGE_Q))
    print("読み取り  P %d 点、Q %d 点（画像 %d x %d px）" % (len(P), len(Q), W, H))
    for name, big in (("P", bigP), ("Q", bigQ)):
        if big:
            print("  注意：%s に、ほかの点の2倍より大きい点が %d 個ある。2人が重なって1つに見えているかもしれない" % (name, big))
    if len(P) < 3 or len(Q) < 3:
        sys.exit("点が少なすぎて数えられない")

    out = M.run(P, Q, W, H, M.SET)
    print("")
    if "N_range" in out:
        lo, hi = out["N_range"]
        top = out["top"]
        if lo == hi:
            head = "人数 %d 人" % lo
            print(head)
        else:
            head = "人数 %d〜%d 人（1つに決められなかった）" % (lo, hi)
            print(head)
            if len(out["surv"]) > 1:
                print("  重ね方の候補が %d 本残った（一致率の差が誤差の範囲で、どれが正しいか決めきれない）"
                      % len(out["surv"]))
            if any(u["N_min"] < u["N"] for u in out["surv"]):
                print("  解像できない点がある（近くの人どうしが近すぎて、同じ人か別の人か決められない）")
        A, b = top["A"], top["b"]
        sg = lambda v, f: ("− " if v < 0 else "+ ") + f % abs(v)
        first = lambda v: ("−" if v < 0 else "") + "%.4f" % abs(v)
        print("")
        print("一番よい重ね方（変換 T）")
        print("  P の点 (x, y) は、Q では (x', y') に写る")
        print("  （x は画像の左端から右へ、y は上端から下へ測った位置。単位は画素 px）")
        print("    x' = %s x %s y %s" % (first(A[0, 0]), sg(A[0, 1], "%.4f"), sg(b[0], "%.1f")))
        print("    y' = %s x %s y %s" % (first(A[1, 0]), sg(A[1, 1], "%.4f"), sg(b[1], "%.1f")))
        print("  x' の式：x の前の数は横の倍率、y の前の数は下の行ほど横にずれる度合い")
        print("  y' の式：x の前の数は右の列ほど縦にずれる度合い、y の前の数は縦の倍率")
        print("  最後の数：P の左上の隅 (0, 0) が Q で写る位置")
        print("")
        print("同じ人の組")
        print("  組になった人 %d 人 → 人数 = %d + %d − %d = %d" % (top["M"], len(P), len(Q), top["M"], top["N"]))
        if top["N_min"] < top["N"]:
            print("  解像できない点（下の許容 ε を参照）が、組になりうるだけ組になると、あと %d 組増えて人数 %d 人（範囲の下端）"
                  % (top["N"] - top["N_min"], top["N_min"]))
        print("  一致率 %.2f（Q の視野に写った P の点 %d のうち、組になった割合）" % (top["rate"], top["n_in"]))
        print("")
        print("許容 ε（写した P の点と Q の点が ε 以内なら、同じ人として組にする。1人は1つの組だけ）")
        print("  ε は Q の点ごとに、次の2つの小さいほう")
        print("    ・R = %.1f px" % out["floor"])
        print("        重ね方の候補ごとに「写した P の点から、一番近い Q の点までの距離」の最大を測り、その一番小さい値。")
        print("        正しく重ねても、同じ人どうしがこれだけ離れうる、という見積もり")
        print("    ・その Q の点から、一番近いほかの Q の点までの距離の半分")
        print("        これより近ければ、隣の人と取り違えない")
        print("  解像できない点 %d 個" % int(np.sum(out["closed"])))
        print("    R が「一番近いほかの Q の点までの距離の半分」より大きい Q の点。同じ人か隣の人か決められないので、組にしない")
    else:
        head = "人数を決められなかった（%s）" % out.get("fail", "")
        print(head)
    draw(out, P, Q, W, H, result_name(path_P), head)
    print("")
    print("確認用の図：%s" % result_name(IMAGE_P))


if __name__ == "__main__":
    main()
