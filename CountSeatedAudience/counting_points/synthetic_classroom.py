# -*- coding: utf-8 -*-
"""点列教室：カメラを使わず、点列だけで2枚ぶんのデータをつくる（2026-11-19 追加）。

1つの座席配置（画素座標）から、P と Q の視野を切り出す。
Q の視野は P の視野から、横に大きく（tx）、縦に少し（ty）ずれている。
座席の位置は2枚で共通で、検出のずれ（delta）だけが2枚で独立に入る。
縦のずれがあるので、上端と下端の行の人は片方の画像にしか写らない。
検出漏れ（miss）は、各画像で各人を確率 miss で落とす。2枚で独立（2026-11-19 追加。0 なら以前と同じ点列）。

正しい変換：Q の座標 = A (P の座標 − c0) + c0 − (tx, ty)。c0 は P の視野の中心
    A = 倍率 × ロールの回転 × せん断 [[1, shear], [0, 1]]（2026-11-19 追加）
    倍率 1、せん断 0、ロール 0（既定）なら A は単位行列で、以前と同じ平行移動だけの点列になる
正しい人数 N_true は「少なくとも片方の画像で検出された人の数」（どちらでも検出されない人は数えようがない）
"""
import numpy as np

DEFAULT = dict(
    cols=18,          # 列数
    rows=9,           # 行数
    sx=100.0,         # 席の横の間隔 [px]
    sy=70.0,          # 席の縦の間隔 [px]
    occ=0.55,         # 着席率
    sd_seat=10.0,     # 座る位置のゆらぎ [px]（2枚で共通）
    W=1000,           # 視野の幅 [px]
    H=560,            # 視野の高さ [px]
    tx=500.0,         # Q の視野の横のずれ [px]
    ty=42.0,          # Q の視野の縦のずれ [px]
    delta=3.0,        # 検出のずれ [px]（各画像の各点に独立に足す標準偏差）
    miss=0.0,         # 検出漏れの率（各画像で独立）
    scale=1.0,        # 倍率（撮る距離やズームの違い）
    shear=0.0,        # せん断：横のずれが縦の位置で変わる割合（横に動いたときの視差）
    roll=0.0,         # ロール：光軸のまわりの回転 [度]（カメラの傾き）
)

# 倍率・せん断・ロールを引く範囲。手法の探索範囲（a11, a22 は 0.9〜1.1、a12 は ±0.2、a21 は ±0.05）に
# 余裕をもって収まるように取る（2026-11-19）
AFFINE_RANGE = dict(scale=(0.95, 1.05), shear=(-0.12, 0.12), roll=(-2.0, 2.0))


def draw_affine(seed):
    """倍率・せん断・ロールを AFFINE_RANGE から一様に引く。点列の乱数とは別の乱数を使う。"""
    rng = np.random.default_rng([seed, 1])
    return {k: float(rng.uniform(*v)) for k, v in AFFINE_RANGE.items()}


def affine_matrix(p):
    """A = 倍率 × ロールの回転 × せん断。"""
    t = np.deg2rad(p["roll"])
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return p["scale"] * rot @ np.array([[1.0, p["shear"]], [0.0, 1.0]])


def generate(prm, seed):
    p = dict(DEFAULT)
    p.update(prm)
    rng = np.random.default_rng(seed)
    cc, rr = np.meshgrid(np.arange(p["cols"]), np.arange(p["rows"]))
    cc, rr = cc.ravel(), rr.ravel()
    seated = rng.random(cc.size) < p["occ"]
    cc, rr = cc[seated], rr[seated]
    n = cc.size
    base = np.stack([(cc + 0.5) * p["sx"] + rng.normal(0, p["sd_seat"], n),
                     (rr + 0.5) * p["sy"] + rng.normal(0, p["sd_seat"], n)], axis=1)
    uvP = base + rng.normal(0, p["delta"], (n, 2))
    shift = np.array([p["tx"], p["ty"]])
    A = affine_matrix(p)
    if np.array_equal(A, np.eye(2)):
        mapped, b = base - shift, -shift          # 以前と同じ計算（点列をそのまま再現するため）
    else:
        c0 = np.array([p["W"], p["H"]], float) / 2.0
        mapped, b = (base - c0) @ A.T + c0 - shift, c0 - shift - A @ c0
    uvQ = mapped + rng.normal(0, p["delta"], (n, 2))

    def inside(uv):
        return (uv[:, 0] >= 0) & (uv[:, 0] < p["W"]) & (uv[:, 1] >= 0) & (uv[:, 1] < p["H"])

    # 検出漏れ。miss = 0 のときは乱数を引かない（以前の点列をそのまま再現するため）
    if p["miss"] > 0:
        hitP, hitQ = rng.random(n) >= p["miss"], rng.random(n) >= p["miss"]
    else:
        hitP = hitQ = np.ones(n, bool)
    idP, idQ = np.where(inside(uvP) & hitP)[0], np.where(inside(uvQ) & hitQ)[0]
    posP = {k: i for i, k in enumerate(idP)}
    posQ = {k: j for j, k in enumerate(idQ)}
    both = np.intersect1d(idP, idQ)
    pairs = np.array([(posP[k], posQ[k]) for k in both], dtype=int).reshape(-1, 2)
    P, Q = uvP[idP], uvQ[idQ]
    resid = np.linalg.norm(P[pairs[:, 0]] @ A.T + b - Q[pairs[:, 1]], axis=1)
    return dict(prm=p, P=P, Q=Q, pairs=pairs, a=len(P), b=len(Q),
                N_true=len(np.union1d(idP, idQ)), M_true=len(both), seated=n,
                N_present=int((inside(uvP) | inside(uvQ)).sum()),
                A_true=A, b_true=b, resid_true=resid, rows_P=rr[idP], rows_Q=rr[idQ])
