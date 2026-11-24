# -*- coding: utf-8 -*-
"""手法本体：段0〜段5 を問題定義 §4 の仕様どおりに実装したもの（既定値 SET は 2026-11-19 時点の仕様）。

仕様が決めていない部分は、この実験のための選択として下の「設定」に明記した。
README.md の「実装上の選択」と対応している。

変換の表し方
    P の点を P の重心で中心化した座標 p~ で表し、
        Q の横 ≒ a11 p~x + a12 p~y + c1
        Q の縦 ≒ a21 p~x + a22 p~y + c2
    とする。(a11, a12) は「横の方向 × 倍率」を直交座標で表したもので、問題定義の
    （方向、倍率、ずれ）と同じ3自由度である。横で3つ、縦で3つ、合わせて自由度6。
"""
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

# ------------------------------------------------------------------ 設定
# 2026-11-11 時点の仕様。実験1〜3 はこれを元に行ったので、記録を再現するために残す
SET_2026_11_11 = dict(
    w_h=40.0,            # 段1 の横の粗いビン幅 [px]
    w_v=16.0,            # 段1 の縦の粗いビン幅 [px]
    a11=(0.85, 1.15),    # 横の倍率の探索範囲
    a12=(-2.5, 2.5),     # 横のせん断の探索範囲（カメラが左右どちらへ動いてもよい）
    a21=(-0.05, 0.05),   # 縦のせん断（ロールの差）の探索範囲
    a22=(0.85, 1.15),    # 縦の倍率の探索範囲
    min_box=8,           # 比べる箱数の下限
    min_n=8,             # 段5 で比べる人数の下限（問題定義 §10 の 1.8 で未決定。実験用に置いた）
    # ---- 診断のための選択肢（仕様には無い。既定値は仕様どおり） ----
    min_people=0,        # 段1・段2 で、共通の台の中に P・Q とも何人以上いることを求めるか
    band_support=False,  # 横の共通の台を、画像の四隅ではなく人が座っている帯の高さで切るか
    floor_source="best_score",  # R をどの候補の値から取るか
                                #   best_score : 段2 の点数が最大の候補（仕様どおり）
                                #   min_D      : 全候補の値の最小値（全候補で共通の1つの値）
    keep_h=20,           # 段1 で残す横の候補の数（局所最大だけを数える）
    keep_c=10,           # 段2 で残す組み合わせの数
    levels=9,            # 段3 で刻みを半分にする回数の上限
    w_min=1.0,           # 段3 のビン幅の下限 [px]
    stop_ratio=0.95,     # 段3 の止めどき：pct で読んだ D が前の段の 95% を下回らなくなったら止める
    pct=95,              # 下限に使う D のパーセンタイル
    # ---- 2026-11-18 に足した選択肢。None なら以前と同じ動き ----
    w_ratio_h=None,      # 横のビン幅を、Q の最近傍距離の中央値 d̄ の何倍にするか（None なら w_h [px]）
    w_ratio_v=None,      # 縦のビン幅（同上。None なら w_v [px]）
    n_min=None,          # 人数の下限を1つにまとめた値（None なら min_people と min_n を別々に使う）
    r_min_coarse=None,   # 段1・段2 で残す一致度の下限（None なら keep_h・keep_c の本数で残す）
    r_min_v=None,        # 段3 の後で課す縦の一致度の下限（None なら課さない）
    c_min=None,          # 段5 の一致率の下限（None なら課さない）
    C_max=None,          # 横の候補・変換の候補の数の上限。超えたら「曖昧」（None なら上限なし）
    ty_max=None,         # 縦のずれの探索範囲を ±ty_max × H にする（None なら −H〜2H。2026-11-19 追加）
    stop_rule="ratio",   # 段3 で最大距離を読む段。"ratio" は stop_ratio で止める、"all" は全段の最小（2026-11-19 追加）
    refit=False,         # 段3 の後に、仮の照合の組から変換を最小二乗で当て直すか（2026-11-19 追加）
    match_outside=False, # 組は距離だけで決め、写した位置が Q の画面の少し外でも組にするか（2026-11-19 追加）
    edge_margin=False,   # 最大距離を測るとき、Q の画面の端から d/2 以内に写る点を外すか（2026-11-19 追加）
    prefilter_f=None,    # 段3 の前の仮の照合で、一致率が最良の (1 − f) 倍に届かない候補を落とす（None なら落とさない）
    # ---- 2026-11-19 のずれの修正。既定値は以前と同じ動き ----
    n_view="inside",     # 視野内の人数 n(T) の数え方。"inside" は以前どおり（段ごとに数え方が違う）。
                         #   "union" は全段で「T(p) が Q の画面の中にある p ＋ 画面のすぐ外で組になった p」にそろえる
    refit_required=False,  # 当て直せなかった候補を落とすか（全候補の最大距離を、当て直した T の1通りで測るため）
    R_strict=False,      # 人数の下限を満たす候補がないとき「曖昧」とするか（False なら全候補から R を取る）
    # ---- 2026-11-19 の仮の組の直し。既定値は以前と同じ動き ----
    pair_both=False,     # 仮の組に、P の側でも「写した P の点の最近傍距離の半分以内」を求めるか
    D_outside=False,     # 最大距離に、画面のすぐ外で仮の組になった点も入れるか（段4 は組を距離だけで決めるので、それに合わせる）
)

# 2026-11-16 時点の仕様（問題定義 §4）。実験4A〜4C はこれを元にしたので、記録を再現するために残す
SET_2026_11_16 = dict(
    SET_2026_11_11,
    min_people=15,         # 台の下限は人数に課す。値は未定（問題定義 §10 の 1.8）で、15 は実験で使った値
    floor_source="min_D",  # R は、人数の下限を満たす候補のうち、最大距離が最も小さい候補から取る
    pct=100,               # 最大距離として D の最大値を読む（100 パーセンタイル ＝ 最大値）
)

# 2026-11-19 時点の仕様（仮の組を直す前）。実験5〜7 はこれを使ったので、記録を再現するために残す。
# 値は点列教室（合成データ）で決めたもので、実際の教室で測った値ではない
SET_2026_11_19 = dict(
    SET_2026_11_16,
    # 探索範囲：撮影の制約（カメラを平行にずらし、縦のずれは小さく）から決める（実験4A）
    a11=(0.9, 1.1), a12=(-0.2, 0.2), a21=(-0.05, 0.05), a22=(0.9, 1.1), ty_max=0.25,
    # 段1・段2（実験4A）
    min_box=3, w_ratio_h=0.35, w_ratio_v=0.25, n_min=5, r_min_coarse=0.66,
    # 段3 の前の仮の照合（実験4C）
    prefilter_f=0.24,
    # 段3：全段の最小、当て直し、端で近くに誰もいない点を外す（実験4B）
    stop_rule="all", refit=True, edge_margin="far",
    # 段4・段5：組は距離だけで決める、一致率の下限（実験4B）。C_max は候補の数の最大から置いた仮の値
    match_outside=True, c_min=0.40, C_max=60,
    # ずれの修正
    n_view="union", refit_required=True, R_strict=True,
)

# 既定値。仮の組の直し（pair_both、D_outside）は、実験8 で新しい種では良くならなかったので入れていない（2026-11-21）
SET = SET_2026_11_19


# ------------------------------------------------------------------ 形の一致度
def grid_scores(Pt, qv, L_ax, corners, w, a1s, a2s, cs, min_box, min_people=0, inner=None,
                return_counts=False):
    """方向1つぶんの格子探索。共通の台は視野から決め、端の欠けた箱は捨てる。

    Pt      P の点（中心化）(nP, 2)
    qv      Q の点のこの方向の座標 (nQ,)
    L_ax    この方向の Q の画像の長さ
    corners P の画像の四隅（中心化）(4, 2)。台は四隅の像の最小〜最大
    inner   (左端の2点, 右端の2点)。与えると台を「左端の像の最大〜右端の像の最小」に
            狭める（帯の全体が P に写っている範囲だけを使う）
    返り値   (len(a1s), len(a2s), len(cs)) の一致度。計算不能と下限不足は nan
             return_counts=True なら、台の中の P と Q の人数も同じ形で返す
    """
    nb = int(np.ceil(L_ax / w))
    hQ = np.bincount(np.clip(np.floor(qv / w).astype(int), 0, nb - 1),
                     minlength=nb).astype(float)
    cQ = np.concatenate([[0.0], np.cumsum(hQ)])
    cQQ = np.concatenate([[0.0], np.cumsum(hQ * hQ)])
    cs = np.asarray(cs, dtype=float)
    Nc = len(cs)
    out = np.full((len(a1s), len(a2s), Nc), np.nan)
    cntP = np.zeros((len(a1s), len(a2s), Nc))
    cntQ = np.zeros((len(a1s), len(a2s), Nc))
    rows = np.arange(Nc)[:, None]
    for i, a1 in enumerate(a1s):
        for j, a2 in enumerate(a2s):
            m = a1 * Pt[:, 0] + a2 * Pt[:, 1]
            if inner is None:
                cv = a1 * corners[:, 0] + a2 * corners[:, 1]
                lo0, hi0 = cv.min(), cv.max()
            else:
                left, right = inner
                lo0 = (a1 * left[:, 0] + a2 * left[:, 1]).max()
                hi0 = (a1 * right[:, 0] + a2 * right[:, 1]).min()
            lo = np.maximum(0.0, lo0 + cs)
            hi = np.minimum(L_ax, hi0 + cs)
            klo = np.ceil(lo / w).astype(int)
            khi = np.floor(hi / w).astype(int)
            K = khi - klo
            kP = np.floor((m[None, :] + cs[:, None]) / w).astype(int)
            ok = (kP >= klo[:, None]) & (kP < khi[:, None])
            flat = (np.broadcast_to(rows, kP.shape)[ok] * nb + kP[ok])
            hP = np.bincount(flat, minlength=Nc * nb).reshape(Nc, nb).astype(float)
            klo_c = np.clip(klo, 0, nb)
            khi_c = np.clip(khi, 0, nb)
            SQ = cQ[khi_c] - cQ[klo_c]
            SQQ = cQQ[khi_c] - cQQ[klo_c]
            SP = hP.sum(1)
            SPP = (hP * hP).sum(1)
            SPQ = (hP * hQ[None, :]).sum(1)
            Kf = np.maximum(K, 1).astype(float)
            num = SPQ - SP * SQ / Kf
            vP = SPP - SP * SP / Kf
            vQ = SQQ - SQ * SQ / Kf
            good = (K >= min_box) & (vP > 1e-9) & (vQ > 1e-9) & \
                (SP >= min_people) & (SQ >= min_people)
            r = np.full(Nc, np.nan)
            r[good] = num[good] / np.sqrt(vP[good] * vQ[good])
            out[i, j] = r
            cntP[i, j], cntQ[i, j] = SP, SQ
    if return_counts:
        return out, cntP, cntQ
    return out


def axis(lo, hi, step):
    """lo〜hi を step 刻みで覆う格子（両端を含む）。"""
    n = max(int(np.ceil((hi - lo) / step)), 0) + 1
    return lo + step * np.arange(n)


# ------------------------------------------------------------------ 道具
class Ctx(object):
    """1回の照合で使う共通の量。"""

    def __init__(self, P, Q, W, H, s=SET):
        self.P, self.Q, self.W, self.H, self.s = P, Q, float(W), float(H), s
        self.mP = P.mean(axis=0)
        self.Pt = P - self.mP
        self.corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], float) - self.mP
        self.Xext = max(np.abs(self.Pt[:, 0]).max(), 1.0)
        self.Yext = max(np.abs(self.Pt[:, 1]).max(), 1.0)
        # Q の最近傍距離の中央値 d̄（ビン幅を比で与えるときの尺度）
        Dq = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=2)
        np.fill_diagonal(Dq, np.inf)
        self.dq = Dq.min(axis=1)
        self.dbar = float(np.median(self.dq)) if len(Q) > 1 else 1.0
        # 実際に使うビン幅と人数の下限（比や1つにまとめた値が与えられていればそちらを使う）
        self.w_h = s["w_h"] if s.get("w_ratio_h") is None else s["w_ratio_h"] * self.dbar
        self.w_v = s["w_v"] if s.get("w_ratio_v") is None else s["w_ratio_v"] * self.dbar
        self.min_people = s["min_people"] if s.get("n_min") is None else s["n_min"]
        self.min_n = s["min_n"] if s.get("n_min") is None else s["n_min"]
        # 帯で台を切る場合：P の点が占める縦の範囲で、P の視野の左端と右端を切る
        self.inner_h = None
        if s.get("band_support"):
            v0, v1 = P[:, 1].min(), P[:, 1].max()
            left = np.array([[0, v0], [0, v1]], float) - self.mP
            right = np.array([[W, v0], [W, v1]], float) - self.mP
            self.inner_h = (left, right)

    def steps(self, w, which):
        """ビン幅 w に対応する刻み。端の点の動きが w を超えないように取る。"""
        return (w / self.Xext, w / self.Yext, w / 2.0)

    def affine(self, h, v):
        A = np.array([[h[0], h[1]], [v[0], v[1]]])
        b = np.array([h[2], v[2]]) - A @ self.mP
        return A, b

    def h_overlap(self, h):
        """横の仮説が主張する重なりの範囲（Q の横座標）。"""
        if self.inner_h is None:
            cv = h[0] * self.corners[:, 0] + h[1] * self.corners[:, 1] + h[2]
            return max(0.0, cv.min()), min(self.W, cv.max())
        left, right = self.inner_h
        lo = (h[0] * left[:, 0] + h[1] * left[:, 1]).max() + h[2]
        hi = (h[0] * right[:, 0] + h[1] * right[:, 1]).min() + h[2]
        return max(0.0, lo), min(self.W, hi)

    def v_subsets(self, h):
        """縦の数列は、横の仮説が主張する重なり列に限って作り直す（§4）。"""
        lo, hi = self.h_overlap(h)
        mx = h[0] * self.Pt[:, 0] + h[1] * self.Pt[:, 1] + h[2]
        selP = (mx >= lo) & (mx < hi)
        selQ = (self.Q[:, 0] >= lo) & (self.Q[:, 0] < hi)
        return self.Pt[selP], self.Q[selQ, 1]


def score_h(ctx, h, w):
    a1, a2, c = h
    return grid_scores(ctx.Pt, ctx.Q[:, 0], ctx.W, ctx.corners, w,
                       [a1], [a2], [c], ctx.s["min_box"],
                       ctx.min_people, ctx.inner_h)[0, 0, 0]


def score_v(ctx, h, v, w):
    Pt, qv = ctx.v_subsets(h)
    if len(Pt) < 2 or len(qv) < 2:
        return np.nan
    return grid_scores(Pt, qv, ctx.H, ctx.corners, w,
                       [v[0]], [v[1]], [v[2]], ctx.s["min_box"],
                       ctx.min_people)[0, 0, 0]


def peaks(S, r_min=None, keep=None):
    """局所最大（同じ山の格子点は1本）を、点数の高い順に返す。
    r_min を与えればその値以上をすべて、与えなければ上位 keep 本。"""
    Sf = np.where(np.isnan(S), -np.inf, S)
    peak = (Sf == maximum_filter(Sf, size=3, mode="nearest")) & np.isfinite(Sf)
    idx = np.argwhere(peak)
    vals = Sf[peak]
    order = np.argsort(-vals)
    if r_min is not None:
        order = order[vals[order] >= r_min]
    elif keep is not None:
        order = order[:keep]
    return [(tuple(idx[o]), float(vals[o])) for o in order]


# ------------------------------------------------------------------ 段1
def stage1_grid(ctx):
    """段1 の探索格子（横の仮説の並び）。"""
    s, w = ctx.s, ctx.w_h
    d1, d2, dc = ctx.steps(w, "h")
    a1s = axis(s["a11"][0], s["a11"][1], d1)
    a2s = axis(s["a12"][0], s["a12"][1], d2)
    cv_all = []
    for a1 in (a1s[0], a1s[-1]):
        for a2 in (a2s[0], a2s[-1]):
            cv_all.append(a1 * ctx.corners[:, 0] + a2 * ctx.corners[:, 1])
    cv_all = np.concatenate(cv_all)
    cs = axis(-cv_all.max(), ctx.W - cv_all.min(), dc)
    return a1s, a2s, cs


def stage1(ctx):
    """横の粗い探索。局所最大のうち、r_min_coarse 以上（未設定なら上位 keep_h 本）を残す。"""
    s, w = ctx.s, ctx.w_h
    a1s, a2s, cs = stage1_grid(ctx)
    S = grid_scores(ctx.Pt, ctx.Q[:, 0], ctx.W, ctx.corners, w, a1s, a2s, cs, s["min_box"],
                    ctx.min_people, ctx.inner_h)
    pk = peaks(S, s.get("r_min_coarse"), s["keep_h"])
    cands = [dict(h=(a1s[i], a2s[j], cs[k]), r_h=r) for (i, j, k), r in pk]
    Sf = np.where(np.isnan(S), -np.inf, S)
    return cands, int(np.isfinite(Sf).sum()), Sf


# ------------------------------------------------------------------ 段2
def stage2_grid(ctx):
    """段2 の探索格子（縦の仮説の並び）。縦のずれは ty_max があればその範囲に限る。"""
    s, w = ctx.s, ctx.w_v
    d1, d2, dc = ctx.steps(w, "v")
    a1s = axis(s["a21"][0], s["a21"][1], d1)
    a2s = axis(s["a22"][0], s["a22"][1], d2)
    if s.get("ty_max") is None:
        cs = axis(-ctx.H, 2 * ctx.H, dc)
    else:
        m = ctx.mP[1]
        cs = axis(m - s["ty_max"] * ctx.H, m + s["ty_max"] * ctx.H, dc)
    return a1s, a2s, cs


def stage2(ctx, hcands):
    """横の候補ごとに縦を探し、組にする。結合は最小値（両方で整合することを要求）。
    r_min_coarse が未設定なら、横1本につき縦の最良1本を組にして上位 keep_c 本を残す（以前の動き）。
    設定されていれば、縦の局所最大のうち r_min_coarse 以上をすべて組にする。"""
    s, w = ctx.s, ctx.w_v
    r_min = s.get("r_min_coarse")
    out = []
    n_eval = 0
    for hc in hcands:
        Pt, qv = ctx.v_subsets(hc["h"])
        if len(Pt) < 2 or len(qv) < 2:
            continue
        a1s, a2s, cs = stage2_grid(ctx)
        S = grid_scores(Pt, qv, ctx.H, ctx.corners, w, a1s, a2s, cs, s["min_box"],
                        ctx.min_people)
        n_eval += int(np.isfinite(S).sum())
        if not np.isfinite(S).any():
            continue
        if r_min is None:
            i, j, k = np.unravel_index(np.nanargmax(S), S.shape)
            vs = [((i, j, k), float(S[i, j, k]))]
        else:
            vs = peaks(S, r_min)
        for (i, j, k), r_v in vs:
            out.append(dict(h=hc["h"], v=(a1s[i], a2s[j], cs[k]), r_h=hc["r_h"], r_v=r_v,
                            r=min(hc["r_h"], r_v)))
    out.sort(key=lambda c: -c["r"])
    if r_min is None:
        out = out[: s["keep_c"]]
    return out, n_eval


# ------------------------------------------------------------------ 段3
def nn_dist(ctx, A, b):
    """T(p) から Q の最近傍までの距離。Q の画像の中に写った p だけ。
    edge_margin が真なら、Q の画面の端から d(q)/2 以内に写る p は外す
    （q は T(p) に一番近い Q の点。本人が画面のすぐ外にいるかもしれないため）。"""
    TP = ctx.P @ A.T + b
    inside = (TP[:, 0] >= 0) & (TP[:, 0] < ctx.W) & (TP[:, 1] >= 0) & (TP[:, 1] < ctx.H)
    if inside.sum() == 0:
        return np.array([]), inside, TP
    Dm = np.linalg.norm(TP[inside][:, None, :] - ctx.Q[None, :, :], axis=2)
    D = Dm.min(axis=1)
    if ctx.s.get("edge_margin"):
        T_in = TP[inside]
        margin = np.minimum.reduce([T_in[:, 0], ctx.W - T_in[:, 0], T_in[:, 1], ctx.H - T_in[:, 1]])
        half = ctx.dq[Dm.argmin(axis=1)] / 2.0
        near_edge = margin < half
        if ctx.s.get("edge_margin") == "far":
            # 端の近くにあり、かつ近くに誰もいない（D > d/2）点だけを外す（2026-11-19 追加）
            keep = ~(near_edge & (D > half))
        else:
            keep = ~near_edge
        if keep.any():
            D = D[keep]
    if ctx.s.get("D_outside"):
        # 画面のすぐ外に写ったが、仮の組になった点も入れる（段4 ではこの点も組にするため）
        ext = [np.linalg.norm(TP[i] - ctx.Q[j]) for i, j in provisional_pairs(ctx, A, b) if not inside[i]]
        D = np.concatenate([D, ext])
    return D, inside, TP


def climb(f, x, steps, iters=30):
    """3×3×3 の近傍で山登りする。"""
    x = tuple(x)
    best = f(x)
    best = -np.inf if np.isnan(best) else best
    for _ in range(iters):
        moved = False
        for d0 in (-1, 0, 1):
            for d1 in (-1, 0, 1):
                for d2 in (-1, 0, 1):
                    if d0 == d1 == d2 == 0:
                        continue
                    y = (x[0] + d0 * steps[0], x[1] + d1 * steps[1], x[2] + d2 * steps[2])
                    val = f(y)
                    if not np.isnan(val) and val > best + 1e-12:
                        best, xb, moved = val, y, True
        if not moved:
            break
        x = xb
    return x, best


def stage3(ctx, cand, truth_pairs=None):
    """刻みとビン幅を半分にしながら精密化し、各段で D を記録する。"""
    s = ctx.s
    h, v = cand["h"], cand["v"]
    wh, wv = ctx.w_h, ctx.w_v
    sh, sv = ctx.steps(wh, "h"), ctx.steps(wv, "v")
    levels = []
    for lev in range(s["levels"]):
        if lev > 0:
            wh, wv = max(wh / 2, s["w_min"]), max(wv / 2, s["w_min"])
            sh = tuple(x / 2 for x in sh)
            sv = tuple(x / 2 for x in sv)
            h, _ = climb(lambda x: score_h(ctx, x, wh), h, sh)
            v, _ = climb(lambda x: score_v(ctx, h, x, wv), v, sv)
        A, b = ctx.affine(h, v)
        D, inside, TP = nn_dist(ctx, A, b)
        rec = dict(level=lev, w_h=wh, w_v=wv, A=A, b=b, h=h, v=v,
                   D95=float(np.percentile(D, s["pct"])) if len(D) else np.nan,
                   D50=float(np.median(D)) if len(D) else np.nan, n_in=int(inside.sum()))
        if truth_pairs is not None and len(truth_pairs):
            i, j = truth_pairs[:, 0], truth_pairs[:, 1]
            rt = np.linalg.norm(TP[i] - ctx.Q[j], axis=1)
            rec["true95"] = float(np.percentile(rt, s["pct"]))
        levels.append(rec)
    # 止めどき：D の95%点が前の段の stop_ratio 倍を下回らなくなった最初の段
    # （stop_rule="all" なら止めずに全段を見て、その最小を最大距離にする。2026-11-19 追加）
    stop = len(levels) - 1
    for k in range(1, len(levels)):
        if s.get("stop_rule", "ratio") == "all":
            break
        if not (levels[k]["D95"] < s["stop_ratio"] * levels[k - 1]["D95"]):
            stop = k
            break
    best_k = int(np.nanargmin([levels[k]["D95"] for k in range(stop + 1)]))
    return levels, stop, best_k


# ------------------------------------------------------------------ ε
def epsilons(Q, floor):
    """ε_q = min(R, d_q / 2)。R > d_q / 2 の点は窓が閉じている（解像できない）。"""
    Dq = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=2)
    np.fill_diagonal(Dq, np.inf)
    d = Dq.min(axis=1)
    eps = np.minimum(floor, d / 2.0)
    closed = floor > d / 2.0
    return eps, closed, d


# ------------------------------------------------------------------ 当て直し（2026-11-19 追加）
def provisional_pairs(ctx, A, b):
    """仮の照合：Q の視野に写った p と q が互いに最近傍で、かつ d(q)/2 以内の組。
    pair_both なら、写した P の点の最近傍距離 d(p) の半分以内も求める。周りに人がいない q は d(q)/2 が
    大きいので、q の側だけで測ると、相手のいない p と離れたまま組になりうるため（2026-11-19 追加）。
    d(q) と d(p) は Q と P の中だけで測れるので、R を使わずに組を決められる（循環しない）。"""
    TP = ctx.P @ A.T + b
    inside = (TP[:, 0] >= 0) & (TP[:, 0] < ctx.W) & (TP[:, 1] >= 0) & (TP[:, 1] < ctx.H)
    idx = np.arange(len(TP)) if ctx.s.get("match_outside") else np.where(inside)[0]
    if len(idx) == 0:
        return []
    Dm = np.linalg.norm(TP[idx][:, None, :] - ctx.Q[None, :, :], axis=2)
    qi = Dm.argmin(axis=1)
    pj = Dm.argmin(axis=0)
    lim = ctx.dq[qi] / 2.0
    if ctx.s.get("pair_both"):
        Dp = np.linalg.norm(TP[:, None, :] - TP[None, :, :], axis=2)
        np.fill_diagonal(Dp, np.inf)
        lim = np.minimum(lim, Dp.min(axis=1)[idx] / 2.0)
    return [(int(idx[i]), int(qi[i])) for i in range(len(idx))
            if pj[qi[i]] == i and Dm[i, qi[i]] <= lim[i]]


def n_view(ctx, A, b, paired=()):
    """視野内の人数 n(T)：T(p) が Q の画面の中にある p と、画面のすぐ外で組になった p（2026-11-19 にそろえた）。"""
    TP = ctx.P @ A.T + b
    inside = (TP[:, 0] >= 0) & (TP[:, 0] < ctx.W) & (TP[:, 1] >= 0) & (TP[:, 1] < ctx.H)
    if len(paired):
        inside[list(paired)] = True
    return int(inside.sum())


def refit(ctx, A, b, iters=3):
    """仮の照合の組からアフィンを最小二乗で当て直す。組が n_min 未満なら当て直さない。"""
    used = 0
    for _ in range(iters):
        pr = provisional_pairs(ctx, A, b)
        if len(pr) < max(ctx.min_n, 3):
            break
        Pm = ctx.P[[i for i, _ in pr]]
        Qm = ctx.Q[[j for _, j in pr]]
        X = np.hstack([Pm, np.ones((len(Pm), 1))])
        Mx, _, _, _ = np.linalg.lstsq(X, Qm, rcond=None)
        A2, b2 = Mx[:2].T, Mx[2]
        used = len(pr)
        if np.allclose(A2, A) and np.allclose(b2, b):
            break
        A, b = A2, b2
    return A, b, used


# ------------------------------------------------------------------ 段4
def stage4(ctx, A, b, eps, closed, floor):
    """ε_q 以内を辺とする二部グラフの最大マッチング。相互最近傍の一致率も取る。

    解像できない点（窓が閉じた Q の点）について、残った P の点とR 以内で
    いくつ対応しうるかも数える（区間の下端に使う）。
    """
    TP = ctx.P @ A.T + b
    inside = (TP[:, 0] >= 0) & (TP[:, 0] < ctx.W) & (TP[:, 1] >= 0) & (TP[:, 1] < ctx.H)
    Dm = np.linalg.norm(TP[:, None, :] - ctx.Q[None, :, :], axis=2)
    # match_outside なら、組は距離だけで決める（q は必ず画面の中にあるので、T(p) が少し外でもよい）
    cand = np.ones(len(TP), bool) if ctx.s.get("match_outside") else inside
    edge = (Dm <= eps[None, :]) & (~closed)[None, :] & cand[:, None]
    match = maximum_bipartite_matching(csr_matrix(edge.astype(np.int8)), perm_type="column")
    pairs = [(i, int(j)) for i, j in enumerate(match) if j >= 0]
    # 解像できない点が、もし対応していたらいくつ対応しうるか（区間の下端に使う）
    usedP = np.zeros(len(TP), bool)
    usedP[[i for i, _ in pairs]] = True
    if ctx.s.get("match_outside"):
        inside = inside | usedP              # 視野内の人数には、少し外で組になった人も入れる
    edge_u = (Dm <= floor) & closed[None, :] & cand[:, None] & (~usedP)[:, None]
    m_u = 0
    if edge_u.any():
        mu = maximum_bipartite_matching(csr_matrix(edge_u.astype(np.int8)), perm_type="column")
        m_u = int((mu >= 0).sum())
    # 相互最近傍の一致率（見張り）
    mutual = np.nan
    if inside.sum() > 0:
        Din = Dm[inside]
        q_star = Din.argmin(axis=1)
        p_star = Din.argmin(axis=0)
        mutual = float(np.mean(p_star[q_star] == np.arange(inside.sum())))
    return pairs, int(inside.sum()), mutual, m_u


# ------------------------------------------------------------------ 段5
def same_T(ctx, A1, b1, A2, b2, tol):
    d = np.linalg.norm((ctx.P @ A1.T + b1) - (ctx.P @ A2.T + b2), axis=1)
    return float(np.mean(d)) < tol


def run(P, Q, W, H, s=SET, truth_pairs=None):
    """段0〜段5 を通す。途中経過をすべて返す。"""
    ctx = Ctx(P, Q, W, H, s)
    out = dict(a=len(P), b=len(Q), dbar=ctx.dbar, w_h=ctx.w_h, w_v=ctx.w_v)
    C_max = s.get("C_max")

    # 段1・段2（候補が C_max を超えたら「曖昧」）
    hc, n1, _ = stage1(ctx)
    out.update(n_eval1=n1, hcands=hc)
    if C_max is not None and len(hc) > C_max:
        out["fail"] = "曖昧：横の候補が C_max を超えた"
        return out
    cc, n2 = stage2(ctx, hc)
    out.update(n_eval2=n2, ccands=cc)
    if not cc:
        out["fail"] = "段2 で候補が残らなかった"
        return out
    # 段3 の前の仮の照合：粗い変換のまま一致率を測り、最良から大きく離れた候補を落とす
    if s.get("prefilter_f") is not None:
        for c in cc:
            A, b = ctx.affine(c["h"], c["v"])
            pr = provisional_pairs(ctx, A, b)
            if s.get("n_view") == "union":
                n = n_view(ctx, A, b, [i for i, _ in pr])
            else:
                _, inside, _ = nn_dist(ctx, A, b)
                n = int(inside.sum())
            c["pre_rate"] = len(pr) / max(n, 1)
        best = max(c["pre_rate"] for c in cc)
        cc = [c for c in cc if c["pre_rate"] >= (1 - s["prefilter_f"]) * best]
        out.update(ccands_pre=cc)
    if C_max is not None and len(cc) > C_max:
        out["fail"] = "曖昧：変換の候補が C_max を超えた"
        return out

    # 段3：全候補を精密化する
    refined = []
    for c in cc:
        levels, stop, best_k = stage3(ctx, c, truth_pairs)
        lv = levels[best_k]
        refined.append(dict(cand=c, levels=levels, stop=stop, best_k=best_k,
                            A=lv["A"], b=lv["b"],
                            r_v_final=score_v(ctx, lv["h"], lv["v"], ctx.w_v)))
    # 精密化の後で、縦を厳しく判定する（粗いビン幅 w_v で採点し直す）
    if s.get("r_min_v") is not None:
        out["n_dropped_v"] = sum(1 for r in refined
                                 if not (r["r_v_final"] >= s["r_min_v"]))
        refined = [r for r in refined if r["r_v_final"] >= s["r_min_v"]]
        if not refined:
            out["fail"] = "曖昧：縦の下限を満たす候補がない"
            return out
    for r in refined:
        r["floor"] = r["levels"][r["best_k"]]["D95"]
        r["n_in0"] = r["levels"][r["best_k"]]["n_in"]
    # 当て直し：仮の照合の組から変換を詰め直し、その変換で最大距離を測り直す
    if s.get("refit"):
        for r in refined:
            A2, b2, used = refit(ctx, r["A"], r["b"])
            r["refit_pairs"] = used
            r["refit_ok"] = False
            if used:
                D, inside, _ = nn_dist(ctx, A2, b2)
                if len(D):
                    r["A"], r["b"] = A2, b2
                    r["floor"] = float(np.percentile(D, s["pct"]))
                    r["n_in0"] = int(inside.sum())
                    r["refit_ok"] = True
        # 当て直せなかった候補は、最大距離の測り方が違うので R の比較に入れない
        if s.get("refit_required"):
            refined = [r for r in refined if r["refit_ok"]]
            if not refined:
                out["fail"] = "曖昧：当て直せる候補がない"
                return out
    if s.get("n_view") == "union":
        for r in refined:
            r["n_in0"] = n_view(ctx, r["A"], r["b"],
                                [i for i, _ in provisional_pairs(ctx, r["A"], r["b"])])
    if s["floor_source"] == "min_D":
        pool = [r for r in refined if r["n_in0"] >= ctx.min_n]
        if not pool:
            if s.get("R_strict"):
                out.update(refined=refined)
                out["fail"] = "曖昧：人数の下限を満たす候補がなく、R を決められない"
                return out
            pool = refined
        floor = float(np.nanmin([r["floor"] for r in pool]))
    else:
        floor = refined[0]["floor"]
    out.update(refined=refined, floor=floor)

    # ε（段0 の最近傍距離と、段3 の R から）
    eps, closed, dq = epsilons(Q, floor)
    out.update(eps=eps, closed=closed, dq=dq)

    # 段4：全候補を同じ ε_q で照合する
    for r in refined:
        pairs, n_in, mutual, m_u = stage4(ctx, r["A"], r["b"], eps, closed, floor)
        N = len(P) + len(Q) - len(pairs)
        r.update(pairs=pairs, M=len(pairs), n_in=n_in, mutual=mutual,
                 rate=len(pairs) / n_in if n_in else 0.0, N=N, N_min=N - m_u)

    # 段5：視野内の人数が下限に満たない候補と、一致率が c_min に満たない候補を外し、
    #       同じ変換に収束した候補をまとめ、一致率で並べ、Δ_B 以内を残す
    c_min = s.get("c_min")
    enough = [r for r in refined if r["n_in"] >= ctx.min_n
              and (c_min is None or r["rate"] >= c_min)]
    out["n_dropped_small"] = len(refined) - len(enough)
    if not enough:
        out["fail"] = "曖昧：人数の下限と一致率の下限を満たす候補がない"
        return out
    uniq = []
    for r in sorted(enough, key=lambda x: -x["rate"]):
        if not any(same_T(ctx, r["A"], r["b"], u["A"], u["b"], max(floor, 1.0)) for u in uniq):
            uniq.append(r)
    out["uniq"] = uniq
    top = uniq[0]
    if len(uniq) >= 2:
        r2, n2_ = uniq[1]["rate"], max(uniq[1]["n_in"], 1)
        dB = 2.0 * np.sqrt(max(r2 * (1 - r2), 0.0) / n2_)
        surv = [u for u in uniq if u["rate"] >= top["rate"] - dB]
    else:
        dB, surv = 0.0, [top]
    out.update(dB=dB, surv=surv, top=top)
    # 出力：残った候補の人数の範囲。解像できない点がある場合は、それらが対応して
    # いた場合（下端）と対応していなかった場合（上端）の両方を含める
    out["N_out"] = (min(u["N"] for u in surv), max(u["N"] for u in surv))
    out["N_range"] = (min(u["N_min"] for u in surv), max(u["N"] for u in surv))
    return out
