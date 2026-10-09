"""
NAYA PATTERN (Aman ke rules):
  Base: W din (≈2–3 mahine) se naya low nahi, range <= R%
  Chupchap volume: base ka volume aur delivery, usse pehle ke W din se zyada; kharidari-din turnover > bikwali-din
  Tags: ⬇ Gira hua (pehle 25%+ gira), 🚀 Toda (base high toda), 🏔️ Life High
Breakout ka wait nahi — pattern dikhte hi signal.
Saare calculations "wide" DataFrames (date x symbol) pe, taaki backtest tez chale.
"""
import numpy as np, pandas as pd

DEFAULT = {"W": 60, "R": 35.0, "VOL": 1.3, "DEL": 1.2, "UD": 1.1, "MIN_TURN_LACS": 20.0}


def wide(h):
    """h: adjusted long data → dict of wide frames"""
    p = lambda c: h.pivot_table(index="date", columns="symbol", values=c).sort_index()
    w = {c: p(c) for c in ["close", "high", "low", "turnover_lacs"]}
    w["vol"] = p("volume_adj") if "volume_adj" in h else p("volume")
    if "deliv_qty" in h:
        dq = h.assign(dq=h["deliv_qty"] * (h["volume_adj"] / h["volume"]).where(h["volume"] > 0, 1) if "volume_adj" in h else h["deliv_qty"])
        w["dq"] = dq.pivot_table(index="date", columns="symbol", values="dq").sort_index()
    return w


def signals(w, prm=None):
    prm = {**DEFAULT, **(prm or {})}
    W = prm["W"]; H = W // 2
    c, hi, lo, v, t = w["close"], w["high"], w["low"], w["vol"], w["turnover_lacs"]
    dq = w.get("dq")
    mp = int(W * 0.8)
    lo_w = lo.rolling(W, min_periods=mp).min()
    hi_w = hi.rolling(W, min_periods=mp).max()
    rng = (hi_w / lo_w - 1) * 100
    lo_first = lo.shift(H).rolling(W - H, min_periods=int((W - H) * 0.8)).min()   # base ka pehla hissa
    lo_second = lo.rolling(H, min_periods=int(H * 0.8)).min()                    # doosra hissa
    no_new_low = lo_second >= lo_first * 0.98
    vol_r = v.rolling(W, min_periods=mp).mean() / v.shift(W).rolling(W, min_periods=mp).mean()
    del_r = (dq.rolling(W, min_periods=mp).mean() / dq.shift(W).rolling(W, min_periods=mp).mean()) if dq is not None else vol_r
    up = c.diff() > 0
    ud = t.where(up, 0).rolling(W, min_periods=mp).sum() / t.where(~up, 0).rolling(W, min_periods=mp).sum().replace(0, np.nan)
    liquid = t.rolling(20, min_periods=15).mean() >= prm["MIN_TURN_LACS"]
    sig = (rng <= prm["R"]) & no_new_low & (vol_r >= prm["VOL"]) & (del_r >= prm["DEL"]) & (ud >= prm["UD"]) & liquid
    # tags
    peak_before = c.shift(W).rolling(250, min_periods=60).max()
    fell = (lo_w / peak_before - 1) * 100 <= -25
    broke = c > hi.shift(5).rolling(W - 5, min_periods=mp - 5).max()
    ath = c >= c.cummax() * 0.999
    lifehigh = (ath.rolling(60, min_periods=40).sum() >= 3) & (c.notna().cumsum() >= 400) & liquid
    return {"sig": sig.fillna(False), "fell": fell.fillna(False), "broke": broke.fillna(False),
            "lifehigh": lifehigh.fillna(False), "range": rng, "vol_r": vol_r, "del_r": del_r, "ud": ud}


def fresh(sig, gap=20):
    """Pehli baar signal (pichhle `gap` din mein signal nahi tha)"""
    prev = sig.shift(1).rolling(gap, min_periods=1).max().fillna(0).astype(bool)
    return sig & ~prev
