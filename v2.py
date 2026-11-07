"""
SMART MONEY v2 — naye sire se (Aman ke rules, Nov 2026)
Groups : Screener ki official basic industry; jahan meri value-chain/niche group (micro_industries) hai wo override
Size   : < ₹400 cr bahar · Small 400–2000 · Mid 2000–20000 · Large 20000+   (koi turnover filter nahi)
Tags   : 🏗️ Base (sabse lambi range ≤35%, 2M–12M) · 📏 MA squeeze · 📈 Volume aa raha · 🔜 Breakout ke paas · 🚀 BO · 🏃 Chal raha
Industry "🎯 Taiyaar": >=2 (ya 25%) shares setup mein (Base + Volume) AND >=1 share chal raha/BO (pichhle 20 din)
"""
import os, glob
import numpy as np, pandas as pd
import pipeline as P, pattern as PT, micro_industries as M

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_LENS = [40, 60, 90, 120, 180, 250]   # trading days (~2M .. 12M)
BASE_MAX_RANGE = 35.0


def groups():
    """symbol → group name, size info"""
    sc = pd.read_csv(os.path.join(ROOT, "data", "screener_industry.csv"))
    sc = sc[sc["mcap_cr"] >= 400]
    micro = {}
    for g, syms in M.GROUPS.items():
        for s in syms.split():
            micro.setdefault(s, g)
    sc["group"] = [micro.get(s, b) for s, b in zip(sc["symbol"], sc["basic_industry"])]
    return sc.set_index("symbol")


def load_prices(symbols, hist=True):
    files = sorted(glob.glob(os.path.join(ROOT, "data", "hist", "*.parquet"))) if hist else []
    h = pd.concat([pd.read_parquet(f) for f in files]) if files else pd.read_parquet(os.path.join(ROOT, "data", "bhav.parquet"))
    h = h[h["symbol"].isin(symbols) & h["series"].isin(["EQ", "BE", "BZ"])].drop_duplicates(["date", "symbol"], keep="last")
    return PT.wide(P.adjust_prices(h))


def tags(w):
    c, hi, lo, v = w["close"], w["high"], w["low"], w["vol"]
    dq = w.get("dq")
    # Base: sabse lambi range jo <= 35% ho; base high = us range ka high (aakhri 5 din chhod ke)
    base_len = pd.DataFrame(0, index=c.index, columns=c.columns)
    base_hi = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    for L in BASE_LENS:
        h_ = hi.shift(5).rolling(L - 5, min_periods=int((L - 5) * 0.8)).max()
        l_ = lo.rolling(L, min_periods=int(L * 0.8)).min()
        ok = ((h_ / l_ - 1) * 100 <= BASE_MAX_RANGE) & (c <= h_ * 1.10)       # range tight, abhi base ke andar/just upar
        base_len = base_len.where(~ok, L)
        base_hi = base_hi.where(~ok, h_)
    in_base = base_len > 0
    sma = {n: c.rolling(n, min_periods=int(n * 0.8)).mean() for n in (20, 50, 100)}
    mx = pd.concat(sma.values()).groupby(level=0).max() if False else np.maximum(np.maximum(sma[20], sma[50]), sma[100])
    mn = np.minimum(np.minimum(sma[20], sma[50]), sma[100])
    squeeze = (mx / mn <= 1.05) & (c >= mn * 0.97)
    vol_r = v.rolling(30, min_periods=24).mean() / v.shift(30).rolling(90, min_periods=70).mean()
    del_r = (dq.rolling(30, min_periods=24).mean() / dq.shift(30).rolling(90, min_periods=70).mean()) if dq is not None else vol_r
    vol_in = (vol_r >= 1.5) & (del_r >= 1.3)
    near_bo = in_base & (c >= base_hi * 0.95) & (c <= base_hi)
    bo = (c > hi.shift(5).rolling(55, min_periods=45).max()) & (v.rolling(5).mean() >= 1.5 * v.shift(5).rolling(50, min_periods=40).mean())
    running = (c > sma[50]) & (sma[50] > sma[50].shift(10)) & (c / c.shift(20) - 1 >= 0.10)
    setup = in_base & vol_in & ~running
    return {"base_len": base_len, "base_hi": base_hi, "in_base": in_base.fillna(False), "squeeze": squeeze.fillna(False),
            "vol_in": vol_in.fillna(False), "vol_r": vol_r, "near_bo": near_bo.fillna(False), "bo": bo.fillna(False),
            "running": running.fillna(False), "setup": setup.fillna(False)}


def tier_frame(c, g):
    mnow = g["mcap_cr"].reindex(c.columns)
    last = c.ffill().iloc[-1]
    mc = c.div(last).mul(mnow, axis=1)                       # point-in-time approx market cap
    t = pd.DataFrame(np.select([mc < 400, mc < 2000, mc < 20000], [0, 1, 2], 3), index=c.index, columns=c.columns).where(mc.notna())
    return mc, t                                             # 0 = <400 (bahar), 1 Small, 2 Mid, 3 Large


def industry_ready(T, gmap, tier, min_setup=2, min_pct=0.25, need_large_leader=False):
    """per group per day: ready flag + counts"""
    out = {}
    act20 = (T["running"] | T["bo"]).rolling(20, min_periods=1).max().astype(bool)
    valid = tier >= 1
    for grp, mem in gmap.items():
        if len(mem) < 3:
            continue
        v_ = valid[mem]
        n_valid = v_.sum(axis=1).replace(0, np.nan)
        n_setup = (T["setup"][mem] & v_).sum(axis=1)
        movers = act20[mem] & v_
        n_move = movers.sum(axis=1)
        cond = ((n_setup >= min_setup) | (n_setup / n_valid >= min_pct) & (n_setup >= 1)) & (n_move >= 1)
        if need_large_leader:
            cond &= (movers & (tier[mem] >= 2)).any(axis=1)
        out[grp] = cond.fillna(False)
    return pd.DataFrame(out)
