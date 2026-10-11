"""
🎯 AGLA NUMBER — "group ka ek share bhaaga + theme strong → baaki peers".
5 saal ke test mein sabse achha version (peeche wale peers: 3M avg +2.9%, 6M +4.5% market se extra; random -0.4%).
Rules:
  - Trigger: group ka koi member pichhle 10 din mein 60-din high toda + 5-din volume >= 2x (50-din avg)
  - Theme strong: group index ka pichhla 3 mahina (60 din) market se behtar
  - Agla number: baaki members jo pichhle 20 din mein khud nahi bhaage, liquid (>= ₹20 lakh/din)
"""
import numpy as np, pandas as pd
import pattern as PT, micro_industries as M

TRIGGER_DAYS = 10


def compute(h):
    """h = adjusted bhav (long). Returns list of opportunities (latest date)."""
    w = PT.wide(h)
    c, hi, v, t = w["close"], w["high"], w["vol"], w["turnover_lacs"]
    liquid = t.rolling(20, min_periods=15).mean() >= 20
    rets = c.pct_change(fill_method=None).clip(-0.25, 0.25).where(liquid.shift(1, fill_value=False))
    mkt = (1 + rets.mean(axis=1).fillna(0)).cumprod()
    brk = ((c > hi.shift(1).rolling(60, min_periods=50).max()) &
           (v.rolling(5).mean() >= 2 * v.shift(5).rolling(50, min_periods=40).mean()) & liquid).fillna(False)
    rng = (hi.rolling(60, min_periods=48).max() / w["low"].rolling(60, min_periods=48).min() - 1) * 100
    hi60 = hi.rolling(60, min_periods=48).max()
    dq = w.get("dq")
    last = c.index[-1]
    r1 = lambda x: None if x is None or pd.isna(x) else round(float(x), 1)
    out = []
    for g, syms in M.GROUPS.items():
        mem = [s for s in syms.split() if s in c.columns and pd.notna(c[s].iloc[-1])]
        if len(mem) < 2:
            continue
        gi = (1 + rets[mem].mean(axis=1).fillna(0)).cumprod()
        theme = (gi.iloc[-1] / gi.iloc[-61] - mkt.iloc[-1] / mkt.iloc[-61]) * 100 if len(gi) > 61 else np.nan
        recentB = brk[mem].iloc[-TRIGGER_DAYS:]
        leaders = [s for s in mem if recentB[s].any()]
        if not leaders:
            continue
        b20 = brk[mem].iloc[-20:].any()
        lead_info = []
        for s in leaders:
            d = recentB.index[recentB[s]][0]
            lead_info.append({"s": s, "since": d.strftime("%d %b"), "move": r1((c[s].iloc[-1] / c[s].loc[:d].iloc[-2] - 1) * 100)})
        lag = []
        for s in mem:
            if s in leaders or b20[s] or not liquid[s].iloc[-1]:
                continue
            dp = None
            if dq is not None and s in dq.columns:
                vv = v[s].iloc[-10:].sum()
                dp = r1(100 * dq[s].iloc[-10:].sum() / vv) if vv else None
            lag.append({"s": s, "c": r1(c[s].iloc[-1]), "r20": r1((c[s].iloc[-1] / c[s].iloc[-21] - 1) * 100) if len(c) > 21 else None,
                        "from_hi": r1((c[s].iloc[-1] / hi60[s].iloc[-1] - 1) * 100), "rng": r1(rng[s].iloc[-1]),
                        "vr": r1(v[s].iloc[-10:].mean() / v[s].iloc[-60:].mean()) if v[s].iloc[-60:].mean() else None,
                        "dp": dp})
        if not lag:
            continue
        lag.sort(key=lambda x: (x["from_hi"] if x["from_hi"] is not None else -99), reverse=True)
        out.append({"group": g, "theme": r1(theme), "strong": bool(pd.notna(theme) and theme > 0),
                    "leaders": lead_info, "laggards": lag,
                    "first_trigger": min(pd.Timestamp(recentB.index[recentB[s]][0]) for s in leaders).strftime("%Y-%m-%d")})
    out.sort(key=lambda x: (not x["strong"], -(x["theme"] or -99)))
    return {"asof": last.strftime("%d %b %Y"), "items": out}
