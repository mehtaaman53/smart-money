"""
Naye pattern ka backtest + Aman ke examples.
python backtest_pattern.py [hist]   (hist = 5 saal ka data/hist/*, warna 1 saal data/bhav.parquet)
"""
import os, sys, glob, json
import numpy as np, pandas as pd
import pipeline as P, pattern as PT

ROOT = P.ROOT
ARGS = sys.argv[1:]


def load():
    if "hist" in ARGS and glob.glob(os.path.join(ROOT, "data", "hist", "*.parquet")):
        h = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ROOT, "data", "hist", "*.parquet")))])
    else:
        h = pd.read_parquet(os.path.join(ROOT, "data", "bhav.parquet"))
    h = h.drop_duplicates(["date", "symbol"], keep="last")
    h = h[h["series"].isin(["EQ", "BE", "BZ"])]
    return P.adjust_prices(h)


def stats(x):
    x = pd.Series(x).dropna()
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "avg": round(float(x.mean()), 1), "median": round(float(x.median()), 1),
            "win%": int(round((x > 0).mean() * 100)), "big_win%(>20%)": int(round((x > 20).mean() * 100))}


def main():
    h = load()
    ind = pd.read_csv(os.path.join(ROOT, "data", "industry.csv"))
    imap = ind.set_index("symbol")["industry"].to_dict()
    w = PT.wide(h)
    c = w["close"]
    liquid = w["turnover_lacs"].rolling(20, min_periods=15).mean() >= 20
    rets = c.pct_change(fill_method=None).clip(-0.25, 0.25).where(liquid.shift(1, fill_value=False))
    mkt = (1 + rets.mean(axis=1).fillna(0)).cumprod()
    fwd = lambda k: (c.shift(-k) / c - 1).mul(100).sub((mkt.shift(-k) / mkt - 1) * 100, axis=0)
    bwd = lambda k: (c / c.shift(k) - 1).mul(100).sub((mkt / mkt.shift(k) - 1) * 100, axis=0)
    F20, F60, F120, B20 = fwd(20), fwd(60), fwd(120), bwd(20)
    dates = c.index
    print(f"Data: {dates[0].date()} → {dates[-1].date()}, {c.shape[1]} stocks")
    out = {"from": str(dates[0].date()), "to": str(dates[-1].date()), "runs": {}}

    for W in (40, 60):
        for R in (35.0, 50.0):
            s = PT.signals(w, {"W": W, "R": R})
            fr = PT.fresh(s["sig"])
            m = fr.stack(); m = m[m]
            idx = m.index
            pick = lambda df: df.stack().reindex(idx)
            res = {"signals": int(len(idx)), "before20": stats(pick(B20)), "after20": stats(pick(F20)),
                   "after60": stats(pick(F60)), "after120": stats(pick(F120))}
            fell = s["fell"].stack().reindex(idx).fillna(False).astype(bool)
            res["after60_gira_hua"] = stats(pick(F60)[fell])
            res["after60_bina_gire"] = stats(pick(F60)[~fell])
            out["runs"][f"W{W}_R{int(R)}"] = res
            print(f"\nW={W} R={R}: {res['signals']} signals")
            for k in ["before20", "after20", "after60", "after120", "after60_gira_hua", "after60_bina_gire"]:
                print(f"  {k:18} {res[k]}")

    # ── chosen default (W60 R35) se examples + industry ──
    s = PT.signals(w, {})
    fr = PT.fresh(s["sig"])
    ex = {}
    for sym in ["KRBL", "PINELABS"]:
        if sym in fr.columns:
            ds = fr.index[fr[sym]]
            ex[sym] = [{"date": str(d.date()), "after20": round(float(F20.at[d, sym]), 1) if pd.notna(F20.at[d, sym]) else None,
                        "after60": round(float(F60.at[d, sym]), 1) if pd.notna(F60.at[d, sym]) else None,
                        "close": round(float(c.at[d, sym]), 1)} for d in ds]
    # industry: active signal share
    act = s["sig"]
    groups = {}
    for sym in act.columns:
        i = imap.get(sym)
        if i:
            groups.setdefault(i, []).append(sym)
    irets = {i: (1 + rets[m].mean(axis=1).fillna(0)).cumprod() for i, m in groups.items() if len(m) >= 3}
    isig = {}
    for i, m in groups.items():
        if len(m) < 3:
            continue
        live = liquid[m].sum(axis=1)
        n = act[m].sum(axis=1)
        isig[i] = (n >= 3) & (n / live.replace(0, np.nan) >= 0.2)
    I = pd.DataFrame(isig).fillna(False)
    Ifr = PT.fresh(I)
    rows = []
    for i in Ifr.columns:
        for d in Ifr.index[Ifr[i]]:
            s_ = irets[i]; p = s_.index.get_loc(d)
            f = lambda k: round(float((s_.iloc[p + k] / s_.iloc[p] - mkt.iloc[p + k] / mkt.iloc[p]) * 100), 1) if p + k < len(s_) else None
            b = round(float((s_.iloc[p] / s_.iloc[p - 20] - mkt.iloc[p] / mkt.iloc[p - 20]) * 100), 1) if p >= 20 else None
            rows.append({"industry": i, "date": str(d.date()), "before20": b, "after20": f(20), "after60": f(60), "after120": f(120)})
    R_ = pd.DataFrame(rows)
    out["industry"] = {k: stats(R_[k]) for k in ["before20", "after20", "after60", "after120"]}
    out["industry"]["signals"] = int(len(R_))
    hc = [i for i in groups if any(k in i for k in ["Precision", "Pharma", "Medical", "Diagnostics", "Drug", "Healthcare"])]
    out["industry_examples"] = R_[R_["industry"].isin(hc)].sort_values(["industry", "date"]).to_dict("records")
    out["stock_examples"] = ex
    out["industry_best"] = R_.dropna(subset=["after60"]).sort_values("after60", ascending=False).head(15).to_dict("records")
    print("\nINDUSTRY:", json.dumps(out["industry"]))
    print("\nEXAMPLES:", json.dumps(ex, indent=0))
    for r in out["industry_examples"]:
        print("  ", r)
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "docs", "backtest_pattern.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
