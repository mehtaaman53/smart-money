"""
BACKTEST — pichhle data pe har hafte scanner chalao, phir dekho signal ke baad kya hua.
Har signal ke liye: signal se PEHLE 20 din ka return (kitna move nikal chuka tha)
aur BAAD mein 20 / 60 din ka return, sab "saari industries ke average" se compare karke (excess).
Output: docs/backtest.json
"""
import os, json
import numpy as np, pandas as pd
import pipeline as P
import analyze as A

ROOT = os.path.dirname(os.path.abspath(__file__))
STEP = 5          # har 5 trading din (≈ har hafte)
WARMUP = 120      # pehle 120 din signals ke liye history


def main():
    h = pd.read_parquet(os.path.join(ROOT, "data", "bhav.parquet"))
    u = pd.read_csv(os.path.join(ROOT, "data", "universe.csv"))
    ind = pd.read_csv(os.path.join(ROOT, "data", "industry.csv"))
    ok = set(u[u["ok"]]["symbol"])
    h = P.adjust_prices(h[h["symbol"].isin(ok)])
    imap = ind.set_index("symbol")["industry"].to_dict()
    days = sorted(h["date"].unique())

    # forward returns ke liye: industry index (full data) + market (sab stocks equal weight)
    piv = h.pivot_table(index="date", columns="symbol", values="close").sort_index()
    rets = piv.pct_change(fill_method=None).clip(-0.25, 0.25)
    mkt = (1 + rets.mean(axis=1).fillna(0)).cumprod()
    members = {}
    for s, i in imap.items():
        if s in rets.columns:
            members.setdefault(i, []).append(s)
    idx = {i: (1 + rets[m].mean(axis=1).fillna(0)).cumprod() for i, m in members.items() if len(m) >= A.MIN_STOCKS}

    def ex(i, d, k):
        """d se k din aage (k<0 = pichhe) industry return minus market return, %"""
        s = idx[i]; p = s.index.get_loc(d); q = p + k
        if q < 0 or q >= len(s):
            return None
        a, b = (p, q) if k > 0 else (q, p)
        return round(float((s.iloc[b] / s.iloc[a] - mkt.iloc[b] / mkt.iloc[a]) * 100), 2)

    rows = []
    cuts = list(range(WARMUP, len(days) - 1, STEP))
    for n, ci in enumerate(cuts):
        d = days[ci]
        hh = h[h["date"] <= d]
        try:
            inds = A.compute(hh, imap, {})
        except Exception as e:
            print("skip", d, e); continue
        for m in inds:
            if m["stage"] in ("breakout", "accum", "base") and m["name"] in idx:
                rows.append({"date": str(pd.Timestamp(d).date()), "industry": m["name"], "stage": m["stage"],
                             "before20": ex(m["name"], d, -20), "after20": ex(m["name"], d, 20), "after60": ex(m["name"], d, 60)})
        print(f"{n+1}/{len(cuts)} {pd.Timestamp(d).date()}: {sum(1 for m in inds if m['stage'] in ('breakout','accum','base'))} signals")

    df = pd.DataFrame(rows)
    # sirf "naya" signal gino: industry pichhle cut pe us stage mein nahi thi
    df = df.sort_values(["industry", "date"])
    df["prev_date"] = df.groupby(["industry", "stage"])["date"].shift(1)
    cut_dates = [str(pd.Timestamp(days[c]).date()) for c in cuts]
    pos = {d: i for i, d in enumerate(cut_dates)}
    df["fresh"] = df.apply(lambda r: pd.isna(r["prev_date"]) or pos[r["date"]] - pos[r["prev_date"]] > 1, axis=1)
    f = df[df["fresh"]]

    summ = {}
    for st, g in f.groupby("stage"):
        def stat(c):
            x = g[c].dropna()
            return {"n": int(len(x)), "avg": round(float(x.mean()), 2) if len(x) else None,
                    "median": round(float(x.median()), 2) if len(x) else None,
                    "win": round(float((x > 0).mean() * 100), 0) if len(x) else None}
        summ[st] = {"signals": int(len(g)), "before20": stat("before20"), "after20": stat("after20"), "after60": stat("after60")}
    out = {"from": cut_dates[0], "to": cut_dates[-1], "cuts": len(cuts), "summary": summ,
           "best": f.dropna(subset=["after60"]).sort_values("after60", ascending=False).head(15).drop(columns=["prev_date", "fresh"]).to_dict("records"),
           "worst": f.dropna(subset=["after60"]).sort_values("after60").head(10).drop(columns=["prev_date", "fresh"]).to_dict("records"),
           "recent": f.sort_values("date", ascending=False).head(25).drop(columns=["prev_date", "fresh"]).to_dict("records")}
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "docs", "backtest.json"), "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
