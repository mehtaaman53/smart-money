"""Haath se bani chhoti industries pe 'ek chala to baaki' test (5 saal)."""
import os, json
import numpy as np, pandas as pd
import peers as PR, pattern as PT, micro_industries as M

ROOT = PR.ROOT


def main():
    h = PR.load()
    w = PT.wide(h)
    c, hi, v, t = w["close"], w["high"], w["vol"], w["turnover_lacs"]
    liquid = t.rolling(20, min_periods=15).mean() >= 20
    rets = c.pct_change(fill_method=None).clip(-0.25, 0.25).where(liquid.shift(1, fill_value=False))
    mkt = (1 + rets.mean(axis=1).fillna(0)).cumprod()
    fwd = lambda k: (c.shift(-k) / c - 1).mul(100).sub((mkt.shift(-k) / mkt - 1) * 100, axis=0)
    F20, F60, F120 = fwd(20), fwd(60), fwd(120)
    brk = ((c > hi.shift(1).rolling(60, min_periods=50).max()) &
           (v.rolling(5).mean() >= 2 * v.shift(5).rolling(50, min_periods=40).mean()) & liquid).fillna(False)
    rng = (hi.rolling(60, min_periods=48).max() / w["low"].rolling(60, min_periods=48).min() - 1) * 100
    in_base = ((rng <= 35) & liquid).fillna(False)
    dates = c.index[c.index >= pd.Timestamp("2022-04-01")]
    rows = []
    for g, syms in M.GROUPS.items():
        mem = [s for s in syms.split() if s in c.columns]
        if len(mem) < 2:
            continue
        B = brk[mem].loc[dates]
        last = None
        for d in B.index[B.any(axis=1)]:
            p = c.index.get_loc(d)
            if last is not None and p - c.index.get_loc(last) < 40:
                continue
            last = d
            leaders = [s for s in mem if B.at[d, s]]
            recent = brk[mem].iloc[max(0, p - 20):p + 1].any()
            # confirmation: 20 din mein 2+ members bhaage?
            conf = int(brk[mem].iloc[max(0, p - 20):p + 1].any().sum())
            for s in mem:
                if s in leaders or recent[s] or not liquid.at[d, s]:
                    continue
                rows.append({"group": g, "date": d.strftime("%Y-%m-%d"), "leader": ",".join(leaders), "stock": s,
                             "in_base": bool(in_base.at[d, s]), "gsize": len(mem), "conf": conf,
                             "f20": F20.at[d, s], "f60": F60.at[d, s], "f120": F120.at[d, s]})
    R = pd.DataFrame(rows)
    S = PR.stats
    basel = F60.loc[dates].where(liquid.loc[dates]).stack().dropna()
    out = {"triggers": int(R.groupby(["group", "date"]).ngroups), "laggards": int(len(R)),
           "all": {k: S(R[k]) for k in ["f20", "f60", "f120"]},
           "in_base": {k: S(R[R.in_base][k]) for k in ["f20", "f60", "f120"]},
           "small_groups_2_6": {k: S(R[R.gsize <= 6][k]) for k in ["f20", "f60", "f120"]},
           "small_and_base": {k: S(R[(R.gsize <= 6) & R.in_base][k]) for k in ["f20", "f60", "f120"]},
           "baseline_f60": S(basel.sample(min(200000, len(basel)), random_state=1)),
           "baseline_f120": S(F120.loc[dates].where(liquid.loc[dates]).stack().dropna().sample(200000, random_state=1))}
    by_g = R.groupby("group")["f60"].agg(["count", "mean", "median"]).round(1).sort_values("mean", ascending=False)
    out["best_groups"] = by_g.head(12).reset_index().to_dict("records")
    out["worst_groups"] = by_g.tail(8).reset_index().to_dict("records")
    ex = {}
    for g in ["Rice (Basmati)", "Oleochemicals / Surfactants", "Precision Engineering", "Hospitals", "Shipbuilding", "Cables & Wires"]:
        x = R[R.group == g].groupby(["date", "leader"]).agg(laggards=("stock", lambda z: ",".join(z)), f60=("f60", "mean")).round(1).reset_index()
        ex[g] = x.to_dict("records")
    out["examples"] = ex
    print(json.dumps({k: v for k, v in out.items() if k not in ("examples",)}, indent=1, default=str))
    for g, xs in ex.items():
        print("\n##", g)
        for x in xs:
            print("  ", x)
    R.to_csv(os.path.join(ROOT, "docs", "peers_curated_rows.csv"), index=False)
    json.dump(out, open(os.path.join(ROOT, "docs", "peers_curated.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
