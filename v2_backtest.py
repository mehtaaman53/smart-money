"""v2 ka 5 saal backtest + Aman ke examples (Jewellery, Sugar, Precision)."""
import os, json
import numpy as np, pandas as pd
import v2, pattern as PT, peers as PR

ROOT = v2.ROOT
S = PR.stats


def main():
    g = v2.groups()
    w = v2.load_prices(set(g.index))
    c = w["close"]
    T = v2.tags(w)
    mc, tier = v2.tier_frame(c, g)
    valid = tier >= 1
    rets = c.pct_change(fill_method=None).clip(-0.25, 0.25).where(valid.shift(1, fill_value=False))
    mkt = (1 + rets.mean(axis=1).fillna(0)).cumprod()
    fwd = lambda k: (c.shift(-k) / c - 1).mul(100).sub((mkt.shift(-k) / mkt - 1) * 100, axis=0)
    F60, F120 = fwd(60), fwd(120)
    gmap = {}
    for s in c.columns:
        gmap.setdefault(g.at[s, "group"], []).append(s)
    dates = c.index[c.index >= pd.Timestamp("2022-06-01")]
    print(f"{c.shape[1]} stocks, {len(gmap)} groups, {dates[0].date()}→{dates[-1].date()}")

    def run(name, **kw):
        R = v2.industry_ready(T, gmap, tier, **kw).loc[dates]
        Fr = PT.fresh(R, 40)
        rows = []
        for grp in Fr.columns:
            for d in Fr.index[Fr[grp]]:
                mem = gmap[grp]
                for s in mem:
                    if T["setup"].at[d, s] and valid.at[d, s]:
                        rows.append({"group": grp, "date": d.strftime("%Y-%m-%d"), "stock": s, "tier": int(tier.at[d, s]),
                                     "base_len": int(T["base_len"].at[d, s]), "sqz": bool(T["squeeze"].at[d, s]),
                                     "near_bo": bool(T["near_bo"].at[d, s]), "f60": F60.at[d, s], "f120": F120.at[d, s]})
        return pd.DataFrame(rows), int(Fr.values.sum())

    out = {}
    base_rows, n_sig = run("base")
    R = base_rows
    out["industry_taiyaar_setup_stocks"] = {"signals": n_sig, "f60": S(R.f60), "f120": S(R.f120)}
    big, n_big = run("big", need_large_leader=True)
    out["+ leader Mid/Large"] = {"signals": n_big, "f60": S(big.f60), "f120": S(big.f120)}
    for k, m in {"+ base >= 90 din": R.base_len >= 90, "+ base >= 180 din": R.base_len >= 180, "+ MA squeeze": R.sqz,
                 "+ breakout ke paas": R.near_bo, "peer Small": R.tier == 1, "peer Mid": R.tier == 2, "peer Large": R.tier == 3}.items():
        out[k] = {"f60": S(R[m].f60), "f120": S(R[m].f120)}
    # comparison: setup stocks kahin bhi (industry ki shart bina) + random
    allsetup = (T["setup"] & valid).loc[dates]
    pick = lambda F: F.loc[dates].where(allsetup).stack().dropna()
    out["setup kahin bhi (industry shart bina)"] = {"f60": S(pick(F60)), "f120": S(pick(F120))}
    rnd = lambda F: F.loc[dates].where(valid.loc[dates]).stack().dropna().sample(200000, random_state=1)
    out["random (400cr+)"] = {"f60": S(rnd(F60)), "f120": S(rnd(F120))}
    for k, v in out.items():
        print(f"{k:42}", v)
    # examples
    ex = {}
    for grp in ["Gems, Jewellery And Watches", "Sugar", "Precision Engineering"]:
        x = R[R.group == grp].groupby("date").agg(stocks=("stock", lambda z: ",".join(z)), f60=("f60", "mean"), f120=("f120", "mean")).round(1)
        ex[grp] = x.reset_index().to_dict("records")
        print("\n##", grp); print(x.to_string() if len(x) else "  (koi signal nahi)")
    out["examples"] = ex
    R["y"] = R.date.str[:4]
    out["by_year_f120"] = R.groupby("y").f120.agg(["count", "mean", "median"]).round(1).reset_index().to_dict("records")
    print(R.groupby("y").f120.agg(["count", "mean", "median"]).round(1))
    R.to_csv(os.path.join(ROOT, "docs", "v2_backtest_rows.csv"), index=False)
    json.dump(out, open(os.path.join(ROOT, "docs", "v2_backtest.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
