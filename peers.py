"""
PEER GROUPS + "EK CHALA TO BAAKI" TEST
1) Har Yahoo industry ko chhote peer groups mein todo — jo stocks asal mein saath chalte hain
   (market hata ke weekly returns ka correlation, hierarchical clustering).
   Backtest ke liye groups SIRF pehle 2 saal (Oct-21 → Sep-23) ke data se bante hain, test baad ke 3 saal pe — taaki future leak na ho.
2) Trigger: group ka koi member base se volume ke saath bhaage (60 din high toda + 5 din volume >= 2x).
   Group ka "pehla" trigger (40 din se koi nahi bhaaga tha).
3) Baaki peers (jo abhi nahi bhaage) ka agle 20/60/120 din ka market se extra return dekho.
python peers.py
"""
import os, glob, json
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import pipeline as P, pattern as PT

ROOT = P.ROOT
TRAIN_END = pd.Timestamp("2023-09-30")
CORR_CUT = 0.30          # group ke andar average residual correlation itna to ho
MAX_GROUP = 20


def load():
    h = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ROOT, "data", "hist", "*.parquet")))])
    h = h.drop_duplicates(["date", "symbol"], keep="last")
    return P.adjust_prices(h[h["series"].isin(["EQ", "BE", "BZ"])])


def build_groups(c, imap, end=None, min_weeks=60):
    """c = wide close. Returns {symbol: group_name}"""
    cc = c if end is None else c[c.index <= end]
    wk = cc.resample("W-FRI").last().pct_change(fill_method=None).clip(-0.4, 0.4)
    resid = wk.sub(wk.mean(axis=1), axis=0)                     # market hatao
    by_ind = {}
    for s, i in imap.items():
        if s in resid.columns and resid[s].notna().sum() >= min_weeks:
            by_ind.setdefault(i, []).append(s)
    groups = {}
    for ind, mem in by_ind.items():
        if len(mem) < 2:
            continue
        r = resid[mem].corr(min_periods=min_weeks // 2).fillna(0).to_numpy(copy=True)
        np.fill_diagonal(r, 1)
        if len(mem) == 2:
            lab = [1, 1] if r[0, 1] >= CORR_CUT else [1, 2]
        else:
            Z = linkage(squareform(1 - r, checks=False), "average")
            lab = fcluster(Z, t=1 - CORR_CUT, criterion="distance")
        cl = {}
        for s, l in zip(mem, lab):
            cl.setdefault(l, []).append(s)
        k = 0
        for l, ms in sorted(cl.items(), key=lambda x: -len(x[1])):
            if len(ms) < 2:
                continue
            # bahut bada group → top correlated MAX_GROUP hi
            k += 1
            name = f"{ind} #{k}"
            for s in ms[:MAX_GROUP] if len(ms) <= MAX_GROUP else ms:
                groups[s] = name
    return groups


def stats(x):
    x = pd.Series(x).dropna()
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "avg": round(float(x.mean()), 1), "median": round(float(x.median()), 1),
            "win%": int(round((x > 0).mean() * 100)), "big>20%": int(round((x > 20).mean() * 100))}


def main():
    h = load()
    imap = pd.read_csv(os.path.join(ROOT, "data", "industry.csv")).set_index("symbol")["industry"].to_dict()
    w = PT.wide(h)
    c, hi, v, t = w["close"], w["high"], w["vol"], w["turnover_lacs"]
    liquid = t.rolling(20, min_periods=15).mean() >= 20
    rets = c.pct_change(fill_method=None).clip(-0.25, 0.25).where(liquid.shift(1, fill_value=False))
    mkt = (1 + rets.mean(axis=1).fillna(0)).cumprod()
    fwd = lambda k: (c.shift(-k) / c - 1).mul(100).sub((mkt.shift(-k) / mkt - 1) * 100, axis=0)
    F20, F60, F120 = fwd(20), fwd(60), fwd(120)

    # trigger: 60 din high toda + 5 din avg volume >= 2x 50 din avg
    brk = (c > hi.shift(1).rolling(60, min_periods=50).max()) & \
          (v.rolling(5).mean() >= 2 * v.shift(5).rolling(50, min_periods=40).mean()) & liquid
    brk = brk.fillna(False)
    base = PT.signals(w, {"W": 60, "R": 35.0})
    in_base = ((base["range"] <= 35) & liquid).fillna(False)

    groups = build_groups(c, imap, end=TRAIN_END)
    gmem = {}
    for s, g in groups.items():
        gmem.setdefault(g, []).append(s)
    sizes = pd.Series({g: len(m) for g, m in gmem.items()})
    print(f"Peer groups (train data se): {len(gmem)} groups, size median {sizes.median():.0f}, max {sizes.max()}")

    test_dates = c.index[c.index > TRAIN_END]
    rows = []
    for g, mem in gmem.items():
        B = brk[mem].loc[test_dates]
        anyb = B.any(axis=1)
        last_trig = None
        for d in anyb.index[anyb]:
            if last_trig is not None and (c.index.get_loc(d) - c.index.get_loc(last_trig)) < 40:
                continue
            last_trig = d
            leaders = [s for s in mem if B.at[d, s]]
            # laggards: pichhle 20 din mein khud nahi bhaage, liquid hain
            p = c.index.get_loc(d)
            recent = brk[mem].iloc[max(0, p - 20):p + 1].any()
            for s in mem:
                if s in leaders or recent[s] or not liquid.at[d, s]:
                    continue
                rows.append({"group": g, "date": d, "leader": ",".join(leaders), "stock": s,
                             "in_base": bool(in_base.at[d, s]),
                             "f20": F20.at[d, s], "f60": F60.at[d, s], "f120": F120.at[d, s],
                             "lead_f60": float(np.nanmean([F60.at[d, x] for x in leaders]))})
    R = pd.DataFrame(rows)
    # baseline: test period ke saare liquid stock-days
    basel = F60.loc[test_dates].where(liquid.loc[test_dates]).stack().dropna()
    out = {"groups": int(len(gmem)), "triggers": int(R.groupby(["group", "date"]).ngroups),
           "laggards_all": {k: stats(R[k]) for k in ["f20", "f60", "f120"]},
           "laggards_in_base": {k: stats(R[R["in_base"]][k]) for k in ["f20", "f60", "f120"]},
           "leaders_f60": stats(R.drop_duplicates(["group", "date"])["lead_f60"]),
           "baseline_f60": stats(basel.sample(min(200000, len(basel)), random_state=1))}
    # group size ke hisaab se (chhote groups behtar?)
    R["gsize"] = R["group"].map(sizes)
    out["by_size_f60"] = {f"{a}-{b}": stats(R[(R.gsize >= a) & (R.gsize <= b)]["f60"]) for a, b in [(2, 4), (5, 10), (11, 99)]}
    print(json.dumps(out, indent=1))

    # examples
    def show(sym):
        g = groups.get(sym)
        if not g:
            return {"stock": sym, "group": None}
        return {"stock": sym, "group": g, "members": gmem[g]}
    out["examples"] = [show(s) for s in ["FINEORG", "KRBL", "LTFOODS", "AZAD", "MTARTECH"]]
    for e in out["examples"]:
        print(e)
    R.assign(date=R["date"].dt.strftime("%Y-%m-%d")).to_csv(os.path.join(ROOT, "docs", "peers_backtest_rows.csv"), index=False)
    json.dump(out, open(os.path.join(ROOT, "docs", "peers_backtest.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
