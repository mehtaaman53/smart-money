"""
'Agla number' noise kam karne ke filters — har filter alag se 5 saal pe test.
Tiers (Aman): Small ₹500–2000 cr · Mid ₹2000–10000 cr · Large > ₹10000 cr · (< ₹500 cr = bahar)
"""
import os, json
import numpy as np, pandas as pd
import peers as PR, pattern as PT, micro_industries as M

ROOT = PR.ROOT


def tier(mc):
    if pd.isna(mc):
        return np.nan
    return 0 if mc < 500 else 1 if mc < 2000 else 2 if mc < 10000 else 3   # 0=micro(bahar) 1=small 2=mid 3=large


def main():
    h = PR.load()
    w = PT.wide(h)
    c, hi, v, t = w["close"], w["high"], w["vol"], w["turnover_lacs"]
    sh = pd.read_csv(os.path.join(ROOT, "data", "shares.csv")).set_index("symbol")["shares"]
    mcap = c.mul(sh.reindex(c.columns), axis=1) / 1e7          # ₹ crore (aaj ke share count se)
    liquid = t.rolling(20, min_periods=15).mean() >= 20
    rets = c.pct_change(fill_method=None).clip(-0.25, 0.25).where(liquid.shift(1, fill_value=False))
    mkt = (1 + rets.mean(axis=1).fillna(0)).cumprod()
    fwd = lambda k: (c.shift(-k) / c - 1).mul(100).sub((mkt.shift(-k) / mkt - 1) * 100, axis=0)
    F60, F120 = fwd(60), fwd(120)
    brk = ((c > hi.shift(1).rolling(60, min_periods=50).max()) &
           (v.rolling(5).mean() >= 2 * v.shift(5).rolling(50, min_periods=40).mean()) & liquid).fillna(False)
    above200 = (c > c.rolling(200, min_periods=150).mean())
    dq = w.get("dq")
    dper = (dq.rolling(10).sum() / v.rolling(10).sum() * 100) if dq is not None else None
    dates = c.index[c.index >= pd.Timestamp("2022-04-01")]
    rows = []
    for g, syms in M.GROUPS.items():
        mem = [s for s in syms.split() if s in c.columns]
        if len(mem) < 2:
            continue
        gi = (1 + rets[mem].mean(axis=1).fillna(0)).cumprod()
        B = brk[mem].loc[dates]
        last = None
        for d in B.index[B.any(axis=1)]:
            p = c.index.get_loc(d)
            if last is not None and p - c.index.get_loc(last) < 40:
                continue
            last = d
            leaders = [s for s in mem if B.at[d, s]]
            n10 = int(brk[mem].iloc[max(0, p - 10):p + 1].any().sum())
            theme = (gi.iloc[p] / gi.iloc[p - 60] - mkt.iloc[p] / mkt.iloc[p - 60]) * 100 if p >= 60 else np.nan
            lt = max([tier(mcap.at[d, s]) for s in leaders if pd.notna(mcap.at[d, s])] or [np.nan])
            l_del = np.nanmean([dper.at[d, s] for s in leaders]) if dper is not None else np.nan
            recent = brk[mem].iloc[max(0, p - 20):p + 1].any()
            for s in mem:
                if s in leaders or recent[s] or not liquid.at[d, s]:
                    continue
                rows.append({"group": g, "date": d, "stock": s, "theme": theme, "n_lead": n10, "lead_tier": lt,
                             "tier": tier(mcap.at[d, s]), "above200": bool(above200.at[d, s]), "lead_deliv": l_del,
                             "f60": F60.at[d, s], "f120": F120.at[d, s]})
    R = pd.DataFrame(rows)
    S = PR.stats
    base = R[R.theme > 0]                       # abhi live wala rule
    V = {
        "LIVE abhi (theme > 0)": base,
        "+ <500cr bahar": base[base.tier >= 1],
        "+ same size (leader ka tier = peer ka tier)": base[(base.tier >= 1) & (base.tier == base.lead_tier)],
        "+ bada pehle, chhota baad (leader tier > peer tier)": base[(base.tier >= 1) & (base.lead_tier > base.tier)],
        "+ theme > 5%": base[(base.tier >= 1) & (base.theme > 5)],
        "+ peer 200 DMA ke upar": base[(base.tier >= 1) & base.above200],
        "+ small peers: 2+ leaders": base[(base.tier >= 1) & ((base.tier >= 2) | (base.n_lead >= 2))],
        "+ leader delivery >= 40%": base[(base.tier >= 1) & (base.lead_deliv >= 40)],
    }
    out = {}
    print(f"{'Variant':55} {'n':>6} {'60d avg':>8} {'med':>6} {'win%':>5} {'>20%':>5} | {'120d avg':>8} {'med':>6} {'win%':>5} {'>20%':>5}")
    for k, df in V.items():
        a, b = S(df.f60), S(df.f120)
        out[k] = {"f60": a, "f120": b}
        print(f"{k:55} {a.get('n',0):>6} {a.get('avg','-'):>8} {a.get('median','-'):>6} {a.get('win%','-'):>5} {a.get('big>20%','-'):>5} | {b.get('avg','-'):>8} {b.get('median','-'):>6} {b.get('win%','-'):>5} {b.get('big>20%','-'):>5}")
    # tier-wise
    for tname, tv in [("small", 1), ("mid", 2), ("large", 3)]:
        a = S(base[base.tier == tv].f60); out[f"peer tier {tname}"] = a
        print(f"  peer tier {tname:6}: {a}")
    bl = F60.loc[dates].where(liquid.loc[dates]).stack().dropna().sample(200000, random_state=1)
    out["random_f60"] = S(bl); print("random 60d:", out["random_f60"])
    R.assign(date=R["date"].dt.strftime("%Y-%m-%d")).to_csv(os.path.join(ROOT, "docs", "peers_v2_rows.csv"), index=False)
    json.dump(out, open(os.path.join(ROOT, "docs", "peers_v2.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
