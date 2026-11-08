"""Aman ka rule: industry ke >=10% shares chal rahe + >=20/30% setup mein → setup shares ka aage ka return."""
import numpy as np, pandas as pd, v2, peers as PR, pattern as PT
S = PR.stats
g = v2.groups(); w = v2.load_prices(set(g.index)); c = w["close"]; T = v2.tags(w)
mc, tier = v2.tier_frame(c, g); valid = tier >= 1
r = c.pct_change(fill_method=None).clip(-.25, .25).where(valid.shift(1, fill_value=False)); mkt = (1 + r.mean(axis=1).fillna(0)).cumprod()
F = lambda k: (c.shift(-k) / c - 1).mul(100).sub((mkt.shift(-k) / mkt - 1) * 100, axis=0)
F60, F120 = F(60), F(120)
gm = {}
for s in c.columns: gm.setdefault(g.at[s, "group"], []).append(s)
dates = c.index[c.index >= pd.Timestamp("2022-06-01")]
act20 = (T["running"] | T["bo"]).rolling(20, min_periods=1).max().astype(bool)
base_only = T["in_base"] & ~T["running"]           # setup variant B: sirf base (volume shart bina)
rnd = lambda Fx: Fx.loc[dates].where(valid.loc[dates]).stack().dropna().sample(200000, random_state=1)
print("random:", S(rnd(F60)), "|120:", S(rnd(F120)))
for setup_name, SET in [("Base+Volume", T["setup"]), ("sirf Base", base_only)]:
    for mv, sp in [(0.10, 0.20), (0.10, 0.30), (0.15, 0.25), (0.20, 0.30)]:
        rows = []
        for grp, mem in gm.items():
            if len(mem) < 5: continue                  # chhoti industries (<5 shares) chhodo
            v_ = valid[mem]; n = v_.sum(axis=1).replace(0, np.nan)
            pm = (act20[mem] & v_).sum(axis=1) / n; ps = (SET[mem] & v_).sum(axis=1) / n
            gi = (1 + r[mem].mean(axis=1).fillna(0)).cumprod(); th = (gi / gi.shift(60) - mkt / mkt.shift(60)) * 100
            ready = ((pm >= mv) & (ps >= sp)).loc[dates]
            fr = ready & ~ready.shift(1).rolling(40, min_periods=1).max().fillna(0).astype(bool)
            for d in fr.index[fr.fillna(False)]:
                for s in mem:
                    if SET.at[d, s] and valid.at[d, s]:
                        rows.append((grp, d, s, th.at[d], F60.at[d, s], F120.at[d, s]))
        R = pd.DataFrame(rows, columns=["g", "d", "s", "th", "f60", "f120"])
        if not len(R): print(setup_name, mv, sp, "no signals"); continue
        sig = R.groupby(["g", "d"]).ngroups
        print(f"{setup_name:12} chal>={int(mv*100)}% setup>={int(sp*100)}%  signals {sig:4}  ", "60:", S(R.f60), "|120:", S(R.f120))
        t = R[R.th > 5]
        print(f"{'':12} ...+ theme>5%{'':12}signals {t.groupby(['g','d']).ngroups:4}  ", "60:", S(t.f60), "|120:", S(t.f120))
