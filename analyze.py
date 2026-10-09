"""
SMART MONEY — Part 2: industry stage nikalo (Accumulation / Base / Breakout / Running / Distribution)
Input: data/bhav.parquet, data/universe.csv, data/industry.csv, data/deals.csv
Output: docs/data.json + docs/index.html, data/stages.csv (history), Telegram alert (sirf naye 🧲/🚀)
"""
import os, json, datetime as dt
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pipeline as P

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "docs")
MIN_STOCKS = 3
BASE_DAYS = 30            # ~6 hafte ka base
BASE_RANGE_IND = 12.0     # industry index ka range % (base)
BASE_RANGE_STK = 18.0     # stock ka range % (base)
STAGES = {"breakout": "🚀 Breakout", "accum": "🧲 Accumulation", "base": "🏗️ Base", "running": "🏃 Running",
          "distrib": "⚠️ Distribution", "quiet": "😴 Quiet"}
ORDER = ["breakout", "accum", "base", "running", "distrib", "quiet"]


def r1(x):
    return None if x is None or pd.isna(x) else round(float(x), 1)


def stock_signals(g):
    """g = ek stock ka daily data (adjusted), date-sorted."""
    g = g.dropna(subset=["close"])
    if len(g) < 60:
        return None
    c, h, l, v, t = g["close"].values, g["high"].values, g["low"].values, g["volume_adj"].values, g["turnover_lacs"].values
    dq = g["deliv_qty"].values / np.where(g["volume"].values > 0, g["volume"].values / g["volume_adj"].values, 1)
    dp = g["deliv_per"].values
    n = len(c)
    v50 = np.nanmean(v[-50:])
    prior_hi = np.nanmax(h[-BASE_DAYS - 1:-1])
    lo30 = np.nanmin(l[-BASE_DAYS:])
    rng = (np.nanmax(h[-BASE_DAYS:]) / lo30 - 1) * 100
    # breakout pichhle 5 din mein: close > pichhle 30 din ka high, volume 1.8x
    bo = False
    for k in range(1, 6):
        if n - k - BASE_DAYS < 0:
            break
        ph = np.nanmax(h[n - k - BASE_DAYS:n - k])
        if c[n - k] > ph and v[n - k] >= 1.8 * v50:
            bo = True
            break
    hi52 = np.nanmax(c[-250:])
    ret = lambda d: (c[-1] / c[-1 - d] - 1) * 100 if n > d else np.nan
    up = np.diff(c[-21:]) > 0
    tv = t[-20:]
    up_t, dn_t = np.nansum(tv[up]), np.nansum(tv[~up])
    return {
        "close": c[-1], "ret5": ret(5), "ret20": ret(20), "ret60": ret(60),
        "above50": c[-1] > np.nanmean(c[-50:]), "range30": rng,
        "in_base": rng <= BASE_RANGE_STK, "breakout": bo, "near_hi": c[-1] >= 0.97 * prior_hi,
        "new52": c[-1] >= hi52 * 0.999 and n >= 120,
        "vol_ratio": np.nanmean(v[-5:]) / v50 if v50 else np.nan,
        "dry": np.nanmean(v[-10:]) / v50 if v50 else np.nan,
        "deliv_ratio": np.nanmean(dq[-10:]) / np.nanmean(dq[-50:]) if np.nanmean(dq[-50:]) else np.nan,
        "deliv_per": np.nanmean(dp[-10:]), "deliv_per50": np.nanmean(dp[-50:]),
        "ud": up_t / dn_t if dn_t else np.nan, "turnover": np.nanmean(t[-20:]),
    }


def industry_stage(m):
    rng, surge = m["range30"], m["turn_surge"]
    not_down = (m["ret60"] if m["ret60"] is not None and not pd.isna(m["ret60"]) else 0) >= -5
    if (m["ind_breakout"] and surge >= 1.3) or (m["n_breakout"] >= 3 and m["pct_breakout"] >= 25 and m["ret5"] > 0):
        return "breakout"
    if rng <= BASE_RANGE_IND and m["ud"] >= 1.3 and m["deliv_ratio"] >= 1.1 and not_down:
        return "accum"
    if m["ret20"] >= 5 and m["pct_above50"] >= 50 and m["ud"] < 0.8 and m["near_hi"]:
        return "distrib"
    if rng <= BASE_RANGE_IND and m["dry"] <= 0.9 and not_down and m["near_top"]:
        return "base"
    if m["ret20"] >= 4 and m["pct_above50"] >= 55:
        return "running"
    return "quiet"


def main():
    h = pd.read_parquet(os.path.join(ROOT, "data", "bhav.parquet"))
    u = pd.read_csv(os.path.join(ROOT, "data", "universe.csv"))
    ind = pd.read_csv(os.path.join(ROOT, "data", "industry.csv"))
    deals_p = os.path.join(ROOT, "data", "deals.csv")
    deals = pd.read_csv(deals_p) if os.path.exists(deals_p) else pd.DataFrame()
    ok = set(u[u["ok"]]["symbol"])
    h = P.adjust_prices(h[h["symbol"].isin(ok)])
    last = h["date"].max()
    imap = ind.set_index("symbol")["industry"].to_dict()

    # bulk/block buys last ~20 din
    buys = {}
    if len(deals):
        dd = deals.copy()
        dd["d"] = pd.to_datetime(dd["Date"], format="%d-%b-%Y", errors="coerce")
        dd = dd[(dd["d"] >= last - pd.Timedelta(days=30)) & (dd["Buy/Sell"].astype(str).str.upper() == "BUY")]
        buys = dd.groupby("Symbol").size().to_dict()

    sig = {}
    for s, g in h.groupby("symbol"):
        x = stock_signals(g.sort_values("date"))
        if x:
            x["industry"] = imap.get(s)
            x["bulk_buys"] = int(buys.get(s, 0))
            sig[s] = x
    S = pd.DataFrame(sig).T
    S = S[S["industry"].notna()]
    for col in S.columns:
        if col != "industry":
            S[col] = pd.to_numeric(S[col], errors="coerce")

    # industry index (equal weight daily returns)
    piv = h.pivot_table(index="date", columns="symbol", values="close").sort_index()
    turn = h.pivot_table(index="date", columns="symbol", values="turnover_lacs").sort_index()
    rets = piv.pct_change(fill_method=None).clip(-0.25, 0.25)
    inds = []
    for name, grp in S.groupby("industry"):
        mem = list(grp.index)
        if len(mem) < MIN_STOCKS:
            continue
        idx = 100 * (1 + rets[mem].mean(axis=1).fillna(0)).cumprod()
        tt = turn[mem].sum(axis=1)
        prior_hi = idx.iloc[-BASE_DAYS - 6:-1].max()
        rng = (idx.iloc[-BASE_DAYS:].max() / idx.iloc[-BASE_DAYS:].min() - 1) * 100
        m = {
            "name": name, "n": len(mem), "range30": rng,
            "ind_breakout": bool(idx.iloc[-1] > idx.iloc[-BASE_DAYS - 6:-5].max() and idx.iloc[-1] >= idx.iloc[-6]),
            "near_top": bool(idx.iloc[-1] >= 0.95 * idx.iloc[-BASE_DAYS:].max()),
            "turn_surge": tt.iloc[-5:].mean() / tt.iloc[-50:].mean() if tt.iloc[-50:].mean() else 1,
            "dry": tt.iloc[-10:].mean() / tt.iloc[-50:].mean() if tt.iloc[-50:].mean() else 1,
            "near_hi": bool(idx.iloc[-1] >= 0.95 * idx.iloc[-60:].max()),
            "ret5": (idx.iloc[-1] / idx.iloc[-6] - 1) * 100, "ret20": (idx.iloc[-1] / idx.iloc[-21] - 1) * 100,
            "ret60": (idx.iloc[-1] / idx.iloc[-61] - 1) * 100 if len(idx) > 61 else np.nan,
            "n_breakout": int(grp["breakout"].sum()), "pct_breakout": 100 * grp["breakout"].mean(),
            "n_base": int(grp["in_base"].sum()), "n_new52": int(grp["new52"].sum()),
            "pct_above50": 100 * grp["above50"].astype(float).mean(),
            "deliv_ratio": grp["deliv_ratio"].median(), "deliv_per": grp["deliv_per"].median(),
            "ud": grp["ud"].median(), "bulk_buys": int(grp["bulk_buys"].sum()),
            "turnover_cr": tt.iloc[-20:].mean() / 100,
            "spark": [round(float(x), 2) for x in idx.iloc[-120:]],
        }
        m["stage"] = industry_stage(m)
        score = (m["n_breakout"] * 3 + m["n_new52"] * 2 + (m["deliv_ratio"] - 1) * 20 + (m["ud"] - 1) * 10
                 + (m["turn_surge"] - 1) * 10 + m["bulk_buys"])
        m["score"] = score
        # leading stocks: breakout > base+delivery > RS
        g2 = grp.copy()
        g2["lead"] = g2["breakout"] * 5 + g2["new52"] * 3 + g2["in_base"] * 1 + (g2["deliv_ratio"] - 1) * 5 + g2["ret20"].fillna(0) / 5
        g2 = g2.sort_values("lead", ascending=False)
        m["stocks"] = [{"s": s, "c": r1(r["close"]), "r20": r1(r["ret20"]), "dp": r1(r["deliv_per"]), "dr": r1(r["deliv_ratio"]),
                        "vr": r1(r["vol_ratio"]), "bo": bool(r["breakout"]), "base": bool(r["in_base"]),
                        "h52": bool(r["new52"]), "bb": int(r["bulk_buys"])} for s, r in g2.iterrows()]
        for k in ["range30", "turn_surge", "dry", "ret5", "ret20", "ret60", "pct_breakout", "pct_above50", "deliv_ratio",
                  "deliv_per", "ud", "turnover_cr", "score"]:
            m[k] = r1(m[k]) if k not in ("turn_surge", "dry", "deliv_ratio", "ud") else (None if pd.isna(m[k]) else round(float(m[k]), 2))
        inds.append(m)

    inds.sort(key=lambda m: (ORDER.index(m["stage"]), -(m["score"] or 0)))

    # stage history → naye 🧲/🚀
    hist_p = os.path.join(ROOT, "data", "stages.csv")
    prev = pd.read_csv(hist_p) if os.path.exists(hist_p) else pd.DataFrame(columns=["date", "industry", "stage"])
    prev_last = prev[prev["date"] < str(last.date())]
    prev_map = prev_last.sort_values("date").groupby("industry")["stage"].last().to_dict() if len(prev_last) else {}
    for m in inds:
        m["prev"] = prev_map.get(m["name"])
        m["new"] = m["prev"] is not None and m["prev"] != m["stage"]
    today_rows = pd.DataFrame([{"date": str(last.date()), "industry": m["name"], "stage": m["stage"]} for m in inds])
    prev = pd.concat([prev[prev["date"] != str(last.date())], today_rows], ignore_index=True)
    prev.to_csv(hist_p, index=False)

    data = {"asof": last.strftime("%d %b %Y"),
            "generated": dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).strftime("%d %b %Y, %I:%M %p IST"),
            "universe": int(len(ok)), "with_industry": int(len(S)), "industries": inds,
            "stages": STAGES, "order": ORDER, "first_run": len(prev_map) == 0}
    os.makedirs(OUT, exist_ok=True)
    json.dump(data, open(os.path.join(OUT, "data.json"), "w"), separators=(",", ":"))
    tpl = open(os.path.join(ROOT, "template.html"), encoding="utf-8").read()
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))
    print({k: sum(1 for m in inds if m["stage"] == k) for k in ORDER})

    # alert: sirf naye breakout / accumulation
    fresh = [m for m in inds if m["stage"] in ("breakout", "accum") and (m["new"] or data["first_run"])]
    if fresh and os.environ.get("SEND_ALERT") == "1":
        send_alert(data, fresh)


def send_alert(data, fresh):
    import requests
    tok = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    chat = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    if not tok or not chat:
        print("Telegram secrets nahi"); return
    L = [f"💰 *Smart Money — {data['asof']}*", ""]
    for m in fresh[:10]:
        lead = ", ".join(x["s"] for x in m["stocks"][:3])
        bits = [f"delivery {m['deliv_per']}%" if m["deliv_per"] else "",
                f"volume {m['turn_surge']}x" if m["turn_surge"] else "",
                f"{m['n_breakout']} breakout" if m["n_breakout"] else "",
                f"{m['bulk_buys']} bulk buy" if m["bulk_buys"] else ""]
        L.append(f"{STAGES[m['stage']]}: *{m['name']}* ({m['n']} stocks)")
        L.append("   " + " · ".join(b for b in bits if b))
        L.append(f"   Lead: {lead}")
    url = os.environ.get("DASHBOARD_URL", "")
    if url:
        L += ["", f"Dashboard: {url}"]
    L += ["", "_Educational data, buy/sell advice nahi._"]
    txt = "\n".join(L)
    r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data={"chat_id": chat, "text": txt, "parse_mode": "Markdown", "disable_web_page_preview": "true"}, timeout=30)
    if r.status_code != 200:
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data={"chat_id": chat, "text": txt.replace("*", "").replace("_", "")}, timeout=30)
    print("Telegram:", r.status_code)


if __name__ == "__main__":
    main()
