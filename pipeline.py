"""
SMART MONEY — Part 1: data pipeline (NSE official data)
- Bhavcopy (price, volume, turnover, delivery) ~1 saal, roz naya din judta hai
- Circuit bands (sec_list), Bulk/Block deals (roz jama hote hain)
- Industry list (Yahoo, ek baar + naye stocks), universe filter (Aman ke rules)
"""
import io, os, json, time, datetime as dt
import pandas as pd, numpy as np, requests

ROOT = os.path.dirname(os.path.abspath(__file__))
D = lambda *p: os.path.join(ROOT, "data", *p)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
KEEP_SERIES = {"EQ", "BE", "BZ"}          # BE/BZ = T2T — Aman: rakhna hai. SME (SM/ST) + bonds/ETF bahar
HIST_DAYS = 400                           # calendar days history
MIN_TURNOVER_LACS = 20                    # 20 din avg turnover >= ₹20 lakh
CIRCUIT_DAYS, CIRCUIT_MAX = 62, 40        # 62 din mein 40+ din circuit → bahar
S = requests.Session(); S.headers.update(UA)


def _get(url):
    for _ in range(2):
        try:
            r = S.get(url, timeout=25)
            return r
        except Exception:
            time.sleep(3)
    return None


# ── Bhavcopy ───────────────────────────────────────────────────────────
def fetch_bhav(d):
    r = _get(f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d:%d%m%Y}.csv")
    if r is None or r.status_code != 200 or "SYMBOL" not in r.text[:200]:
        return None
    df = pd.read_csv(io.StringIO(r.text)); df.columns = [c.strip() for c in df.columns]
    df["SERIES"] = df["SERIES"].astype(str).str.strip()
    df = df[df["SERIES"].isin(KEEP_SERIES)]
    num = lambda c: pd.to_numeric(df[c].astype(str).str.strip().replace({"-": None, "": None}), errors="coerce")
    return pd.DataFrame({"date": pd.Timestamp(d), "symbol": df["SYMBOL"].str.strip(), "series": df["SERIES"],
                         "prev_close": num("PREV_CLOSE"), "open": num("OPEN_PRICE"), "high": num("HIGH_PRICE"),
                         "low": num("LOW_PRICE"), "close": num("CLOSE_PRICE"), "volume": num("TTL_TRD_QNTY"),
                         "turnover_lacs": num("TURNOVER_LACS"), "trades": num("NO_OF_TRADES"),
                         "deliv_qty": num("DELIV_QTY"), "deliv_per": num("DELIV_PER")})


def update_bhav():
    path = D("bhav.parquet")
    hist = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
    start = (hist["date"].max().date() + dt.timedelta(days=1)) if len(hist) else dt.date.today() - dt.timedelta(days=HIST_DAYS)
    days = [start + dt.timedelta(days=i) for i in range((dt.date.today() - start).days + 1)]
    days = [d for d in days if d.weekday() < 5]
    print(f"Bhavcopy: {len(days)} din check")
    new = []
    for i, d in enumerate(days):
        x = fetch_bhav(d)
        if x is not None:
            new.append(x)
        time.sleep(0.3)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(days)} — {len(new)} trading days")
    if new:
        hist = pd.concat([hist] + new, ignore_index=True)
        hist = hist[hist["date"] >= pd.Timestamp(dt.date.today() - dt.timedelta(days=HIST_DAYS))]
        hist = hist.drop_duplicates(["date", "symbol"], keep="last").sort_values(["symbol", "date"])
        hist.to_parquet(path, index=False)
    print(f"Bhavcopy: +{len(new)} naye din, total {hist['date'].nunique() if len(hist) else 0} din")
    return hist


def adjust_prices(h):
    """Split/bonus adjust: NSE ka prev_close corporate action ke baad adjust hota hai.
    factor = prev_close(t) / close(t-1); usse pichhle saare prices ko scale karo."""
    h = h.sort_values(["symbol", "date"]).copy()
    prev_c = h.groupby("symbol")["close"].shift(1)
    f = (h["prev_close"] / prev_c).where(lambda x: (x - 1).abs() > 0.03, 1.0).fillna(1.0)
    # factor din t pe laga hai → t se pehle ke din adjust honge
    h["_f"] = f
    h["_cum"] = h.iloc[::-1].groupby("symbol")["_f"].cumprod().iloc[::-1]
    h["_cum"] = h.groupby("symbol")["_cum"].shift(-1).fillna(1.0)
    for c in ["open", "high", "low", "close", "prev_close"]:
        h[c] = h[c] * h["_cum"]
    h["volume_adj"] = h["volume"] / h["_cum"]
    return h.drop(columns=["_f", "_cum"])


# ── Bands, deals ───────────────────────────────────────────────────────
def update_bands():
    r = _get("https://nsearchives.nseindia.com/content/equities/sec_list.csv")
    if r is not None and r.status_code == 200 and "Band" in r.text[:100]:
        b = pd.read_csv(io.StringIO(r.text)); b.columns = [c.strip() for c in b.columns]
        b.to_csv(D("bands.csv"), index=False)
    return pd.read_csv(D("bands.csv")) if os.path.exists(D("bands.csv")) else pd.DataFrame(columns=["Symbol", "Band"])


def update_deals():
    path = D("deals.csv")
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    parts = []
    for kind in ["bulk", "block"]:
        r = _get(f"https://nsearchives.nseindia.com/content/equities/{kind}.csv")
        if r is not None and r.status_code == 200 and "Symbol" in r.text[:200]:
            x = pd.read_csv(io.StringIO(r.text)); x.columns = [c.strip() for c in x.columns]; x["kind"] = kind
            parts.append(x)
    if parts:
        allx = pd.concat([old] + parts, ignore_index=True).drop_duplicates()
        allx.to_csv(path, index=False)
        return allx
    return old


# ── Universe filter (Aman ke rules) ────────────────────────────────────
def universe(h, bands):
    last = h["date"].max()
    days = sorted(h["date"].unique())
    d20, d62 = days[-20:], days[-CIRCUIT_DAYS:]
    cur = h[h["date"] == last][["symbol", "series", "close"]]
    t20 = h[h["date"].isin(d20)].groupby("symbol")["turnover_lacs"].mean().rename("avg_turnover_lacs")
    band = bands.rename(columns={"Symbol": "symbol"})[["symbol", "Band"]].drop_duplicates("symbol")
    band["Band"] = pd.to_numeric(band["Band"], errors="coerce")
    w = h[h["date"].isin(d62)].merge(band, on="symbol", how="left")
    chg = (w["close"] / w["prev_close"] - 1) * 100
    at_up = (w["close"] >= w["high"] - 1e-6) & (chg >= w["Band"] - 0.15)
    at_dn = (w["close"] <= w["low"] + 1e-6) & (chg <= -(w["Band"] - 0.15))
    w["circuit"] = (at_up | at_dn) & w["Band"].notna()
    circ = w.groupby("symbol")["circuit"].sum().rename("circuit_days")
    u = cur.merge(t20, on="symbol", how="left").merge(circ, on="symbol", how="left").merge(band, on="symbol", how="left")
    u["circuit_days"] = u["circuit_days"].fillna(0).astype(int)
    u["reason"] = ""
    u.loc[u["avg_turnover_lacs"].fillna(0) < MIN_TURNOVER_LACS, "reason"] = "low liquidity"
    u.loc[u["circuit_days"] >= CIRCUIT_MAX, "reason"] = "circuit stock"
    u["ok"] = u["reason"] == ""
    return u.sort_values("symbol")


# ── Industry (Yahoo, cached) ───────────────────────────────────────────
INDIAN_NAMES = {
    "Specialty Industrial Machinery": "Precision Engg / Industrial Machinery",
    "Textile Manufacturing": "Textiles", "Apparel Manufacturing": "Readymade Garments",
    "Apparel Retail": "Apparel Retail (Showrooms)", "Department Stores": "Value Retail / Dept Stores",
    "Specialty Chemicals": "Specialty Chemicals", "Chemicals": "Commodity Chemicals",
    "Agricultural Inputs": "Agrochem & Fertilisers", "Drug Manufacturers - Specialty & Generic": "Pharma (Generic)",
    "Auto Parts": "Auto Ancillary", "Aerospace & Defense": "Defence & Aerospace",
    "Electrical Equipment & Parts": "Electrical Equipment", "Engineering & Construction": "EPC / Construction",
    "Capital Markets": "Capital Markets (Brokers/AMC)", "Credit Services": "NBFC", "Banks - Regional": "Banks",
    "Information Technology Services": "IT Services", "Software - Application": "Software Products",
}


def update_industry(symbols, max_new=1400):
    path = D("industry.csv")
    ind = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["symbol", "sector", "industry_raw", "industry"])
    have = set(ind["symbol"])
    todo = [s for s in symbols if s not in have][:max_new]
    print(f"Industry: {len(todo)} naye stocks Yahoo se (cache: {len(have)})")
    if not todo:
        return ind
    import yfinance as yf
    rows, fails = [], 0
    for i, s in enumerate(todo):
        sec = raw = None
        try:
            info = yf.Ticker(s + ".NS").info
            sec, raw = info.get("sector"), info.get("industry")
            fails = 0
        except Exception as e:
            fails += 1
            if "Too Many" in str(e) or "429" in str(e):
                print("  Yahoo rate limit — 60s ruk raha hoon"); time.sleep(60)
            if fails > 25:
                print("  Yahoo bahut fail — baaki agle run mein"); break
        if raw:
            rows.append({"symbol": s, "sector": sec, "industry_raw": raw, "industry": INDIAN_NAMES.get(raw, raw)})
        time.sleep(0.35)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(todo)}")
    if rows:
        ind = pd.concat([ind, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("symbol", keep="last")
        ind.to_csv(path, index=False)
    return ind


def main():
    os.makedirs(D(), exist_ok=True)
    h = update_bhav()
    bands = update_bands()
    deals = update_deals()
    u = universe(h, bands)
    u.to_csv(D("universe.csv"), index=False)
    ok = u[u["ok"]]["symbol"].tolist()
    ind = update_industry(ok)
    status = {"updated": dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).strftime("%d %b %Y %I:%M %p IST"),
              "bhav_days": int(h["date"].nunique()), "last_date": str(h["date"].max().date()),
              "stocks_total": int(len(u)), "stocks_ok": int(u["ok"].sum()),
              "removed": u["reason"].value_counts().drop("", errors="ignore").to_dict(),
              "series_ok": u[u["ok"]]["series"].value_counts().to_dict(),
              "industry_known": int(ind["symbol"].isin(ok).sum()), "industries": int(ind[ind["symbol"].isin(ok)]["industry"].nunique()),
              "deals_rows": int(len(deals))}
    json.dump(status, open(D("status.json"), "w"), indent=1)
    print(json.dumps(status, indent=1))


if __name__ == "__main__":
    main()
