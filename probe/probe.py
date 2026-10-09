"""Test: Smart Money ke liye kaun sa NSE data GitHub se milta hai."""
import io, json, time, zipfile, datetime as dt, requests
import pandas as pd
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36", "Accept": "*/*"}
out = {}
def rec(k, v): out[k] = v; print(k, json.dumps(v, default=str)[:600])
def get(url):
    try:
        r = requests.get(url, headers=UA, timeout=25); return r
    except Exception as e:
        class X: status_code = 0; text = str(e); content = b""
        return X()
def last_days(n=8):
    d = dt.date.today()
    for i in range(n):
        x = d - dt.timedelta(days=i)
        if x.weekday() < 5: yield x

# 1) Full bhavcopy with delivery
for d in last_days():
    r = get(f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d:%d%m%Y}.csv")
    if r.status_code == 200 and "SYMBOL" in r.text[:200]:
        df = pd.read_csv(io.StringIO(r.text)); df.columns = [c.strip() for c in df.columns]
        rec("bhav_full", {"date": str(d), "rows": len(df), "cols": list(df.columns),
                          "series": df["SERIES"].str.strip().value_counts().head(12).to_dict(),
                          "sample": df.head(2).to_dict("records")}); break
else:
    rec("bhav_full", {"status": "not found"})

# 2) UDiFF bhavcopy
for d in last_days():
    r = get(f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip")
    if r.status_code == 200 and r.content[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(r.content)); df = pd.read_csv(z.open(z.namelist()[0]))
        rec("udiff", {"date": str(d), "rows": len(df), "cols": list(df.columns)}); break
else:
    rec("udiff", {"status": "not found"})

# 3) Bulk / block deals
for name in ["bulk", "block"]:
    r = get(f"https://nsearchives.nseindia.com/content/equities/{name}.csv")
    rec(name, {"status": r.status_code, "head": r.text[:300]})

# 4) Price bands
for url in ["https://nsearchives.nseindia.com/content/equities/sec_list.csv",
            f"https://nsearchives.nseindia.com/content/equities/sec_list_{dt.date.today():%d%m%Y}.csv"]:
    r = get(url); rec("band " + url.split("/")[-1], {"status": r.status_code, "head": r.text[:250]})

# 5) Equity master list
r = get("https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv")
rec("equity_list", {"status": r.status_code, "lines": r.text.count("\n"), "head": r.text[:200]})
r = get("https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITIES.csv")
rec("sme_list", {"status": r.status_code, "lines": r.text.count("\n")})

# 6) Industry: Nifty Total Market constituents (has Industry col)
for f in ["ind_niftytotalmarket_list.csv", "ind_niftymicrocap250_list.csv"]:
    r = get(f"https://niftyindices.com/IndexConstituent/{f}")
    try:
        df = pd.read_csv(io.StringIO(r.text))
        rec("ni " + f, {"status": r.status_code, "rows": len(df), "cols": list(df.columns),
                        "industries": df.iloc[:, 1].nunique() if "Industry" in df.columns else None,
                        "sample_ind": df["Industry"].value_counts().head(25).to_dict() if "Industry" in df.columns else None})
    except Exception as e:
        rec("ni " + f, {"status": r.status_code, "err": str(e)[:100]})

# 7) Yahoo industry (fallback) for few tickers
try:
    import yfinance as yf
    res = {}
    for t in ["AZAD.NS", "KPRMILL.NS", "TRENT.NS", "AARTIIND.NS", "MTARTECH.NS", "VMART.NS"]:
        try:
            i = yf.Ticker(t).info; res[t] = [i.get("sector"), i.get("industry")]
        except Exception as e:
            res[t] = str(e)[:60]
        time.sleep(1)
    rec("yahoo_industry", res)
except Exception as e:
    rec("yahoo_industry", {"err": str(e)[:150]})

# 8) NSE quote API (basic industry)
s = requests.Session(); s.headers.update(UA)
try:
    s.get("https://www.nseindia.com/", timeout=20)
    r = s.get("https://www.nseindia.com/api/quote-equity?symbol=AZAD", timeout=20)
    rec("nse_quote", {"status": r.status_code, "head": r.text[:300]})
except Exception as e:
    rec("nse_quote", {"err": str(e)[:150]})

json.dump(out, open("probe/result.json", "w"), indent=1, default=str)
