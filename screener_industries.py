"""
Screener.in /market (NSE/BSE official 4-level classification) → data/screener_industry.csv
Columns: symbol, basic_industry, industry, sector, macro, mcap_cr
Source file: data/screener_raw.json (browser se ek baar nikala, ₹300 cr+ NSE companies)
"""
import os, json, glob
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))


def build():
    raw = json.load(open(os.path.join(ROOT, "data", "screener_raw.json")))
    ind, par = raw["ind"], raw["parents"]
    out = []
    for sym, i, mc in raw["rows"]:
        href, name = ind[int(i)]
        p = href.strip("/").split("/")[1:]
        lv = lambda k: par.get("/market/" + "/".join(p[:k]) + "/")
        out.append({"symbol": sym, "basic_industry": name, "industry": lv(3), "sector": lv(2), "macro": lv(1), "mcap_cr": float(mc)})
    df = pd.DataFrame(out).drop_duplicates("symbol")
    df.to_csv(os.path.join(ROOT, "data", "screener_industry.csv"), index=False)
    return df


if __name__ == "__main__":
    df = build()
    files = sorted(glob.glob(os.path.join(ROOT, "data", "hist", "*.parquet")))
    nse = set(pd.read_parquet(files[-1], columns=["symbol"])["symbol"])
    print(len(df), "companies,", df.basic_industry.nunique(), "basic industries, NSE data match:", df.symbol.isin(nse).sum())
