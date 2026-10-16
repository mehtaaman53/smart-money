"""Share count (Yahoo) cache → market cap = shares x NSE price. Sirf naye/missing symbols laata hai; 30 din purane refresh."""
import os, time, datetime as dt
import pandas as pd
import micro_industries as M

ROOT = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(ROOT, "data", "shares.csv")


def update(max_n=700):
    import yfinance as yf
    old = pd.read_csv(PATH) if os.path.exists(PATH) else pd.DataFrame(columns=["symbol", "shares", "updated"])
    fresh = old[pd.to_datetime(old["updated"]) >= pd.Timestamp.today() - pd.Timedelta(days=30)]
    have = set(fresh["symbol"])
    want = sorted({s for v in M.GROUPS.values() for s in v.split()})
    todo = [s for s in want if s not in have][:max_n]
    print(f"Shares: {len(todo)} laane hain")
    rows, fails = [], 0
    for s in todo:
        n = None
        try:
            fi = yf.Ticker(s + ".NS").fast_info
            n = fi.get("shares") if hasattr(fi, "get") else getattr(fi, "shares", None)
            if not n:
                n = yf.Ticker(s + ".NS").info.get("sharesOutstanding")
        except Exception as e:
            fails += 1
            if "429" in str(e) or "Too Many" in str(e):
                time.sleep(60)
        if n:
            rows.append({"symbol": s, "shares": float(n), "updated": str(dt.date.today())})
        time.sleep(0.3)
    if rows:
        out = pd.concat([old[~old["symbol"].isin([r["symbol"] for r in rows])], pd.DataFrame(rows)], ignore_index=True)
        out.to_csv(PATH, index=False)
    print(f"Shares: +{len(rows)}, fail {fails}")


if __name__ == "__main__":
    update()
