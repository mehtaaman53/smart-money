"""
5 saal ka NSE bhavcopy (delivery ke saath) — backtest aur lifetime-high ke liye.
Saal-wise files: data/hist/bhav_YYYY.parquet. Roz wala pipeline ise nahi chhoota.
Chalao: python history.py [years=5]
"""
import os, sys, time, datetime as dt
import pandas as pd
import pipeline as P

YEARS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
HD = os.path.join(P.ROOT, "data", "hist")


def main():
    os.makedirs(HD, exist_ok=True)
    start = dt.date.today() - dt.timedelta(days=365 * YEARS)
    for yr in range(start.year, dt.date.today().year + 1):
        path = os.path.join(HD, f"bhav_{yr}.parquet")
        old = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
        have = set(pd.to_datetime(old["date"]).dt.date) if len(old) else set()
        d0 = max(dt.date(yr, 1, 1), start); d1 = min(dt.date(yr, 12, 31), dt.date.today())
        days = [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
        days = [d for d in days if d.weekday() < 5 and d not in have]
        print(f"{yr}: {len(days)} din laane hain (pehle se {len(have)})", flush=True)
        new, miss = [], 0
        for i, d in enumerate(days):
            x = P.fetch_bhav(d)
            if x is None:
                miss += 1
            else:
                new.append(x[["date", "symbol", "series", "prev_close", "high", "low", "close", "volume", "turnover_lacs", "deliv_qty"]])
            time.sleep(0.25)
            if (i + 1) % 100 == 0:
                print(f"   {i+1}/{len(days)} — {len(new)} mile", flush=True)
        if new:
            allx = pd.concat([old] + new, ignore_index=True).drop_duplicates(["date", "symbol"], keep="last")
            for c in ["prev_close", "high", "low", "close", "turnover_lacs"]:
                allx[c] = allx[c].astype("float32")
            allx.to_parquet(path, index=False, compression="zstd")
        print(f"{yr}: +{len(new)} trading days, {miss} chhutti/missing", flush=True)


if __name__ == "__main__":
    main()
