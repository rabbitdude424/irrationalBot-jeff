from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

# Allow running from repo root or script directory
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plotting import SymbolPlotter, SECTOR_COLORS  # type: ignore

def main() -> None:
    plotter = SymbolPlotter.from_scan()

    # Select symbols whose company sector is Energy minerals (case-insensitive)
    target_sector = "energy minerals"
    symbols: List[str] = []
    for sym in sorted(plotter.known()):
        info = plotter._dirs.get(sym)  # temporary script: access internal map
        if not info or not info.company:
            continue
        sector = None
        # Case-insensitive key lookup for 'Sector'
        for k, v in info.company.items():
            if str(k).lower() == "sector":
                sector = str(v) if v is not None else None
                break
        if sector and sector.strip().lower() == target_sector:
            symbols.append(sym)

    if not symbols:
        print("No symbols found for sector 'Energy minerals'.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    shown: List[str] = []

    for sym in symbols:
        df = plotter._cache.get(sym) or plotter._load_df_if_needed(sym)  # lazy load
        if df is None or df.empty or "close" not in df.columns:
            continue

        # Determine the last three distinct calendar days present in the index
        # candles_to_df stores timezone-aware NY timestamps
        days = pd.Index(df.index.date)
        unique_days = pd.unique(days)
        if unique_days.size == 0:
            continue
        last_three = list(pd.Series(unique_days).sort_values().iloc[-3:])

        # Slice to the last three days
        mask = days.isin(last_three)
        sub = df.loc[mask, ["close"]]
        if sub.empty:
            continue

        # Normalize to percent change from first available close in window
        series = sub["close"].copy()
        # Guard against zero base (unlikely for prices); skip if cannot normalize
        base = series.iloc[0]
        if base == 0:
            nz = series[series != 0]
            if nz.empty:
                continue
            base = nz.iloc[0]
        series = (series / base) - 1.0

        # Color by sector constant if available
        color = None
        sector = None
        if plotter._dirs.get(sym) and plotter._dirs[sym].company:
            for k, v in plotter._dirs[sym].company.items():
                if str(k).lower() == "sector":
                    sector = str(v) if v is not None else None
                    break
        if isinstance(sector, str):
            color = SECTOR_COLORS.get(sector.strip().lower())

        ax.plot(series.index, series.values, label=sym, linewidth=1.2, color=color)
        shown.append(sym)

    title = "Energy minerals: normalized close % (last 3 days)"
    ax.set_title(title if shown else f"{title} — no data")
    ax.set_xlabel("Time")
    ax.set_ylabel("Percent change")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    if shown:
        ax.legend(loc="best", frameon=False, ncol=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    plt.show()
    print(f"Plotted {len(shown)} symbols: {', '.join(shown)}")


if __name__ == "__main__":
    main()
