from __future__ import annotations

import csv
import json
from pathlib import Path
from shutil import rmtree

import rich
from rich.console import Console
from rich.prompt import Prompt

from file_io import (
    ensure_data_dirs,
    ensure_temp_dir,
)
from questrade import api

TOKEN_PATH = Path("questrade") / "token.json"

def _clear_temp_dir() -> None:
    temp_dir = ensure_temp_dir()
    if temp_dir.exists():
        for child in temp_dir.iterdir():
            if child.is_dir():
                rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)

def _load_token_payload() -> dict:
    if not TOKEN_PATH.exists():
        ensure_data_dirs()
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "access_token": "",
            "refresh_token": "",
            "token_type": "Bearer",
            "api_server": "https://api01.iq.questrade.com/",
            "active_time": 0,
            "issued_at": 0,
            "login_server": "https://login.questrade.com",
        }
        TOKEN_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    return json.loads(TOKEN_PATH.read_text(encoding="utf-8"))

def _persist_token(payload: dict) -> None:
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _prompt_refresh_token(console: Console) -> bool:
    payload = _load_token_payload()
    console.print("[bold]Refresh Token Management[/bold]")
    console.print("The current refresh token is hidden. Provide a new one or press Enter to reuse it.")
    new_token = Prompt.ask("Enter refresh token", default="", show_default=False).strip()
    if not new_token:
        if not payload.get("refresh_token"):
            console.print("[red]No refresh token available. Please provide one to continue online.[/red]")
            return False
        return True
    payload["refresh_token"] = new_token
    payload["access_token"] = ""
    payload["active_time"] = 0
    payload["issued_at"] = 0
    _persist_token(payload)
    console.print("[green]Refresh token saved. A new access token will be negotiated on connect.[/green]")
    return True

def get_tickers() -> list[str]:
    # Resolve target path
    path = Path("data") / "account" / "current_companies.csv"
    # Ensure parent directory exists and file is present
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # Create an empty CSV with a minimal header so downstream parsing works
        path.write_text("Ticker\n", encoding="utf-8")

    tickers: list[str] = []
    seen: set[str] = set()

    # Read CSV robustly: prefer DictReader with 'Ticker' column; fallback to first column.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        # Peek first line to decide reader strategy while keeping stream simple
        first_pos = handle.tell()
        first_line = handle.readline()
        handle.seek(first_pos)

        if "," in first_line or first_line.strip().lower() == "ticker":
            reader = csv.DictReader(handle)
            fieldnames = [f.strip() for f in (reader.fieldnames or [])]
            if "Ticker" in fieldnames:
                for row in reader:
                    val = (row.get("Ticker") or "").strip()
                    if not val:
                        continue
                    sym = val.upper()
                    if sym not in seen:
                        seen.add(sym)
                        tickers.append(sym)
            else:
                # Fall back to positional first column if header missing/unknown
                handle.seek(first_pos)
                rdr = csv.reader(handle)
                for i, cols in enumerate(rdr):
                    if not cols:
                        continue
                    cell = (cols[0] or "").strip()
                    if i == 0 and cell.lower() == "ticker":
                        continue  # skip header
                    if not cell:
                        continue
                    sym = cell.upper()
                    if sym not in seen:
                        seen.add(sym)
                        tickers.append(sym)
        else:
            # Single-column list without commas
            for i, line in enumerate(handle):
                cell = line.strip()
                if i == 0 and cell.lower() == "ticker":
                    continue
                if not cell:
                    continue
                sym = cell.upper()
                if sym not in seen:
                    seen.add(sym)
                    tickers.append(sym)

    return tickers

def main(console: Console) -> None:
    console.print("[bold]no_money: Questrade data orchestrator[/bold]")
    console.print("Select a mode:")
    console.print("1. Online - update token and connect")
    console.print("2. Quit")
    choice = Prompt.ask("Mode", default="1")

    if choice == "2":
        console.print("[yellow]Quitting.[/yellow]")
        return
    if not _prompt_refresh_token(console):
        console.print("[red]Cannot start without a refresh token.[/red]")
        return

    ## Connect to Questrade's API.
    with api.QuestradeSession() as session:
        token = session.get_token()
        console.log(f"Connected to {token['api_server']}")

        ## Fetch the symbol's candles.
        symbols = get_tickers()
        if symbols:
            api.collect_symbol_data(session, symbols, days=30)
            console.print("[green]Symbol data collection completed.[/green]")

    _clear_temp_dir()
    console.print("[bold]Session closed.[/bold]")

if __name__ == "__main__":
    main(rich.get_console())
