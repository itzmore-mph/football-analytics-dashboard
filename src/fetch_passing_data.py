from pathlib import Path
import json
import csv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main(match_id: int = 15946):
    events_path = DATA_DIR / f"{match_id}.json"
    out_path = DATA_DIR / "passing_data.csv"

    if not events_path.exists():
        raise FileNotFoundError(
            f"{events_path} not found. Run fetch_statsbomb.py first."
        )

    with events_path.open("r", encoding="utf-8") as f:
        events = json.load(f)

    rows = []
    for e in events:
        if _get(e, "type", "name") != "Pass":
            continue

        start = e.get("location")
        end = _get(e, "pass", "end_location")
        if not (isinstance(start, list) and len(start) >= 2):
            continue
        if not (isinstance(end, list) and len(end) >= 2):
            continue

        passer = _get(e, "player", "name")
        receiver = _get(e, "pass", "recipient", "name")
        if not receiver:
            continue

        team_passer = _get(e, "team", "name")
        team_receiver = _get(e, "pass", "recipient", "team", "name")

        rows.append({
            "match_id": match_id,
            "minute": e.get("minute"),
            "second": e.get("second"),
            "passer": passer,
            "receiver": receiver,
            "team_passer": team_passer,
            "team_receiver": team_receiver,
            "start_x": float(start[0]),
            "start_y": float(start[1]),
            "end_x": float(end[0]),
            "end_y": float(end[1]),
            # default node position (receiver position)
            "x": float(end[0]),
            "y": float(end[1]),
        })

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "match_id", "minute", "second",
        "passer", "receiver",
        "team_passer", "team_receiver",
        "start_x", "start_y", "end_x", "end_y",
        "x", "y",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    print(f"Wrote {len(rows)} passes → {out_path}")


if __name__ == "__main__":
    main()
