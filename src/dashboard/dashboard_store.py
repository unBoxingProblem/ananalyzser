from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Value of type {type(value).__name__} cannot be stored in a dashboard configuration.")


class DashboardStore:
    """Persist dashboard definitions only; uploaded DataFrames remain in session memory."""

    def __init__(self, path: str | Path | None = None) -> None:
        default = Path(__file__).resolve().parents[2] / ".datapilot" / "dashboards.sqlite3"
        self.path = Path(path or os.environ.get("DATAPILOT_DASHBOARD_DB", default)).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS dashboards (
                dashboard_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                charts_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""")

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        return {"dashboard_id": row["dashboard_id"], "name": row["name"],
                "charts": json.loads(row["charts_json"]), "created_at": row["created_at"],
                "updated_at": row["updated_at"]}

    def list_dashboards(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM dashboards ORDER BY updated_at DESC, name").fetchall()
        return [self._decode(row) for row in rows]

    def get_dashboard(self, dashboard_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM dashboards WHERE dashboard_id = ?", (dashboard_id,)).fetchone()
        return self._decode(row) if row else None

    def create_dashboard(self, name: str = "My Dashboard") -> dict[str, Any]:
        dashboard = {"dashboard_id": str(uuid4()), "name": name.strip() or "Untitled Dashboard",
                     "charts": [], "created_at": _now(), "updated_at": _now()}
        self.save_dashboard(dashboard)
        return dashboard

    def save_dashboard(self, dashboard: dict[str, Any]) -> dict[str, Any]:
        dashboard_id = str(dashboard.get("dashboard_id") or uuid4())
        created_at = dashboard.get("created_at") or _now()
        updated_at = _now()
        name = str(dashboard.get("name") or "Untitled Dashboard").strip() or "Untitled Dashboard"
        charts = dashboard.get("charts", [])
        if not isinstance(charts, list):
            raise ValueError("Dashboard charts must be a list of chart configurations.")
        charts_json = json.dumps(charts, ensure_ascii=False, allow_nan=False, default=_json_default)
        with self._connect() as connection:
            connection.execute("""INSERT INTO dashboards (dashboard_id, name, charts_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(dashboard_id) DO UPDATE SET name=excluded.name,
                    charts_json=excluded.charts_json, updated_at=excluded.updated_at""",
                (dashboard_id, name, charts_json, created_at, updated_at))
        return {"dashboard_id": dashboard_id, "name": name, "charts": charts,
                "created_at": created_at, "updated_at": updated_at}

    def duplicate_dashboard(self, dashboard_id: str, name: str | None = None) -> dict[str, Any]:
        source = self.get_dashboard(dashboard_id)
        if source is None:
            raise KeyError(f"Dashboard {dashboard_id} does not exist.")
        charts = []
        for item in source["charts"]:
            copied = dict(item)
            copied["chart_id"] = str(uuid4())
            copied["title"] = f"{copied.get('title') or 'Chart'} (copy)"
            charts.append(copied)
        duplicate = {"dashboard_id": str(uuid4()), "name": name or f"{source['name']} (copy)",
                     "charts": charts, "created_at": _now(), "updated_at": _now()}
        return self.save_dashboard(duplicate)

    def add_chart(self, dashboard_id: str, chart: dict[str, Any]) -> dict[str, Any]:
        dashboard = self.get_dashboard(dashboard_id)
        if dashboard is None:
            raise KeyError(f"Dashboard {dashboard_id} does not exist.")
        item = dict(chart)
        item["chart_id"] = item.get("chart_id") or str(uuid4())
        dashboard["charts"].append(item)
        return self.save_dashboard(dashboard)

    def update_chart(self, dashboard_id: str, chart_id: str, chart: dict[str, Any]) -> dict[str, Any]:
        dashboard = self.get_dashboard(dashboard_id)
        if dashboard is None:
            raise KeyError(f"Dashboard {dashboard_id} does not exist.")
        if not any(item.get("chart_id") == chart_id for item in dashboard["charts"]):
            raise KeyError(f"Chart {chart_id} does not exist in dashboard {dashboard_id}.")
        item = dict(chart)
        item["chart_id"] = chart_id
        dashboard["charts"] = [item if existing.get("chart_id") == chart_id else existing for existing in dashboard["charts"]]
        return self.save_dashboard(dashboard)

    def duplicate_chart(self, dashboard_id: str, chart_id: str, *, insert_after: bool = True) -> dict[str, Any]:
        dashboard = self.get_dashboard(dashboard_id)
        if dashboard is None:
            raise KeyError(f"Dashboard {dashboard_id} does not exist.")
        index = next((i for i, item in enumerate(dashboard["charts"]) if item.get("chart_id") == chart_id), None)
        if index is None:
            raise KeyError(f"Chart {chart_id} does not exist in dashboard {dashboard_id}.")
        copy = dict(dashboard["charts"][index])
        copy["chart_id"] = str(uuid4())
        copy["title"] = f"{copy.get('title') or copy.get('chart_type', 'Chart')} (copy)"
        dashboard["charts"].insert(index + (1 if insert_after else 0), copy)
        return self.save_dashboard(dashboard)

    def delete_chart(self, dashboard_id: str, chart_id: str) -> dict[str, Any]:
        dashboard = self.get_dashboard(dashboard_id)
        if dashboard is None:
            raise KeyError(f"Dashboard {dashboard_id} does not exist.")
        dashboard["charts"] = [item for item in dashboard["charts"] if item.get("chart_id") != chart_id]
        return self.save_dashboard(dashboard)

    def move_chart(self, dashboard_id: str, chart_id: str, offset: int) -> dict[str, Any]:
        dashboard = self.get_dashboard(dashboard_id)
        if dashboard is None:
            raise KeyError(f"Dashboard {dashboard_id} does not exist.")
        index = next((i for i, item in enumerate(dashboard["charts"]) if item.get("chart_id") == chart_id), None)
        if index is None:
            raise KeyError(f"Chart {chart_id} does not exist in dashboard {dashboard_id}.")
        target = max(0, min(len(dashboard["charts"]) - 1, index + offset))
        dashboard["charts"][index], dashboard["charts"][target] = dashboard["charts"][target], dashboard["charts"][index]
        return self.save_dashboard(dashboard)

    def delete_dashboard(self, dashboard_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM dashboards WHERE dashboard_id = ?", (dashboard_id,))
            return cursor.rowcount > 0
