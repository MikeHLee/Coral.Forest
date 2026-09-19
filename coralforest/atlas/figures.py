"""The two pictures the atlas produces: a world map and a country bar chart.

Both are drawn with matplotlib on a plain latitude-longitude grid. The coastline
is Natural Earth 1:110m land, which is public domain.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import sources

RISK_COLOURS = ["#2b6ca3", "#6bb1d6", "#f2d06b", "#e8913c", "#c0392b"]
RISK_EDGES = [0.0, 0.05, 0.10, 0.20, 0.35, 1.0]


def _land_polygons(offline: bool = False):
    doc = json.loads(sources.fetch("land", offline=offline).read_text())
    for feature in doc["features"]:
        geometry = feature["geometry"]
        parts = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        for polygon in parts:
            yield np.asarray(polygon[0], dtype=float)


def risk_colour(value: float) -> str:
    index = int(np.clip(np.searchsorted(RISK_EDGES, value, side="right") - 1, 0, len(RISK_COLOURS) - 1))
    return RISK_COLOURS[index]


def world_map(regions: pd.DataFrame, path: Path, *, column: str = "impairment_2050",
              title: str = "", subtitle: str = "", offline: bool = False) -> Path:
    """Survey regions as circles: colour is the modelled share of coral bleached."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(figsize=(14, 7.2))
    for polygon in _land_polygons(offline=offline):
        axes.fill(polygon[:, 0], polygon[:, 1], facecolor="#e9e6e1", edgecolor="#cfc9c1", linewidth=0.4)
    frame = regions.dropna(subset=["lat", "lon", column]).sort_values("surveys")
    sizes = 18 + 120 * np.sqrt(frame["surveys"] / frame["surveys"].max())
    axes.scatter(frame["lon"], frame["lat"], s=sizes,
                 c=[risk_colour(v) for v in frame[column]], alpha=0.85,
                 edgecolor="#333333", linewidth=0.4, zorder=3)
    axes.set_xlim(-180, 180)
    axes.set_ylim(-45, 45)
    axes.set_xticks(range(-180, 181, 60))
    axes.set_yticks(range(-40, 41, 20))
    axes.tick_params(labelsize=8, colors="#555555")
    for spine in axes.spines.values():
        spine.set_color("#cccccc")
    axes.set_facecolor("#f7fbfd")
    if title:
        axes.set_title(title, fontsize=15, loc="left", pad=14)
    if subtitle:
        axes.text(0, 1.015, subtitle, transform=axes.transAxes, fontsize=9, color="#666666")
    labels = [f"{int(RISK_EDGES[i] * 100)}-{int(RISK_EDGES[i + 1] * 100)}%" for i in range(len(RISK_COLOURS))]
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=colour,
                      markeredgecolor="#333333", label=label)
               for colour, label in zip(RISK_COLOURS, labels)]
    legend = axes.legend(handles=handles, title="Coral bleached", loc="lower left", fontsize=8,
                         framealpha=0.9, ncol=5)
    legend.get_title().set_fontsize(8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def country_bars(table: pd.DataFrame, path: Path, *, value: str = "exposed_2050",
                 label: str = "", top: int = 15, title: str = "") -> Optional[Path]:
    """Horizontal bars for the countries with the most benefit exposed."""
    if table.empty:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = table.nlargest(top, value).iloc[::-1]
    figure, axes = plt.subplots(figsize=(8, 0.42 * len(frame) + 1.6))
    axes.barh(frame["country"], frame[value], color="#c0392b", alpha=0.85)
    if {"exposed_now"} <= set(frame.columns) and value != "exposed_now":
        axes.barh(frame["country"], frame["exposed_now"], color="#e8913c", alpha=0.95, label="today")
        axes.legend(["2050 heat dose", "today"], fontsize=8, loc="lower right")
    axes.set_xlabel(label, fontsize=9)
    axes.tick_params(labelsize=8)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    if title:
        axes.set_title(title, fontsize=12, loc="left")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
