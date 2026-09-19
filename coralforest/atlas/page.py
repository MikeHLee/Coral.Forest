"""The atlas page: one HTML file with a map, the numbers, and the sources.

The page carries its own data inline, so it works from a file path or from
GitHub Pages with no server behind it. Leaflet comes from a CDN and handles
panning and zooming; the coastline is the Natural Earth land layer, embedded in
the page, so the map calls no tile service and no tile licence applies.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Coral.Forest atlas</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<link rel="icon" href="data:,">
<style>
  :root {{ --ink:#1d2327; --muted:#5c6770; --line:#dfe3e6; --warm:#c0392b; }}
  body {{ margin:0; font:16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         color:var(--ink); background:#fbfbfa; }}
  main {{ max-width:1100px; margin:0 auto; padding:32px 20px 64px; }}
  h1 {{ font-size:30px; margin:0 0 6px; }}
  h2 {{ font-size:20px; margin:40px 0 10px; }}
  p.lede {{ font-size:18px; color:var(--muted); margin:0 0 24px; }}
  #map {{ height:520px; border:1px solid var(--line); border-radius:6px; background:#eef6fb; }}
  .leaflet-container {{ background:#eef6fb; }}
  .legend {{ background:#fff; padding:8px 10px; border-radius:4px; line-height:1.5; font-size:12px;
             box-shadow:0 1px 4px rgba(0,0,0,.2); }}
  .legend i {{ width:12px; height:12px; display:inline-block; margin-right:6px; border-radius:50%;
               border:1px solid #333; vertical-align:-1px; }}
  table {{ border-collapse:collapse; width:100%; font-size:14px; margin-top:8px; }}
  th, td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); }}
  th {{ font-weight:600; color:var(--muted); font-size:13px; }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .cards {{ display:flex; flex-wrap:wrap; gap:14px; margin:20px 0 4px; }}
  .card {{ flex:1 1 200px; border:1px solid var(--line); border-radius:6px; padding:12px 14px; background:#fff; }}
  .card .k {{ font-size:26px; font-weight:600; }}
  .card .t {{ font-size:13px; color:var(--muted); }}
  footer {{ margin-top:48px; font-size:13px; color:var(--muted); border-top:1px solid var(--line); padding-top:16px; }}
  code {{ background:#f0f0ee; padding:1px 5px; border-radius:3px; font-size:13px; }}
</style>
</head>
<body>
<main>
<h1>Coral.Forest atlas</h1>
<p class="lede">{lede}</p>

<div class="cards">{cards}</div>

<div id="map"></div>
<p style="font-size:13px;color:var(--muted);margin-top:8px">
Each circle is a reef region with survey records. Colour is the modelled share of coral bleached at the
2050 heat dose; size is the number of surveys. Click a circle for its numbers.</p>

<h2>What the model says</h2>
{method}

<h2>How well it predicts</h2>
{evaluation}

<h2>Communities</h2>
{community}

<footer>
<p><strong>Data.</strong></p>
{sources}
<p>Built {built} by <code>python -m coralforest.atlas.build</code> in
<a href="https://github.com/MikeHLee/Coral.Forest">MikeHLee/Coral.Forest</a>. Code is GPL-3.0.</p>
</footer>
</main>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const REGIONS = {regions_json};
const LAND = {land_json};
const EDGES = {edges_json};
const COLOURS = {colours_json};
function colour(v) {{
  let i = 0;
  while (i < COLOURS.length - 1 && v >= EDGES[i + 1]) i++;
  return COLOURS[i];
}}
const map = L.map('map', {{ worldCopyJump: true, minZoom: 1, maxZoom: 7,
  attributionControl: false }}).setView([2, 70], 2);
L.geoJSON(LAND, {{ style: {{ color: '#cfc9c1', weight: 0.6, fillColor: '#e9e6e1', fillOpacity: 1 }},
  interactive: false }}).addTo(map);
L.control.attribution({{ prefix: false }}).addAttribution(
  'Coastline: Natural Earth (public domain) | Leaflet').addTo(map);
const pct = x => (100 * x).toFixed(1) + '%';
for (const r of REGIONS) {{
  L.circleMarker([r.lat, r.lon], {{
    radius: 4 + 10 * Math.sqrt(r.surveys / {max_surveys}), color: '#333', weight: 0.7,
    fillColor: colour(r.impairment_2050), fillOpacity: 0.85
  }}).addTo(map).bindPopup(
    `<strong>${{r.region}}</strong><br>${{r.surveys}} surveys, to ${{r.last_year}}` +
    `<br>Heat dose: ${{r.dhw_recent.toFixed(2)}} &rarr; ${{(r.dhw_recent + r.warming_c_weeks).toFixed(2)}} C-weeks` +
    `<br>Coral bleached: ${{pct(r.impairment_now)}} (${{pct(r.impairment_now_p05)}}-${{pct(r.impairment_now_p95)}})` +
    `<br>At 2050 heat dose: <b>${{pct(r.impairment_2050)}}</b> (${{pct(r.impairment_2050_p05)}}-${{pct(r.impairment_2050_p95)}})` +
    (r.coral_species == null ? '' : `<br>${{r.coral_species}} hard-coral species recorded`) +
    (r.population_50km == null ? '' : `<br>${{Math.round(r.population_50km).toLocaleString()}} people within 50 km`)
  );
}}
const legend = L.control({{ position: 'bottomleft' }});
legend.onAdd = function () {{
  const div = L.DomUtil.create('div', 'legend');
  div.innerHTML = '<b>Coral bleached at 2050 heat dose</b><br>';
  for (let i = 0; i < COLOURS.length; i++) {{
    div.innerHTML += `<i style="background:${{COLOURS[i]}}"></i>${{Math.round(EDGES[i] * 100)}}-${{Math.round(EDGES[i + 1] * 100)}}%<br>`;
  }}
  return div;
}};
legend.addTo(map);
</script>
</body>
</html>
"""


def _table(frame: pd.DataFrame, columns: Dict[str, str], numeric: Dict[str, str]) -> str:
    head = "".join(f'<th class="{"num" if key in numeric else ""}">{title}</th>'
                   for key, title in columns.items())
    rows = []
    for row in frame.itertuples(index=False):
        cells = []
        for key in columns:
            value = getattr(row, key)
            if key in numeric:
                text = "&mdash;" if pd.isna(value) else numeric[key].format(value)
                cells.append(f'<td class="num">{text}</td>')
            else:
                cells.append(f"<td>{value}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _cards(summary: Dict) -> str:
    within = summary["within_between"]["within_region_per_c_week"]
    items = [
        (f"{summary['surveys']:,}", f"surveys, {summary['years'][0]}-{summary['years'][1]}"),
        (f"{summary['regions']}", "reef regions with enough surveys"),
        (f"{within:+.2f}", "logit of bleached share per C-week, within a region"),
        (f"{summary['impairment_now_median']:.0%} &rarr; {summary['impairment_2050_median']:.0%}",
         "median share of coral bleached, today and at the 2050 heat dose"),
    ]
    return "".join(f'<div class="card"><div class="k">{k}</div><div class="t">{t}</div></div>' for k, t in items)


def write(path: Path, *, summary: Dict, regions: pd.DataFrame, countries: pd.DataFrame,
          exposure: pd.DataFrame, offline: bool = False) -> Path:
    from . import sources as source_registry
    from .figures import RISK_COLOURS, RISK_EDGES

    land = json.loads(source_registry.fetch("land", offline=offline).read_text())

    keep = ["region", "lat", "lon", "surveys", "last_year", "dhw_recent", "warming_c_weeks",
            "impairment_now", "impairment_now_p05", "impairment_now_p95",
            "impairment_2050", "impairment_2050_p05", "impairment_2050_p95"]
    keep += [column for column in ("coral_species", "population_50km") if column in regions.columns]
    data = regions[keep].dropna(subset=["lat", "lon"]).round(4)

    scores = pd.DataFrame(summary["evaluation"])
    scores = scores[scores["outcome"] == "impairment"][["label", "n", "mae", "mae_baseline", "skill", "spearman"]]
    evaluation = _table(scores, {
        "label": "Test", "n": "Region-years", "mae": "Mean error", "mae_baseline": "Error of the mean",
        "skill": "Skill", "spearman": "Rank correlation"},
        {"n": "{:,.0f}", "mae": "{:.3f}", "mae_baseline": "{:.3f}", "skill": "{:+.2f}", "spearman": "{:.2f}"})

    coefficients = pd.DataFrame(summary["coefficients"])
    within = summary["within_between"]
    method = (
        "<p>Every region-year gets the mean heat dose of its surveys, in degree heating weeks, and the mean "
        "share of surveyed coral that was bleached. The heat dose enters twice: the region's own long-run mean, "
        "and the difference between this year and that mean. Only the second one drives the scenarios, so a "
        "region that is always hot cannot pass for a region that got hot this year.</p>"
        f"<p>The fitted step is <b>{within['within_region_per_c_week']:+.2f}</b> on the logit scale per "
        f"degree heating week within a region, against {within['between_region_per_c_week']:+.2f} between "
        f"regions, on {within['n']} region-years.</p>"
        + _table(coefficients[coefficients["outcome"] == "impairment"][["term", "mean", "sd"]],
                 {"term": "Term", "mean": "Coefficient", "sd": "Standard deviation"},
                 {"mean": "{:+.3f}", "sd": "{:.3f}"})
        + f"<p>The 2050 heat dose adds each region's own trend: the nearest Coral Reef Watch station's annual "
          f"maximum heat dose has risen {summary['station_trend_per_decade_median']:+.2f} degree heating weeks "
          f"per decade at the median station since 1985. Station and survey measures rank years alike "
          f"(rank correlation {summary['calibration']['spearman']:.2f}) but sit on different scales, so the "
          f"step is multiplied by {summary['calibration']['slope']:.2f} before it reaches the model.</p>")

    if exposure is not None and not exposure.empty:
        wide = exposure.pivot_table(index="country", columns="service",
                                    values=["exposed_now", "exposed_2050"], aggfunc="sum")
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]
        wide = wide.reset_index()
        if "population_50km" in countries.columns:
            wide = wide.merge(countries[["country", "population_50km"]], on="country", how="left")
        sort_key = "exposed_2050_shoreline" if "exposed_2050_shoreline" in wide else wide.columns[1]
        wide = wide.nlargest(15, sort_key)
        columns = {"country": "Country"}
        numeric = {}
        if "population_50km" in wide.columns:
            columns["population_50km"] = "People within 50 km of a surveyed reef"
            numeric["population_50km"] = "{:,.0f}"
        for service, title in (("shoreline", "People behind the impaired share"),
                               ("tourism", "Reef tourism, USD a year"),
                               ("fishery", "Reef fishers"),
                               ("habitat", "Coral species behind it")):
            column = f"exposed_2050_{service}"
            if column in wide:
                columns[column] = title
                numeric[column] = "{:,.0f}"
        community = (
            "<p>A country's reefs pass three services to the people near them: reef fishing, reef tourism, and "
            "lower flood exposure behind the reef. The flow model sends each benefit through the reef node, "
            "whose loss rate is the share of coral bleached, so the number below is the part of the benefit "
            "that sits behind impaired reef at the 2050 heat dose. Read it as exposure, not as damage: it does "
            "not say that this value is lost, and it treats every unit of the benefit as coral-dependent, which "
            "is an upper bound. The species column is a count for that country alone: species counts "
            "cannot be added across countries, because a species that lives in several of them would be "
            "counted once for each.</p>" + _table(wide[list(columns)], columns, numeric))
    else:
        community = ("<p>The community tables are not present in this build, so the page shows reef risk only. "
                     "Add them under <code>data/community/</code> and run the build again.</p>")

    sources_html = "<ul>" + "".join(
        f'<li>{s["citation"]} Licence: <a href="{s["licence_url"]}">{s["licence"]}</a>.</li>'
        for s in summary["sources"] + summary.get("community_sources", [])) + "</ul>"

    html = TEMPLATE.format(
        lede=(f"{summary['surveys']:,} reef surveys from {summary['years'][0]} to {summary['years'][1]}, "
              f"the heat dose each one met, and what the reefs behind them are worth to the communities nearby."),
        cards=_cards(summary), method=method, evaluation=evaluation, community=community,
        sources=sources_html, built=summary["built"],
        regions_json=json.dumps(data.to_dict("records")),
        land_json=json.dumps(land, separators=(",", ":")),
        edges_json=json.dumps(RISK_EDGES), colours_json=json.dumps(RISK_COLOURS),
        max_surveys=int(np.nanmax(data["surveys"])) if len(data) else 1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path
