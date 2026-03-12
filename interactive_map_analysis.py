from __future__ import annotations

import csv
import json
import threading
import webbrowser
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

PVGIS_TMY_URL = "https://re.jrc.ec.europa.eu/api/tmy"
SIGMA = 5.670374419e-8
ROOT = Path(__file__).resolve().parent
ROOF_DIR = ROOT / "roof analysis"
HAT_DIR = ROOT / "hat analysis"
HOURLY_ANALYSIS_DIR = ROOT / "Reports" / "Hourly analysis"
HOURLY_ROOF_HTML_DIR = HOURLY_ANALYSIS_DIR / "Roof analysis"
HOURLY_HAT_HTML_DIR = HOURLY_ANALYSIS_DIR / "Hat analysis"


@dataclass
class RoofInputs:
    solar_irradiance_w_m2: float
    ambient_temp_c: float
    sky_temp_c: float
    convective_h_w_m2k: float
    baseline_solar_absorptance: float = 0.85
    baseline_ir_emissivity: float = 0.90
    ant_solar_absorptance: float = 0.35
    ant_ir_emissivity: float = 0.95
    indoor_temp_c: float = 24.0
    roof_u_value_w_m2k: float = 1.2
    cooling_cop: float = 3.2


@dataclass
class HatInputs:
    solar_irradiance_w_m2: float
    ambient_temp_c: float
    sky_temp_c: float
    convective_h_w_m2k: float
    solar_view_factor: float = 0.68
    baseline_solar_absorptance: float = 0.80
    baseline_ir_emissivity: float = 0.90
    ant_solar_absorptance: float = 0.34
    ant_ir_emissivity: float = 0.95
    cooling_cop: float = 3.2


def c_to_k(v: float) -> float:
    return v + 273.15


def k_to_c(v: float) -> float:
    return v - 273.15


def slugify(value: str) -> str:
    chars = []
    for ch in value.strip().lower():
        if ch.isalnum():
            chars.append(ch)
        elif ch in (" ", "-", "_"):
            chars.append("_")
    out = "".join(chars).strip("_")
    return out or "selected_location"


def parse_time(raw: str) -> tuple[int, int]:
    formats = ["%Y%m%d:%H%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S"]
    for fmt in formats:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.month, dt.hour
        except ValueError:
            pass
    if len(raw) >= 10 and raw[:8].isdigit() and raw[9:11].isdigit():
        return int(raw[4:6]), int(raw[9:11])
    return 1, 0


def pick(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in row:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return default


def fetch_pvgis_tmy(lat: float, lon: float, cache_dir: Path, label_slug: str) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"pvgis_tmy_{label_slug}_{lat:.4f}_{lon:.4f}.json".replace("-", "m")

    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        query = urlencode({"lat": lat, "lon": lon, "outputformat": "json"})
        with urlopen(f"{PVGIS_TMY_URL}?{query}", timeout=60) as resp:  # nosec B310
            payload = json.loads(resp.read().decode("utf-8"))
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = payload.get("outputs", {}).get("tmy_hourly")
    if not rows:
        raise ValueError("PVGIS returned no tmy_hourly data.")
    return rows


def solve_root(fn, lo: float, hi: float) -> float:
    f_lo = fn(lo)
    f_hi = fn(hi)

    # Expand the bracket adaptively for edge climates/conditions.
    if f_lo * f_hi > 0:
        span = hi - lo
        for _ in range(12):
            lo = max(1.0, lo - 0.5 * span)
            hi = hi + 0.5 * span
            span *= 2.0
            f_lo = fn(lo)
            f_hi = fn(hi)
            if f_lo * f_hi <= 0:
                break

    # If still unbracketed, search for a sign change; otherwise return best residual point.
    if f_lo * f_hi > 0:
        grid_n = 400
        best_x = lo
        best_abs = abs(f_lo)
        prev_x = lo
        prev_f = f_lo
        step = (hi - lo) / grid_n

        for idx in range(1, grid_n + 1):
            x = lo + idx * step
            f_x = fn(x)
            abs_fx = abs(f_x)
            if abs_fx < best_abs:
                best_abs = abs_fx
                best_x = x
            if prev_f * f_x <= 0:
                lo = prev_x
                hi = x
                f_lo = prev_f
                f_hi = f_x
                break
            prev_x = x
            prev_f = f_x

        if f_lo * f_hi > 0:
            return best_x

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f_mid = fn(mid)
        if abs(f_mid) < 1e-6:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


def roof_hour_result(i: RoofInputs) -> dict[str, float]:
    ta = c_to_k(i.ambient_temp_c)
    tsky = c_to_k(i.sky_temp_c)

    def balance(ts: float, alpha: float, eps: float) -> float:
        return alpha * i.solar_irradiance_w_m2 + i.convective_h_w_m2k * (ta - ts) + eps * SIGMA * (tsky**4 - ts**4)

    base_k = solve_root(lambda ts: balance(ts, i.baseline_solar_absorptance, i.baseline_ir_emissivity), c_to_k(-20), c_to_k(120))
    ant_k = solve_root(lambda ts: balance(ts, i.ant_solar_absorptance, i.ant_ir_emissivity), c_to_k(-20), c_to_k(120))

    base_c = k_to_c(base_k)
    ant_c = k_to_c(ant_k)
    q_base = max(0.0, i.roof_u_value_w_m2k * (base_c - i.indoor_temp_c))
    q_ant = max(0.0, i.roof_u_value_w_m2k * (ant_c - i.indoor_temp_c))
    q_saved = max(0.0, q_base - q_ant)
    e_saved = (q_saved / i.cooling_cop) / 1000.0

    return {
        "baseline_temp_c": base_c,
        "ant_temp_c": ant_c,
        "temp_drop_c": base_c - ant_c,
        "saved_w_m2": q_saved,
        "saved_kwh_m2_h": e_saved,
    }


def hat_hour_result(i: HatInputs) -> dict[str, float]:
    ta = c_to_k(i.ambient_temp_c)
    tsky = c_to_k(i.sky_temp_c)

    def balance(ts: float, alpha: float, eps: float) -> float:
        solar = alpha * i.solar_view_factor * i.solar_irradiance_w_m2
        return solar + i.convective_h_w_m2k * (ta - ts) + eps * SIGMA * (tsky**4 - ts**4)

    base_k = solve_root(lambda ts: balance(ts, i.baseline_solar_absorptance, i.baseline_ir_emissivity), c_to_k(-10), c_to_k(110))
    ant_k = solve_root(lambda ts: balance(ts, i.ant_solar_absorptance, i.ant_ir_emissivity), c_to_k(-10), c_to_k(110))

    base_c = k_to_c(base_k)
    ant_c = k_to_c(ant_k)
    q_reduction = max(
        0.0,
        (i.baseline_solar_absorptance - i.ant_solar_absorptance) * i.solar_view_factor * i.solar_irradiance_w_m2,
    )
    e_saved = (q_reduction / i.cooling_cop) / 1000.0

    return {
        "baseline_temp_c": base_c,
        "ant_temp_c": ant_c,
        "temp_drop_c": base_c - ant_c,
        "saved_w_m2": q_reduction,
        "saved_kwh_m2_h": e_saved,
    }


def month_rows(monthly: dict[int, float]) -> str:
    return "\n".join(f"<tr><td>{m:02d}</td><td>{monthly.get(m, 0.0):.3f}</td></tr>" for m in range(1, 13))


def hour_rows(hourly: list[dict[str, float]]) -> str:
    return "\n".join(
        f"<tr><td>{r['hour']:02d}:00</td><td>{r['avg_temp_drop_c']:.2f}</td><td>{r['avg_kwh_m2_h']:.4f}</td></tr>" for r in hourly
    )


def build_hourly_analysis_index(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    roof_dir = output_dir / "Roof analysis"
    hat_dir = output_dir / "Hat analysis"
    roof_dir.mkdir(parents=True, exist_ok=True)
    hat_dir.mkdir(parents=True, exist_ok=True)

    csv_cards: list[str] = []
    for path in sorted(output_dir.glob("*.csv"), key=lambda item: item.name.lower()):
        csv_cards.append(
            f'      <a class="card" href="{path.name}"><div class="name">{path.name}</div><p class="desc">Hourly CSV data</p></a>'
        )

    sections = [
        ("Roof analysis", "Roof hourly HTML reports", "Roof hourly reports grouped in one folder."),
        ("Hat analysis", "Hat hourly HTML reports", "Hat hourly reports grouped in one folder."),
    ]

    for folder_name, section_title, section_desc in sections:
        section_dir = output_dir / folder_name
        report_cards: list[str] = []
        for path in sorted(section_dir.glob("*.html"), key=lambda item: item.name.lower()):
            if path.name == "index.html":
                continue
            report_cards.append(
                f'      <a class="card" href="{path.name}"><div class="name">{path.name}</div><p class="desc">Hourly HTML report</p></a>'
            )
        section_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{section_title}</title>
  <style>
    :root {{
      --bg: #f2f5f4;
      --card: #ffffff;
      --ink: #1d2a2a;
      --muted: #5b6b6b;
      --accent: #0b7285;
      --border: #d7e1dc;
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: linear-gradient(180deg, #eef6f4 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: "Segoe UI", Tahoma, sans-serif;
    }}
    .wrap {{ max-width: 920px; margin: 0 auto; }}
    .nav {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 18px; padding: 12px; background: var(--card); border: 1px solid var(--border); border-radius: 12px; }}
    .nav a {{ color: var(--accent); text-decoration: none; font-weight: 600; }}
    .nav a:hover {{ text-decoration: underline; }}
    .title {{ font-size: 2rem; font-weight: 700; margin: 0 0 8px; }}
    .subtitle {{ margin: 0 0 18px; color: var(--muted); }}
    .list {{ display: grid; gap: 14px; }}
    .card {{ display: block; padding: 16px 18px; background: var(--card); border: 1px solid var(--border); border-radius: 14px; color: inherit; text-decoration: none; }}
    .card:hover {{ border-color: var(--accent); }}
    .name {{ color: var(--accent); font-size: 1.05rem; font-weight: 700; margin-bottom: 6px; }}
    .desc {{ color: var(--muted); margin: 0; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="nav"><a href="../index.html">Hourly Analysis</a><a href="../../index.html">Reports Home</a></div>
    <p class="title">{section_title}</p>
    <p class="subtitle">{section_desc}</p>
    <div class="list">
{chr(10).join(report_cards) if report_cards else '      <div class="card"><div class="name">No reports yet</div><p class="desc">Run an hourly analysis to populate this folder.</p></div>'}
    </div>
  </div>
</body>
</html>
"""
        (section_dir / "index.html").write_text(section_html, encoding="utf-8")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hourly Analysis Reports</title>
  <style>
    :root {{
      --bg: #f2f5f4;
      --card: #ffffff;
      --ink: #1d2a2a;
      --muted: #5b6b6b;
      --accent: #0b7285;
      --border: #d7e1dc;
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: linear-gradient(180deg, #eef6f4 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: "Segoe UI", Tahoma, sans-serif;
    }}
    .wrap {{ max-width: 920px; margin: 0 auto; }}
    .nav {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 18px; padding: 12px; background: var(--card); border: 1px solid var(--border); border-radius: 12px; }}
    .nav a {{ color: var(--accent); text-decoration: none; font-weight: 600; }}
    .nav a:hover {{ text-decoration: underline; }}
    .title {{ font-size: 2rem; font-weight: 700; margin: 0 0 8px; }}
    .subtitle {{ margin: 0 0 18px; color: var(--muted); }}
    .list {{ display: grid; gap: 14px; }}
    .card {{ display: block; padding: 16px 18px; background: var(--card); border: 1px solid var(--border); border-radius: 14px; color: inherit; text-decoration: none; }}
    .card:hover {{ border-color: var(--accent); }}
    .name {{ color: var(--accent); font-size: 1.05rem; font-weight: 700; margin-bottom: 6px; }}
    .desc {{ color: var(--muted); margin: 0; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="nav"><a href="../index.html">Reports Home</a><a href="../interactive_roof_report.html">Interactive Roof Report</a></div>
    <p class="title">Hourly Analysis</p>
    <p class="subtitle">Hourly roof and hat PVGIS outputs with separate HTML report folders.</p>
    <div class="list">
      <a class="card" href="Roof analysis/index.html"><div class="name">Roof analysis</div><p class="desc">Roof hourly HTML reports</p></a>
      <a class="card" href="Hat analysis/index.html"><div class="name">Hat analysis</div><p class="desc">Hat hourly HTML reports</p></a>
{chr(10).join(csv_cards) if csv_cards else '      <div class="card"><div class="name">No CSV outputs yet</div><p class="desc">Run an hourly analysis to generate hourly CSV files.</p></div>'}
    </div>
  </div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def render_report(title: str, subtitle: str, eq1: str, eq2: str, summary: dict[str, Any], out_path: Path) -> None:
    temp_cards = ""
    if "avg_baseline_roof_temp_c" in summary and "avg_ant_roof_temp_c" in summary:
        temp_cards = f"""
      <div class=\"card\"><div class=\"label\">Average baseline roof surface temperature</div><div class=\"value\">{summary['avg_baseline_roof_temp_c']:.2f} °C</div></div>
      <div class=\"card\"><div class=\"label\">Average ant-hair inspired roof surface temperature</div><div class=\"value\">{summary['avg_ant_roof_temp_c']:.2f} °C</div></div>"""
    elif "avg_baseline_shell_temp_c" in summary and "avg_ant_shell_temp_c" in summary:
        temp_cards = f"""
      <div class=\"card\"><div class=\"label\">Average baseline hat surface temperature</div><div class=\"value\">{summary['avg_baseline_shell_temp_c']:.2f} °C</div></div>
      <div class=\"card\"><div class=\"label\">Average ant-hair inspired hat surface temperature</div><div class=\"value\">{summary['avg_ant_shell_temp_c']:.2f} °C</div></div>"""

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script defer src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js\"></script>
  <style>
    body {{ font-family: Segoe UI, Tahoma, sans-serif; margin: 0; padding: 24px; background: #f5f7f6; color: #1f2a2a; }}
    .wrap {{ max-width: 980px; margin: 0 auto; }}
    .nav {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 18px; padding: 12px; background: white; border: 1px solid #dbe3df; border-radius: 12px; }}
    .nav a {{ color: #0b7285; text-decoration: none; font-weight: 600; }}
    .nav a:hover {{ text-decoration: underline; }}
    .title {{ font-size: 1.7rem; font-weight: 700; margin-bottom: 4px; }}
    .subtitle {{ margin-top: 0; color: #556; }}
    .grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); }}
    .card {{ background: white; border: 1px solid #dbe3df; border-radius: 12px; padding: 12px; }}
    .label {{ color: #5c6c6b; font-size: .9rem; }}
    .value {{ font-size: 1.35rem; font-weight: 700; margin-top: 4px; }}
    details {{ margin-top: 12px; background: white; border: 1px solid #dbe3df; border-radius: 10px; padding: 12px; }}
    summary {{ cursor: pointer; font-weight: 600; color: #0b7285; }}
    .eq {{ background: #eef4f2; border-radius: 8px; padding: 10px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border-bottom: 1px solid #e6ece9; text-align: left; padding: 6px; font-size: .92rem; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"nav\"><a href=\"index.html\">Section Index</a><a href=\"../index.html\">Hourly Analysis</a></div>
    <div class=\"title\">{title}</div>
    <p class=\"subtitle\">{subtitle}</p>

    <div class=\"grid\">
      {temp_cards}
      <div class=\"card\"><div class=\"label\">Average Temperature Drop</div><div class=\"value\">{summary['avg_temp_drop_c']:.2f} °C</div></div>
      <div class=\"card\"><div class=\"label\">Annual Savings</div><div class=\"value\">{summary['annual_kwh_m2']:.2f} kWh/m²/year</div></div>
    </div>

    <details open>
      <summary>Equations Used</summary>
      <div class=\"eq\">{eq1}</div>
      <div class=\"eq\">{eq2}</div>
    </details>

    <details>
      <summary>Monthly Savings (kWh/m²)</summary>
      <table><thead><tr><th>Month</th><th>kWh/m²</th></tr></thead><tbody>{month_rows(summary['monthly_kwh'])}</tbody></table>
    </details>

    <details>
      <summary>Average Benefit By Hour Of Day</summary>
      <table><thead><tr><th>Hour</th><th>Avg Temp Drop (°C)</th><th>Avg Savings (kWh/m²/h)</th></tr></thead><tbody>{hour_rows(summary['hourly_avg'])}</tbody></table>
    </details>
  </div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def run_location_analysis(lat: float, lon: float, label: str) -> dict[str, str]:
    label = label.strip() or f"{lat:.4f}, {lon:.4f}"
    slug = slugify(label)

    roof_rows = []
    hat_rows = []
    roof_monthly = defaultdict(float)
    hat_monthly = defaultdict(float)
    roof_hour_bins_t = defaultdict(list)
    roof_hour_bins_e = defaultdict(list)
    hat_hour_bins_t = defaultdict(list)
    hat_hour_bins_e = defaultdict(list)
    skipped_hours = 0

    pvgis_rows = fetch_pvgis_tmy(lat, lon, ROOF_DIR / "pvgis_cache", slug)

    for row in pvgis_rows:
        raw_time = str(row.get("time(UTC)") or row.get("time") or row.get("time_utc") or "20010101:0000")
        month, hour = parse_time(raw_time)
        ghi = max(0.0, pick(row, "G(h)", "Gh", "GHI"))
        ambient_c = pick(row, "T2m", "T2m(C)")
        ws10 = max(0.0, pick(row, "WS10m", "WS10m(m/s)", "WS10m (m/s)", default=1.0))

        roof_i = RoofInputs(
            solar_irradiance_w_m2=ghi,
            ambient_temp_c=ambient_c,
            sky_temp_c=ambient_c - 12.0,
            convective_h_w_m2k=5.7 + 3.8 * ws10,
        )
        hat_i = HatInputs(
            solar_irradiance_w_m2=ghi,
            ambient_temp_c=ambient_c,
            sky_temp_c=ambient_c - 12.0,
            convective_h_w_m2k=6.0 + 4.0 * ws10,
        )

        try:
            roof_r = roof_hour_result(roof_i)
            hat_r = hat_hour_result(hat_i)
        except Exception:
            skipped_hours += 1
            continue

        roof_rows.append(
            {
                "location": label,
                "lat": lat,
                "lon": lon,
                "time_utc": raw_time,
                "month": month,
                "hour": hour,
                "ghi_w_m2": ghi,
                "ambient_c": ambient_c,
                "wind_m_s": ws10,
                "baseline_roof_temp_c": roof_r["baseline_temp_c"],
                "ant_roof_temp_c": roof_r["ant_temp_c"],
                "temp_drop_c": roof_r["temp_drop_c"],
                "cooling_saved_w_m2": roof_r["saved_w_m2"],
                "electric_saved_kwh_m2_hour": roof_r["saved_kwh_m2_h"],
            }
        )
        hat_rows.append(
            {
                "location": label,
                "lat": lat,
                "lon": lon,
                "time_utc": raw_time,
                "month": month,
                "hour": hour,
                "ghi_w_m2": ghi,
                "ambient_c": ambient_c,
                "wind_m_s": ws10,
                "baseline_shell_temp_c": hat_r["baseline_temp_c"],
                "ant_shell_temp_c": hat_r["ant_temp_c"],
                "temp_drop_c": hat_r["temp_drop_c"],
                "equiv_heat_reduction_w_m2": hat_r["saved_w_m2"],
                "equiv_electric_saved_kwh_m2_hour": hat_r["saved_kwh_m2_h"],
            }
        )

        roof_monthly[month] += roof_r["saved_kwh_m2_h"]
        hat_monthly[month] += hat_r["saved_kwh_m2_h"]
        roof_hour_bins_t[hour].append(roof_r["temp_drop_c"])
        roof_hour_bins_e[hour].append(roof_r["saved_kwh_m2_h"])
        hat_hour_bins_t[hour].append(hat_r["temp_drop_c"])
        hat_hour_bins_e[hour].append(hat_r["saved_kwh_m2_h"])

    if not roof_rows or not hat_rows:
        raise ValueError("No valid hourly results were generated for this location.")

    def build_summary(
        rows: list[dict[str, Any]],
        monthly: dict[int, float],
        bins_t: dict[int, list[float]],
        bins_e: dict[int, list[float]],
        key: str,
        baseline_temp_key: str | None = None,
        ant_temp_key: str | None = None,
    ) -> dict[str, Any]:
        hourly_avg = []
        for h in range(24):
            tv = bins_t.get(h, [0.0])
            ev = bins_e.get(h, [0.0])
            hourly_avg.append({"hour": h, "avg_temp_drop_c": sum(tv) / len(tv), "avg_kwh_m2_h": sum(ev) / len(ev)})
        summary = {
            "annual_kwh_m2": sum(r[key] for r in rows),
            "avg_temp_drop_c": sum(r["temp_drop_c"] for r in rows) / max(1, len(rows)),
            "monthly_kwh": dict(sorted(monthly.items())),
            "hourly_avg": hourly_avg,
        }
        if baseline_temp_key and ant_temp_key:
            summary["avg_baseline_roof_temp_c"] = sum(r[baseline_temp_key] for r in rows) / max(1, len(rows))
            summary["avg_ant_roof_temp_c"] = sum(r[ant_temp_key] for r in rows) / max(1, len(rows))
        return summary

    roof_summary = build_summary(
        roof_rows,
        roof_monthly,
        roof_hour_bins_t,
        roof_hour_bins_e,
        "electric_saved_kwh_m2_hour",
        baseline_temp_key="baseline_roof_temp_c",
        ant_temp_key="ant_roof_temp_c",
    )
    hat_summary = build_summary(hat_rows, hat_monthly, hat_hour_bins_t, hat_hour_bins_e, "equiv_electric_saved_kwh_m2_hour")
    hat_summary["avg_baseline_shell_temp_c"] = sum(r["baseline_shell_temp_c"] for r in hat_rows) / max(1, len(hat_rows))
    hat_summary["avg_ant_shell_temp_c"] = sum(r["ant_shell_temp_c"] for r in hat_rows) / max(1, len(hat_rows))

    HOURLY_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    roof_csv = HOURLY_ANALYSIS_DIR / f"roof_hourly_{slug}.csv"
    hat_csv = HOURLY_ANALYSIS_DIR / f"hat_hourly_{slug}.csv"

    with roof_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(roof_rows[0].keys()))
        w.writeheader()
        w.writerows(roof_rows)

    with hat_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hat_rows[0].keys()))
        w.writeheader()
        w.writerows(hat_rows)

    roof_html = HOURLY_ROOF_HTML_DIR / f"ant_roof_pvgis_hourly_{slug}.html"
    hat_html = HOURLY_HAT_HTML_DIR / f"hat_pvgis_hourly_{slug}.html"

    render_report(
        title=f"Roof Hourly Benefit Analysis - {label}",
        subtitle=f"PVGIS TMY at lat {lat:.4f}, lon {lon:.4f}",
        eq1=r"\[\alpha G + h_{conv}\,(T_a - T_s) + \varepsilon\sigma\,(T_{sky}^{4} - T_{s}^{4}) = 0\]",
        eq2=r"\[q_{in}=\max(0,U(T_s-T_{in})),\; q_{saved}=q_{in,base}-q_{in,ant},\; E_{saved}=\frac{q_{saved}}{COP\cdot 1000}\]",
        summary=roof_summary,
        out_path=roof_html,
    )
    render_report(
        title=f"Hat Hourly Benefit Analysis - {label}",
        subtitle=f"PVGIS TMY at lat {lat:.4f}, lon {lon:.4f}",
        eq1=r"\[\alpha f_{view} G + h_{conv}\,(T_a - T_s) + \varepsilon\sigma\,(T_{sky}^{4} - T_{s}^{4}) = 0\]",
        eq2=r"\[q_{reduction}=(\alpha_{base}-\alpha_{ant})\,f_{view}\,G,\; E_{equiv}=\frac{q_{reduction}}{COP\cdot 1000}\]",
        summary=hat_summary,
        out_path=hat_html,
    )
    build_hourly_analysis_index(HOURLY_ANALYSIS_DIR)

    return {
        "roof_html": str(roof_html),
        "hat_html": str(hat_html),
        "roof_csv": str(roof_csv),
        "hat_csv": str(hat_csv),
        "roof_annual_kwh_m2": f"{roof_summary['annual_kwh_m2']:.2f}",
        "roof_avg_drop_c": f"{roof_summary['avg_temp_drop_c']:.2f}",
        "hat_annual_kwh_m2": f"{hat_summary['annual_kwh_m2']:.2f}",
        "hat_avg_drop_c": f"{hat_summary['avg_temp_drop_c']:.2f}",
        "skipped_hours": str(skipped_hours),
    }

class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/":
            self.send_error(404)
            return

        html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Interactive PVGIS Location Selector</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin=""/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
<style>
body { margin: 0; font-family: Segoe UI, Tahoma, sans-serif; background: #f5f7f6; color: #1f2a2a; }
.wrap { display: grid; grid-template-columns: 1fr 320px; min-height: 100vh; }
#map { height: 100vh; }
.panel { padding: 16px; border-left: 1px solid #d9e2de; background: #fff; }
label { font-size: .9rem; color: #4e5f5d; }
input { width: 100%; margin: 6px 0 12px; padding: 8px; border: 1px solid #cfd9d4; border-radius: 8px; }
button { width: 100%; padding: 10px; border: 0; border-radius: 8px; background: #0b7285; color: #fff; font-weight: 600; cursor: pointer; }
pre { white-space: pre-wrap; background: #eef4f2; border-radius: 8px; padding: 10px; font-size: .85rem; }
.small { color: #60706f; font-size: .85rem; }
</style>
</head>
<body>
<div class="wrap">
  <div id="map"></div>
  <div class="panel">
    <h2>Select A Location</h2>
    <p class="small">Click on the map to choose coordinates, then run roof + hat hourly analysis.</p>
    <label>Latitude</label>
    <input id="lat" type="number" step="any" />
    <label>Longitude</label>
    <input id="lon" type="number" step="any" />
    <label>Location Name (optional)</label>
    <input id="name" type="text" placeholder="e.g., Barcelona Beach" />
    <button id="runBtn">Run Analysis</button>
    <h3>Result</h3>
    <pre id="out">Waiting for input.</pre>
  </div>
</div>
<script>
const map = L.map('map').setView([30, 10], 3);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap' }).addTo(map);
let marker = null;
map.on('click', (e) => {
  const {lat, lng} = e.latlng;
  document.getElementById('lat').value = lat.toFixed(6);
  document.getElementById('lon').value = lng.toFixed(6);
  if (marker) marker.setLatLng(e.latlng); else marker = L.marker(e.latlng).addTo(map);
});

document.getElementById('runBtn').addEventListener('click', async () => {
  const lat = parseFloat(document.getElementById('lat').value);
  const lon = parseFloat(document.getElementById('lon').value);
  const name = document.getElementById('name').value.trim();
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    document.getElementById('out').textContent = 'Please select a valid lat/lon on the map.';
    return;
  }
  document.getElementById('out').textContent = 'Running analysis...';
  try {
    const resp = await fetch('/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({lat, lon, name})
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'Analysis failed.');
    document.getElementById('out').textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    document.getElementById('out').textContent = 'Error: ' + err.message;
  }
});
</script>
</body>
</html>
"""
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/run":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            lat = float(payload["lat"])
            lon = float(payload["lon"])
            name = str(payload.get("name", "")).strip()
        except Exception:
            self._send_json({"error": "Invalid payload."}, status=400)
            return

        try:
            result = run_location_analysis(lat, lon, name)
            self._send_json(result)
        except Exception as exc:  # pragma: no cover
            self._send_json({"error": str(exc)}, status=500)


def main() -> None:
    host = "127.0.0.1"
    port = 8765
    httpd = HTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"Map selector running at {url}")
    print("Click a location, then run analysis from the browser panel.")

    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    httpd.serve_forever()


if __name__ == "__main__":
    main()








