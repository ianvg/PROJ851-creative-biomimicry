from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

SIGMA = 5.670374419e-8  # W/m^2/K^4
PVGIS_TMY_URL = "https://re.jrc.ec.europa.eu/api/tmy"
CITY_COORDS = {
    "marseille": (43.2965, 5.3698),
    "cairo": (30.0444, 31.2357),
}


@dataclass
class HatInputs:
    solar_irradiance_w_m2: float = 1000.0
    ambient_temp_c: float = 35.0
    sky_temp_c: float = 20.0
    convective_h_w_m2k: float = 12.0

    solar_view_factor: float = 0.68

    baseline_solar_absorptance: float = 0.80
    baseline_ir_emissivity: float = 0.90

    ant_solar_absorptance: float = 0.34
    ant_ir_emissivity: float = 0.95

    cooling_cop: float = 3.2


@dataclass
class HatResults:
    baseline_shell_temp_c: float
    ant_shell_temp_c: float
    shell_temp_drop_c: float
    equivalent_heat_reduction_w_m2: float
    equivalent_electric_savings_kwh_m2_hour: float


def c_to_k(celsius: float) -> float:
    return celsius + 273.15


def k_to_c(kelvin: float) -> float:
    return kelvin - 273.15


def energy_balance(ts_k: float, alpha: float, eps: float, i: HatInputs) -> float:
    ta_k = c_to_k(i.ambient_temp_c)
    tsky_k = c_to_k(i.sky_temp_c)

    absorbed_solar = alpha * i.solar_view_factor * i.solar_irradiance_w_m2
    convection = i.convective_h_w_m2k * (ta_k - ts_k)
    thermal_radiation = eps * SIGMA * (tsky_k**4 - ts_k**4)

    return absorbed_solar + convection + thermal_radiation


def solve_temp(alpha: float, eps: float, i: HatInputs) -> float:
    low_k = c_to_k(-10.0)
    high_k = c_to_k(110.0)

    f_low = energy_balance(low_k, alpha, eps, i)
    f_high = energy_balance(high_k, alpha, eps, i)

    if f_low * f_high > 0:
        raise ValueError("Unable to bracket temperature root.")

    for _ in range(100):
        mid = 0.5 * (low_k + high_k)
        f_mid = energy_balance(mid, alpha, eps, i)

        if abs(f_mid) < 1e-6:
            return mid

        if f_low * f_mid < 0:
            high_k = mid
        else:
            low_k = mid
            f_low = f_mid

    return 0.5 * (low_k + high_k)


def analyze(i: HatInputs) -> HatResults:
    base_k = solve_temp(i.baseline_solar_absorptance, i.baseline_ir_emissivity, i)
    ant_k = solve_temp(i.ant_solar_absorptance, i.ant_ir_emissivity, i)

    base_c = k_to_c(base_k)
    ant_c = k_to_c(ant_k)
    drop_c = base_c - ant_c

    q_solar_base = i.baseline_solar_absorptance * i.solar_view_factor * i.solar_irradiance_w_m2
    q_solar_ant = i.ant_solar_absorptance * i.solar_view_factor * i.solar_irradiance_w_m2
    q_reduction = max(0.0, q_solar_base - q_solar_ant)

    e_hour = (q_reduction / i.cooling_cop) / 1000.0

    return HatResults(
        baseline_shell_temp_c=base_c,
        ant_shell_temp_c=ant_c,
        shell_temp_drop_c=drop_c,
        equivalent_heat_reduction_w_m2=q_reduction,
        equivalent_electric_savings_kwh_m2_hour=e_hour,
    )


def _pick(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in row:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return default


def _parse_time(raw: str) -> tuple[int, int]:
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


def fetch_pvgis_tmy(city: str, cache_dir: Path) -> list[dict[str, Any]]:
    city_key = city.strip().lower()
    if city_key not in CITY_COORDS:
        raise ValueError(f"Unsupported city '{city}'. Use: {', '.join(CITY_COORDS)}")

    lat, lon = CITY_COORDS[city_key]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"pvgis_tmy_{city_key}.json"

    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        query = urlencode({"lat": lat, "lon": lon, "outputformat": "json"})
        with urlopen(f"{PVGIS_TMY_URL}?{query}", timeout=60) as resp:  # nosec B310
            payload = json.loads(resp.read().decode("utf-8"))
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = payload.get("outputs", {}).get("tmy_hourly")
    if not rows:
        raise ValueError(f"No tmy_hourly data returned for {city}.")
    return rows


def hourly_analysis_for_city(city: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = fetch_pvgis_tmy(city, Path(args.pvgis_cache_dir))

    hourly_out: list[dict[str, Any]] = []
    monthly_kwh = defaultdict(float)
    hour_bins_temp = defaultdict(list)
    hour_bins_kwh = defaultdict(list)

    for row in rows:
        raw_time = str(row.get("time(UTC)") or row.get("time") or row.get("time_utc") or "20010101:0000")
        month, hour = _parse_time(raw_time)

        ghi = max(0.0, _pick(row, "G(h)", "Gh", "GHI"))
        ambient_c = _pick(row, "T2m", "T2m(C)")
        ws10 = max(0.0, _pick(row, "WS10m", "WS10m(m/s)", "WS10m (m/s)", default=1.0))

        # Higher convection coefficient than roof due open-air, exposed geometry.
        h_dynamic = 6.0 + 4.0 * ws10

        inputs = HatInputs(
            solar_irradiance_w_m2=ghi,
            ambient_temp_c=ambient_c,
            sky_temp_c=ambient_c - args.sky_offset_c,
            convective_h_w_m2k=h_dynamic,
            solar_view_factor=args.view_factor,
            baseline_solar_absorptance=args.alpha_baseline,
            baseline_ir_emissivity=args.eps_baseline,
            ant_solar_absorptance=args.alpha_ant,
            ant_ir_emissivity=args.eps_ant,
            cooling_cop=args.cop,
        )

        result = analyze(inputs)
        kwh_hour = result.equivalent_electric_savings_kwh_m2_hour

        hourly_out.append(
            {
                "city": city,
                "time_utc": raw_time,
                "month": month,
                "hour": hour,
                "ghi_w_m2": ghi,
                "ambient_c": ambient_c,
                "wind_m_s": ws10,
                "baseline_shell_temp_c": result.baseline_shell_temp_c,
                "ant_shell_temp_c": result.ant_shell_temp_c,
                "temp_drop_c": result.shell_temp_drop_c,
                "equiv_heat_reduction_w_m2": result.equivalent_heat_reduction_w_m2,
                "equiv_electric_saved_kwh_m2_hour": kwh_hour,
            }
        )

        monthly_kwh[month] += kwh_hour
        hour_bins_temp[hour].append(result.shell_temp_drop_c)
        hour_bins_kwh[hour].append(kwh_hour)

    annual_kwh = sum(r["equiv_electric_saved_kwh_m2_hour"] for r in hourly_out)
    avg_drop = sum(r["temp_drop_c"] for r in hourly_out) / len(hourly_out)

    hourly_avg = []
    for hour in range(24):
        t_vals = hour_bins_temp.get(hour, [0.0])
        e_vals = hour_bins_kwh.get(hour, [0.0])
        hourly_avg.append(
            {
                "hour": hour,
                "avg_temp_drop_c": sum(t_vals) / len(t_vals),
                "avg_kwh_m2_hour": sum(e_vals) / len(e_vals),
            }
        )

    summary = {
        "city": city,
        "annual_kwh_m2": annual_kwh,
        "monthly_kwh_m2": dict(sorted(monthly_kwh.items())),
        "avg_temp_drop_c": avg_drop,
        "hourly_avg": hourly_avg,
    }
    return hourly_out, summary


def _monthly_rows(monthly: dict[int, float]) -> str:
    rows = []
    for m in range(1, 13):
        rows.append(f"<tr><td>{m:02d}</td><td>{monthly.get(m, 0.0):.3f}</td></tr>")
    return "\n".join(rows)


def _hourly_rows(hourly_avg: list[dict[str, float]]) -> str:
    rows = []
    for r in hourly_avg:
        rows.append(
            f"<tr><td>{r['hour']:02d}:00</td><td>{r['avg_temp_drop_c']:.2f}</td><td>{r['avg_kwh_m2_hour']:.4f}</td></tr>"
        )
    return "\n".join(rows)


def render_hourly_html(summaries: list[dict[str, Any]], output_path: Path) -> None:
    eq1 = r"\[\alpha f_{view} G + h\,(T_a - T_s) + \varepsilon\sigma\,(T_{sky}^{4} - T_{s}^{4}) = 0\]"
    eq2 = r"\[q_{reduction}=(\alpha_{base}-\alpha_{ant})\,f_{view}\,G,\; E_{equiv}=\frac{q_{reduction}}{COP\cdot 1000}\]"

    city_blocks = []
    for s in summaries:
        city_blocks.append(
            f"""
      <details>
        <summary>{s['city'].title()} - Hourly And Monthly Benefits</summary>
        <div class=\"grid\" style=\"margin-top:10px;\">
          <div class=\"card\">
            <div class=\"label\">Average shell temperature drop (all TMY hours)</div>
            <div class=\"value ok\">{s['avg_temp_drop_c']:.2f} °C</div>
          </div>
          <div class=\"card\">
            <div class=\"label\">Annual equivalent savings</div>
            <div class=\"value ok\">{s['annual_kwh_m2']:.2f} kWh/m²/year</div>
          </div>
        </div>
        <h3>Monthly Equivalent Savings (kWh/m²)</h3>
        <table>
          <thead><tr><th>Month</th><th>kWh/m²</th></tr></thead>
          <tbody>{_monthly_rows(s['monthly_kwh_m2'])}</tbody>
        </table>
        <h3>Average Benefit By Hour Of Day</h3>
        <table>
          <thead><tr><th>Hour</th><th>Avg Temp Drop (°C)</th><th>Avg Savings (kWh/m²/h)</th></tr></thead>
          <tbody>{_hourly_rows(s['hourly_avg'])}</tbody>
        </table>
      </details>
"""
        )

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Hat PVGIS Hourly Analysis</title>
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']] }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script defer src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js\"></script>
  <style>
    body {{ font-family: Segoe UI, Tahoma, sans-serif; margin: 0; padding: 24px; background: #f6f2e8; color: #2d2417; }}
    .wrap {{ max-width: 1000px; margin: 0 auto; }}
    .title {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 8px; }}
    .subtitle {{ color: #6a5c46; margin-top: 0; }}
    .grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
    .card {{ background: #fffdfa; border: 1px solid #e9dcc5; border-radius: 12px; padding: 12px; }}
    .label {{ color: #6a5c46; font-size: 0.88rem; }}
    .value {{ font-size: 1.45rem; font-weight: 700; margin-top: 4px; }}
    .ok {{ color: #2b8a3e; }}
    details {{ margin-top: 14px; background: #fffdfa; border: 1px solid #e9dcc5; border-radius: 10px; padding: 12px; }}
    summary {{ cursor: pointer; color: #a35f14; font-weight: 600; }}
    .eq {{ background: #f9f4ea; border-radius: 8px; padding: 10px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border-bottom: 1px solid #efe3d1; text-align: left; padding: 6px; font-size: 0.92rem; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"title\">Round Hat Hourly Benefit Analysis From PVGIS TMY</div>
    <p class=\"subtitle\">Cities: Marseille and Cairo. Hourly simulation uses PVGIS TMY solar irradiance, air temperature, and wind.</p>

    <details open>
      <summary>Equations Used</summary>
      <div class=\"eq\">{eq1}</div>
      <div class=\"eq\">{eq2}</div>
    </details>

    {''.join(city_blocks)}
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def write_hourly_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Round hat analysis with optional PVGIS TMY hourly mode.")
    p.add_argument("--use-pvgis", action="store_true", help="Fetch PVGIS TMY and run hourly analysis.")
    p.add_argument("--cities", default="marseille,cairo", help="Comma-separated city names (marseille,cairo).")
    p.add_argument("--pvgis-cache-dir", default="pvgis_cache")
    p.add_argument("--sky-offset-c", type=float, default=12.0, help="Sky temperature estimate offset from ambient.")

    p.add_argument("--solar", type=float, default=1000.0)
    p.add_argument("--ambient-c", type=float, default=35.0)
    p.add_argument("--sky-c", type=float, default=20.0)
    p.add_argument("--h", type=float, default=12.0)

    p.add_argument("--view-factor", type=float, default=0.68)
    p.add_argument("--alpha-baseline", type=float, default=0.80)
    p.add_argument("--eps-baseline", type=float, default=0.90)
    p.add_argument("--alpha-ant", type=float, default=0.34)
    p.add_argument("--eps-ant", type=float, default=0.95)

    p.add_argument("--cop", type=float, default=3.2)
    p.add_argument("--html", type=Path, default=None, help="Output HTML path")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.use_pvgis:
        cities = [c.strip().lower() for c in args.cities.split(",") if c.strip()]
        summaries: list[dict[str, Any]] = []
        html_path = args.html or Path("hat_pvgis_hourly_report.html")

        for city in cities:
            hourly_rows, summary = hourly_analysis_for_city(city, args)
            summaries.append(summary)
            csv_name = f"hat_hourly_{city}.csv"
            write_hourly_csv(Path(csv_name), hourly_rows)

        render_hourly_html(summaries, html_path)
        print("Hat PVGIS hourly analysis complete")
        for s in summaries:
            print(f"{s['city'].title()}: avg drop {s['avg_temp_drop_c']:.2f} C, annual {s['annual_kwh_m2']:.2f} kWh/m^2/year")
        print(f"HTML report: {html_path.resolve()}")
        return

    inputs = HatInputs(
        solar_irradiance_w_m2=args.solar,
        ambient_temp_c=args.ambient_c,
        sky_temp_c=args.sky_c,
        convective_h_w_m2k=args.h,
        solar_view_factor=args.view_factor,
        baseline_solar_absorptance=args.alpha_baseline,
        baseline_ir_emissivity=args.eps_baseline,
        ant_solar_absorptance=args.alpha_ant,
        ant_ir_emissivity=args.eps_ant,
        cooling_cop=args.cop,
    )

    results = analyze(inputs)
    monthly = results.equivalent_electric_savings_kwh_m2_hour * 6.0 * 30.0
    yearly = monthly * 12.0

    print("Round hat thermal analysis")
    print(f"Baseline shell temperature: {results.baseline_shell_temp_c:.2f} C")
    print(f"Ant-inspired shell temperature: {results.ant_shell_temp_c:.2f} C")
    print(f"Shell temperature drop: {results.shell_temp_drop_c:.2f} C")
    print(f"Equivalent savings: {monthly:.2f} kWh/m^2/month")
    print(f"Equivalent savings: {yearly:.2f} kWh/m^2/year")


if __name__ == "__main__":
    main()

