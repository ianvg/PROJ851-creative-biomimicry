from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

SIGMA = 5.670374419e-8  # W/m2/K4
PVGIS_TMY_URL = "https://re.jrc.ec.europa.eu/api/tmy"
CITY_COORDS = {
    "marseille": (43.2965, 5.3698),
    "cairo": (30.0444, 31.2357),
}


@dataclass
class ModelInputs:
    solar_irradiance_w_m2: float = 900.0
    ambient_temp_c: float = 42.0
    sky_temp_c: float = 15.0
    convective_h_w_m2k: float = 8.0

    baseline_solar_absorptance: float = 0.85
    baseline_ir_emissivity: float = 0.90

    ant_solar_absorptance: float = 0.35
    ant_ir_emissivity: float = 0.95

    indoor_temp_c: float = 24.0
    roof_u_value_w_m2k: float = 1.2
    cooling_cop: float = 3.2


@dataclass
class ModelResults:
    baseline_roof_temp_c: float
    ant_roof_temp_c: float
    roof_temp_drop_c: float
    cooling_power_saved_w_m2: float
    electric_savings_kwh_m2_hour: float


def c_to_k(celsius: float) -> float:
    return celsius + 273.15


def k_to_c(kelvin: float) -> float:
    return kelvin - 273.15


def energy_balance(ts_k: float, solar_absorptance: float, ir_emissivity: float, inputs: ModelInputs) -> float:
    ta_k = c_to_k(inputs.ambient_temp_c)
    tsky_k = c_to_k(inputs.sky_temp_c)

    solar_gain = solar_absorptance * inputs.solar_irradiance_w_m2
    convection = inputs.convective_h_w_m2k * (ta_k - ts_k)
    thermal_radiation = ir_emissivity * SIGMA * (tsky_k**4 - ts_k**4)

    return solar_gain + convection + thermal_radiation


def solve_surface_temp_k(solar_absorptance: float, ir_emissivity: float, inputs: ModelInputs) -> float:
    low_k = c_to_k(-20.0)
    high_k = c_to_k(120.0)

    f_low = energy_balance(low_k, solar_absorptance, ir_emissivity, inputs)
    f_high = energy_balance(high_k, solar_absorptance, ir_emissivity, inputs)

    if f_low == 0:
        return low_k
    if f_high == 0:
        return high_k
    if f_low * f_high > 0:
        raise ValueError("Could not bracket roof temperature solution.")

    for _ in range(100):
        mid_k = 0.5 * (low_k + high_k)
        f_mid = energy_balance(mid_k, solar_absorptance, ir_emissivity, inputs)

        if abs(f_mid) < 1e-6:
            return mid_k

        if f_low * f_mid < 0:
            high_k = mid_k
        else:
            low_k = mid_k
            f_low = f_mid

    return 0.5 * (low_k + high_k)


def compute_results(inputs: ModelInputs) -> ModelResults:
    baseline_ts_k = solve_surface_temp_k(inputs.baseline_solar_absorptance, inputs.baseline_ir_emissivity, inputs)
    ant_ts_k = solve_surface_temp_k(inputs.ant_solar_absorptance, inputs.ant_ir_emissivity, inputs)

    baseline_ts_c = k_to_c(baseline_ts_k)
    ant_ts_c = k_to_c(ant_ts_k)
    temp_drop_c = baseline_ts_c - ant_ts_c

    q_in_baseline = max(0.0, inputs.roof_u_value_w_m2k * (baseline_ts_c - inputs.indoor_temp_c))
    q_in_ant = max(0.0, inputs.roof_u_value_w_m2k * (ant_ts_c - inputs.indoor_temp_c))
    cooling_power_saved = max(0.0, q_in_baseline - q_in_ant)

    electric_savings_kwh_hour = (cooling_power_saved / inputs.cooling_cop) / 1000.0

    return ModelResults(
        baseline_roof_temp_c=baseline_ts_c,
        ant_roof_temp_c=ant_ts_c,
        roof_temp_drop_c=temp_drop_c,
        cooling_power_saved_w_m2=cooling_power_saved,
        electric_savings_kwh_m2_hour=electric_savings_kwh_hour,
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

        # Empirical exterior convection estimate from wind speed.
        h_dynamic = 5.7 + 3.8 * ws10

        inputs = ModelInputs(
            solar_irradiance_w_m2=ghi,
            ambient_temp_c=ambient_c,
            sky_temp_c=ambient_c - args.sky_offset_c,
            convective_h_w_m2k=h_dynamic,
            baseline_solar_absorptance=args.alpha_baseline,
            baseline_ir_emissivity=args.eps_baseline,
            ant_solar_absorptance=args.alpha_ant,
            ant_ir_emissivity=args.eps_ant,
            indoor_temp_c=args.indoor_c,
            roof_u_value_w_m2k=args.u_value,
            cooling_cop=args.cop,
        )

        result = compute_results(inputs)
        kwh_hour = result.electric_savings_kwh_m2_hour

        hourly_out.append(
            {
                "city": city,
                "time_utc": raw_time,
                "month": month,
                "hour": hour,
                "ghi_w_m2": ghi,
                "ambient_c": ambient_c,
                "wind_m_s": ws10,
                "baseline_roof_temp_c": result.baseline_roof_temp_c,
                "ant_roof_temp_c": result.ant_roof_temp_c,
                "temp_drop_c": result.roof_temp_drop_c,
                "cooling_saved_w_m2": result.cooling_power_saved_w_m2,
                "electric_saved_kwh_m2_hour": kwh_hour,
            }
        )

        monthly_kwh[month] += kwh_hour
        hour_bins_temp[hour].append(result.roof_temp_drop_c)
        hour_bins_kwh[hour].append(kwh_hour)

    annual_kwh = sum(r["electric_saved_kwh_m2_hour"] for r in hourly_out)
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
    eq1 = r"\[\alpha G + h\,(T_a - T_s) + \varepsilon\sigma\,(T_{sky}^{4} - T_{s}^{4}) = 0\]"
    eq2 = r"\[q_{in}=\max(0,U(T_s-T_{in})),\; q_{saved}=q_{in,base}-q_{in,ant},\; E_{saved}=\frac{q_{saved}}{COP\cdot 1000}\]"

    max_annual = max((s["annual_kwh_m2"] for s in summaries), default=1.0) or 1.0
    max_drop = max((s["avg_temp_drop_c"] for s in summaries), default=1.0) or 1.0

    compare_rows = []
    annual_bars = []
    drop_bars = []
    for s in summaries:
        city = s["city"].title()
        annual = s["annual_kwh_m2"]
        drop = s["avg_temp_drop_c"]
        annual_w = (annual / max_annual) * 100.0
        drop_w = (drop / max_drop) * 100.0
        compare_rows.append(f"<tr><td>{city}</td><td>{annual:.2f}</td><td>{drop:.2f}</td></tr>")
        annual_bars.append(
            f"<div class=\"bar-row\"><span>{city}</span><div class=\"bar-track\"><div class=\"bar-fill\" style=\"width:{annual_w:.1f}%\"></div></div><strong>{annual:.2f}</strong></div>"
        )
        drop_bars.append(
            f"<div class=\"bar-row\"><span>{city}</span><div class=\"bar-track\"><div class=\"bar-fill-alt\" style=\"width:{drop_w:.1f}%\"></div></div><strong>{drop:.2f} °C</strong></div>"
        )

    city_blocks = []
    for s in summaries:
        city_blocks.append(
            f"""
      <details>
        <summary>{s['city'].title()} - Hourly And Monthly Benefits</summary>
        <div class=\"grid\" style=\"margin-top:10px;\">
          <div class=\"card\">
            <div class=\"label\">Average roof temperature drop (all TMY hours)</div>
            <div class=\"value ok\">{s['avg_temp_drop_c']:.2f} °C</div>
          </div>
          <div class=\"card\">
            <div class=\"label\">Annual electricity savings</div>
            <div class=\"value ok\">{s['annual_kwh_m2']:.2f} kWh/m²/year</div>
          </div>
        </div>
        <h3>Monthly Savings (kWh/m²)</h3>
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
  <title>Roof PVGIS Hourly Analysis</title>
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script defer src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js\"></script>
  <style>
    body {{ font-family: Segoe UI, Tahoma, sans-serif; margin: 0; padding: 24px; background: #f4f7f5; color: #1d2a2a; }}
    .wrap {{ max-width: 1000px; margin: 0 auto; }}
    .title {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 8px; }}
    .subtitle {{ color: #5b6b6b; margin-top: 0; }}
    .grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
    .card {{ background: #fff; border: 1px solid #d7e1dc; border-radius: 12px; padding: 12px; }}
    .label {{ color: #5b6b6b; font-size: 0.88rem; }}
    .value {{ font-size: 1.45rem; font-weight: 700; margin-top: 4px; }}
    .ok {{ color: #2b8a3e; }}
    details {{ margin-top: 14px; background: #fff; border: 1px solid #d7e1dc; border-radius: 10px; padding: 12px; }}
    summary {{ cursor: pointer; color: #0b7285; font-weight: 600; }}
    .eq {{ background: #eef4f2; border-radius: 8px; padding: 10px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border-bottom: 1px solid #e5ece8; text-align: left; padding: 6px; font-size: 0.92rem; }}
    .compare-wrap {{ margin-top: 14px; background: #fff; border: 1px solid #d7e1dc; border-radius: 10px; padding: 12px; }}
    .compare-grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    .bar-row {{ display: grid; grid-template-columns: 96px 1fr 80px; gap: 8px; align-items: center; margin: 8px 0; font-size: 0.9rem; }}
    .bar-track {{ width: 100%; height: 12px; background: #e6efea; border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: linear-gradient(90deg, #0b7285, #4fb3c3); }}
    .bar-fill-alt {{ height: 100%; background: linear-gradient(90deg, #2b8a3e, #80c67b); }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"title\">Roof Hourly Benefit Analysis From PVGIS TMY</div>
    <p class=\"subtitle\">Cities: Marseille and Cairo. Hourly simulation uses PVGIS TMY solar irradiance, air temperature, and wind.</p>

    <details open>
      <summary>Equations Used</summary>
      <div class=\"eq\">{eq1}</div>
      <div class=\"eq\">{eq2}</div>
    </details>

    <div class=\"compare-wrap\">
      <div class=\"title\" style=\"font-size:1.15rem; margin-bottom:6px;\">Side-By-Side City Comparison</div>
      <div class=\"compare-grid\">
        <div>
          <h3>Annual Savings (kWh/m²/year)</h3>
          {''.join(annual_bars)}
        </div>
        <div>
          <h3>Average Temperature Drop (°C)</h3>
          {''.join(drop_bars)}
        </div>
      </div>
      <table>
        <thead><tr><th>City</th><th>Annual Savings (kWh/m²/year)</th><th>Avg Temp Drop (°C)</th></tr></thead>
        <tbody>{''.join(compare_rows)}</tbody>
      </table>
    </div>

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
    p = argparse.ArgumentParser(description="Roof analysis with optional PVGIS TMY hourly mode.")
    p.add_argument("--use-pvgis", action="store_true", help="Fetch PVGIS TMY and run hourly analysis.")
    p.add_argument("--cities", default="marseille,cairo", help="Comma-separated city names (marseille,cairo).")
    p.add_argument("--pvgis-cache-dir", default="pvgis_cache")
    p.add_argument("--sky-offset-c", type=float, default=12.0, help="Sky temperature estimate offset from ambient.")

    p.add_argument("--solar", type=float, default=900.0)
    p.add_argument("--ambient-c", type=float, default=42.0)
    p.add_argument("--sky-c", type=float, default=15.0)
    p.add_argument("--h", type=float, default=8.0)

    p.add_argument("--alpha-baseline", type=float, default=0.85)
    p.add_argument("--eps-baseline", type=float, default=0.90)
    p.add_argument("--alpha-ant", type=float, default=0.35)
    p.add_argument("--eps-ant", type=float, default=0.95)

    p.add_argument("--indoor-c", type=float, default=24.0)
    p.add_argument("--u-value", type=float, default=1.2)
    p.add_argument("--cop", type=float, default=3.2)

    p.add_argument("--html", type=Path, default=None, help="Output HTML path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.use_pvgis:
        cities = [c.strip().lower() for c in args.cities.split(",") if c.strip()]
        summaries: list[dict[str, Any]] = []

        html_path = args.html or Path("ant_roof_pvgis_hourly_report.html")

        for city in cities:
            hourly_rows, summary = hourly_analysis_for_city(city, args)
            summaries.append(summary)
            csv_name = f"roof_hourly_{city}.csv"
            write_hourly_csv(Path(csv_name), hourly_rows)

        render_hourly_html(summaries, html_path)
        print("Roof PVGIS hourly analysis complete")
        for s in summaries:
            print(f"{s['city'].title()}: avg drop {s['avg_temp_drop_c']:.2f} C, annual {s['annual_kwh_m2']:.2f} kWh/m^2/year")
        print(f"HTML report: {html_path.resolve()}")
        return

    inputs = ModelInputs(
        solar_irradiance_w_m2=args.solar,
        ambient_temp_c=args.ambient_c,
        sky_temp_c=args.sky_c,
        convective_h_w_m2k=args.h,
        baseline_solar_absorptance=args.alpha_baseline,
        baseline_ir_emissivity=args.eps_baseline,
        ant_solar_absorptance=args.alpha_ant,
        ant_ir_emissivity=args.eps_ant,
        indoor_temp_c=args.indoor_c,
        roof_u_value_w_m2k=args.u_value,
        cooling_cop=args.cop,
    )
    result = compute_results(inputs)

    monthly = result.electric_savings_kwh_m2_hour * 8.0 * 30.0
    yearly = monthly * 12.0

    print("Sahara ant hair roof cooling analysis")
    print(f"Baseline roof temperature: {result.baseline_roof_temp_c:.2f} C")
    print(f"Ant-inspired roof temperature: {result.ant_roof_temp_c:.2f} C")
    print(f"Temperature drop: {result.roof_temp_drop_c:.2f} C")
    print(f"Cooling electricity savings: {monthly:.2f} kWh/m^2/month")
    print(f"Cooling electricity savings: {yearly:.2f} kWh/m^2/year")


if __name__ == "__main__":
    main()




