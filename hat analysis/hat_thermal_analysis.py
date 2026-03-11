from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

SIGMA = 5.670374419e-8  # W/m^2/K^4


@dataclass
class HatInputs:
    # Beach summer design point (default assumptions)
    solar_irradiance_w_m2: float = 1000.0
    ambient_temp_c: float = 35.0
    sky_temp_c: float = 20.0
    convective_h_w_m2k: float = 12.0

    # Round hat geometry factor: only part of curved shell normal to sun at any moment
    solar_view_factor: float = 0.68

    # Baseline hat optical properties (typical dark fabric)
    baseline_solar_absorptance: float = 0.80
    baseline_ir_emissivity: float = 0.90

    # Ant-hair-inspired coating properties
    ant_solar_absorptance: float = 0.34
    ant_ir_emissivity: float = 0.95

    # Optional cooling-equivalent metric (not building energy, just normalized thermal benefit)
    cooling_cop: float = 3.2
    equivalent_sun_hours_per_day: float = 6.0
    days_per_month: float = 30.0
    months_per_year: float = 12.0


@dataclass
class HatResults:
    baseline_shell_temp_c: float
    ant_shell_temp_c: float
    shell_temp_drop_c: float
    equivalent_heat_reduction_w_m2: float
    equivalent_electric_savings_kwh_m2_month: float
    equivalent_electric_savings_kwh_m2_year: float


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
        raise ValueError("Unable to bracket temperature root. Adjust inputs.")

    for _ in range(100):
        mid = 0.5 * (low_k + high_k)
        f_mid = energy_balance(mid, alpha, eps, i)

        if abs(f_mid) < 1e-6:
            return mid

        if f_low * f_mid < 0:
            high_k = mid
            f_high = f_mid
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

    # Equivalent heat reduction by avoiding absorbed solar flux.
    q_solar_base = i.baseline_solar_absorptance * i.solar_view_factor * i.solar_irradiance_w_m2
    q_solar_ant = i.ant_solar_absorptance * i.solar_view_factor * i.solar_irradiance_w_m2
    q_reduction = max(0.0, q_solar_base - q_solar_ant)

    hours_month = i.equivalent_sun_hours_per_day * i.days_per_month
    hours_year = hours_month * i.months_per_year

    e_month = (q_reduction / i.cooling_cop) * hours_month / 1000.0
    e_year = (q_reduction / i.cooling_cop) * hours_year / 1000.0

    return HatResults(
        baseline_shell_temp_c=base_c,
        ant_shell_temp_c=ant_c,
        shell_temp_drop_c=drop_c,
        equivalent_heat_reduction_w_m2=q_reduction,
        equivalent_electric_savings_kwh_m2_month=e_month,
        equivalent_electric_savings_kwh_m2_year=e_year,
    )


def render_html(inputs: HatInputs, results: HatResults, path: Path) -> None:
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Round Hat Thermal Analysis</title>
  <style>
    :root {{
      --bg: #f6f2e8;
      --ink: #2d2417;
      --card: #fffdfa;
      --accent: #a35f14;
      --ok: #2b8a3e;
      --muted: #6a5c46;
      --border: #e9dcc5;
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: radial-gradient(circle at top right, #ffe8b8 0%, var(--bg) 44%);
      color: var(--ink);
      font-family: "Segoe UI", Tahoma, sans-serif;
      line-height: 1.4;
    }}
    .wrap {{ max-width: 980px; margin: 0 auto; }}
    .title {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 8px; }}
    .subtitle {{ margin-top: 0; color: var(--muted); }}
    .grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 14px; }}
    .label {{ font-size: 0.86rem; color: var(--muted); }}
    .value {{ font-size: 1.48rem; font-weight: 700; margin-top: 6px; }}
    .ok {{ color: var(--ok); }}
    details {{ margin-top: 14px; background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }}
    summary {{ cursor: pointer; color: var(--accent); font-weight: 600; }}
    pre {{ background: #f9f4ea; border-radius: 8px; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"title\">Round Hat Cooling Analysis (Beach Full Sun)</div>
    <p class=\"subtitle\">Click sections to inspect equations, assumptions, input values, and outputs.</p>

    <div class=\"grid\">
      <div class=\"card\">
        <div class=\"label\">Baseline hat shell temperature</div>
        <div class=\"value\">{results.baseline_shell_temp_c:.2f} \u00b0C</div>
      </div>
      <div class=\"card\">
        <div class=\"label\">Ant-inspired hat shell temperature</div>
        <div class=\"value\">{results.ant_shell_temp_c:.2f} \u00b0C</div>
      </div>
      <div class=\"card\">
        <div class=\"label\">Estimated shell temperature drop</div>
        <div class=\"value ok\">{results.shell_temp_drop_c:.2f} \u00b0C</div>
      </div>
      <div class=\"card\">
        <div class=\"label\">Equivalent cooling benefit</div>
        <div class=\"value ok\">{results.equivalent_electric_savings_kwh_m2_month:.2f} kWh/m\u00b2/month</div>
        <div style=\"margin-top:4px;color:var(--muted);\">{results.equivalent_electric_savings_kwh_m2_year:.2f} kWh/m\u00b2/year</div>
      </div>
    </div>

    <details>
      <summary>Equation 1: Hat shell energy balance</summary>
      <pre>alpha * f_view * G + h*(Ta - Ts) + eps*sigma*(Tsky^4 - Ts^4) = 0</pre>
      <p>
      For a round hat, <code>f_view</code> accounts for curved geometry reducing effective direct solar load.
      </p>
    </details>

    <details>
      <summary>Equation 2: Equivalent cooling metric</summary>
      <pre>q_reduction = (alpha_base - alpha_ant) * f_view * G
E_equiv = (q_reduction / COP) * hours / 1000</pre>
      <p>
      This converts heat-flux reduction to an equivalent cooling-electricity intensity basis for comparison.
      </p>
    </details>

    <details>
      <summary>Input values</summary>
      <pre>{json.dumps(asdict(inputs), indent=2)}</pre>
    </details>

    <details>
      <summary>Computed outputs</summary>
      <pre>{json.dumps(asdict(results), indent=2)}</pre>
    </details>
  </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Round hat thermal analysis under full-sun beach conditions.")

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
    p.add_argument("--sun-hours", type=float, default=6.0)
    p.add_argument("--days-month", type=float, default=30.0)
    p.add_argument("--months-year", type=float, default=12.0)

    p.add_argument(
        "--html",
        type=Path,
        default=Path("hat_analysis_report.html"),
        help="Output HTML report path",
    )

    return p.parse_args()


def main() -> None:
    a = parse_args()
    inputs = HatInputs(
        solar_irradiance_w_m2=a.solar,
        ambient_temp_c=a.ambient_c,
        sky_temp_c=a.sky_c,
        convective_h_w_m2k=a.h,
        solar_view_factor=a.view_factor,
        baseline_solar_absorptance=a.alpha_baseline,
        baseline_ir_emissivity=a.eps_baseline,
        ant_solar_absorptance=a.alpha_ant,
        ant_ir_emissivity=a.eps_ant,
        cooling_cop=a.cop,
        equivalent_sun_hours_per_day=a.sun_hours,
        days_per_month=a.days_month,
        months_per_year=a.months_year,
    )

    results = analyze(inputs)
    render_html(inputs, results, a.html)

    print("Round hat thermal analysis")
    print(f"Baseline shell temperature: {results.baseline_shell_temp_c:.2f} C")
    print(f"Ant-inspired shell temperature: {results.ant_shell_temp_c:.2f} C")
    print(f"Shell temperature drop: {results.shell_temp_drop_c:.2f} C")
    print(f"Equivalent savings: {results.equivalent_electric_savings_kwh_m2_month:.2f} kWh/m^2/month")
    print(f"Equivalent savings: {results.equivalent_electric_savings_kwh_m2_year:.2f} kWh/m^2/year")
    print(f"HTML report: {a.html.resolve()}")


if __name__ == "__main__":
    main()
