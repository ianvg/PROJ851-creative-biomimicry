from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant (W/m2/K4)


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

    daytime_hours_per_day: float = 8.0
    days_per_month: float = 30.0
    months_per_year: float = 12.0


@dataclass
class ModelResults:
    baseline_roof_temp_c: float
    ant_roof_temp_c: float
    roof_temp_drop_c: float
    cooling_power_saved_w_m2: float
    electric_savings_kwh_m2_month: float
    electric_savings_kwh_m2_year: float


def c_to_k(celsius: float) -> float:
    return celsius + 273.15


def k_to_c(kelvin: float) -> float:
    return kelvin - 273.15


def energy_balance(ts_k: float, solar_absorptance: float, ir_emissivity: float, inputs: ModelInputs) -> float:
    """Returns residual of steady-state energy equation for roof outer surface.

    Equation (W/m2):
    alpha*G + h*(Ta - Ts) + eps*sigma*(Tsky^4 - Ts^4) = 0
    """
    ta_k = c_to_k(inputs.ambient_temp_c)
    tsky_k = c_to_k(inputs.sky_temp_c)

    solar_gain = solar_absorptance * inputs.solar_irradiance_w_m2
    convection = inputs.convective_h_w_m2k * (ta_k - ts_k)
    thermal_radiation = ir_emissivity * SIGMA * (tsky_k**4 - ts_k**4)

    return solar_gain + convection + thermal_radiation


def solve_surface_temp_k(solar_absorptance: float, ir_emissivity: float, inputs: ModelInputs) -> float:
    """Bisection solve for root of energy balance in Kelvin."""
    low_k = c_to_k(-20.0)
    high_k = c_to_k(120.0)

    f_low = energy_balance(low_k, solar_absorptance, ir_emissivity, inputs)
    f_high = energy_balance(high_k, solar_absorptance, ir_emissivity, inputs)

    if f_low == 0:
        return low_k
    if f_high == 0:
        return high_k
    if f_low * f_high > 0:
        raise ValueError(
            "Could not bracket a valid roof temperature solution. "
            "Try adjusting inputs (solar irradiance, air temp, or optical properties)."
        )

    for _ in range(100):
        mid_k = 0.5 * (low_k + high_k)
        f_mid = energy_balance(mid_k, solar_absorptance, ir_emissivity, inputs)

        if abs(f_mid) < 1e-6:
            return mid_k

        if f_low * f_mid < 0:
            high_k = mid_k
            f_high = f_mid
        else:
            low_k = mid_k
            f_low = f_mid

    return 0.5 * (low_k + high_k)


def compute_results(inputs: ModelInputs) -> ModelResults:
    baseline_ts_k = solve_surface_temp_k(
        inputs.baseline_solar_absorptance,
        inputs.baseline_ir_emissivity,
        inputs,
    )
    ant_ts_k = solve_surface_temp_k(
        inputs.ant_solar_absorptance,
        inputs.ant_ir_emissivity,
        inputs,
    )

    baseline_ts_c = k_to_c(baseline_ts_k)
    ant_ts_c = k_to_c(ant_ts_k)
    temp_drop_c = baseline_ts_c - ant_ts_c

    q_in_baseline = max(0.0, inputs.roof_u_value_w_m2k * (baseline_ts_c - inputs.indoor_temp_c))
    q_in_ant = max(0.0, inputs.roof_u_value_w_m2k * (ant_ts_c - inputs.indoor_temp_c))
    cooling_power_saved = max(0.0, q_in_baseline - q_in_ant)

    cooling_hours_month = inputs.daytime_hours_per_day * inputs.days_per_month
    cooling_hours_year = cooling_hours_month * inputs.months_per_year

    electric_savings_kwh_m2_month = (cooling_power_saved / inputs.cooling_cop) * cooling_hours_month / 1000.0
    electric_savings_kwh_m2_year = (cooling_power_saved / inputs.cooling_cop) * cooling_hours_year / 1000.0

    return ModelResults(
        baseline_roof_temp_c=baseline_ts_c,
        ant_roof_temp_c=ant_ts_c,
        roof_temp_drop_c=temp_drop_c,
        cooling_power_saved_w_m2=cooling_power_saved,
        electric_savings_kwh_m2_month=electric_savings_kwh_m2_month,
        electric_savings_kwh_m2_year=electric_savings_kwh_m2_year,
    )


def render_html(inputs: ModelInputs, results: ModelResults, output_path: Path) -> None:
    inputs_json = json.dumps(asdict(inputs), indent=2)

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Sahara Ant Hair Roof Cooling Analysis</title>
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script defer src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js\"></script>
  <style>
    :root {{
      --bg: #f4f7f5;
      --card: #ffffff;
      --ink: #1d2a2a;
      --accent: #0b7285;
      --ok: #2b8a3e;
      --muted: #5b6b6b;
      --border: #d7e1dc;
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: radial-gradient(circle at top right, #d9ecef 0%, var(--bg) 45%);
      color: var(--ink);
      font-family: \"Segoe UI\", Tahoma, sans-serif;
      line-height: 1.4;
    }}
    .wrap {{ max-width: 1000px; margin: 0 auto; }}
    .title {{ margin-bottom: 8px; font-size: 1.8rem; font-weight: 700; }}
    .subtitle {{ margin: 0 0 20px; color: var(--muted); }}
    .grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }}
    .label {{ font-size: 0.86rem; color: var(--muted); }}
    .value {{ font-size: 1.5rem; font-weight: 700; margin-top: 4px; }}
    .ok {{ color: var(--ok); }}
    details {{
      margin-top: 14px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
    }}
    summary {{ cursor: pointer; font-weight: 600; color: var(--accent); }}
    code, pre {{
      font-family: Consolas, \"Courier New\", monospace;
      background: #f1f5f3;
      border-radius: 8px;
    }}
    pre {{ padding: 12px; overflow-x: auto; }}
    .equation {{
      font-size: 1.04rem;
      background: #f1f5f3;
      border-radius: 8px;
      padding: 10px 12px;
      overflow-x: auto;
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"title\">Sahara Desert Ant Hair Roof Cooling Analysis</div>
    <p class=\"subtitle\">Click sections below to inspect equations, input values, and detailed outputs.</p>

    <div class=\"grid\">
      <div class=\"card\">
        <div class=\"label\">Baseline roof surface temperature</div>
        <div class=\"value\">{results.baseline_roof_temp_c:.2f} \u00b0C</div>
      </div>
      <div class=\"card\">
        <div class=\"label\">Ant-hair-inspired roof surface temperature</div>
        <div class=\"value\">{results.ant_roof_temp_c:.2f} \u00b0C</div>
      </div>
      <div class=\"card\">
        <div class=\"label\">Roof temperature drop</div>
        <div class=\"value ok\">{results.roof_temp_drop_c:.2f} \u00b0C</div>
      </div>
      <div class=\"card\">
        <div class=\"label\">Cooling electricity saved</div>
        <div class=\"value ok\">{results.electric_savings_kwh_m2_month:.2f} kWh/m\u00b2/month</div>
        <div style=\"margin-top:4px;color:var(--muted);\">{results.electric_savings_kwh_m2_year:.2f} kWh/m\u00b2/year</div>
      </div>
    </div>

    <details>
      <summary>Equation 1: Roof surface steady-state energy balance</summary>
      <div class=\"equation\">\\[
      \\alpha G + h\\,(T_a - T_s) + \\varepsilon\\sigma\\,(T_{sky}^{4} - T_{s}^{4}) = 0
      \\]</div>
      <p>
      where:<br>
      \\(\\alpha\\) = solar absorptance, \\(G\\) = solar irradiance (W/m\u00b2), \\(h\\) = convection coefficient (W/m\u00b2K),<br>
      \\(\\varepsilon\\) = IR emissivity, \\(\\sigma\\) = Stefan-Boltzmann constant, \\(T_a\\), \\(T_s\\), \\(T_{{sky}}\\) in Kelvin.
      </p>
    </details>

    <details>
      <summary>Equation 2: Cooling load reduction at roof-to-indoor boundary</summary>
      <div class=\"equation\">\\[
      q_{{in}} = \\max\\left(0, U\\,(T_s - T_{{in}})\\right)
      \\]</div>
      <div class=\"equation\">\\[
      q_{{saved}} = q_{{in,baseline}} - q_{{in,ant}}
      \\]</div>
      <div class=\"equation\">\\[
      E_{{saved}} = \\frac{{q_{{saved}}}}{{COP}}\\cdot\\frac{{hours}}{{1000}}
      \\]</div>
      <p>
      where \\(U\\) is roof U-value (W/m\u00b2K), \\(T_{{in}}\\) indoor setpoint, and \\(COP\\) cooling system coefficient of performance.
      </p>
    </details>

    <details>
      <summary>Input values used</summary>
      <pre>{inputs_json}</pre>
    </details>

    <details>
      <summary>Computed outputs</summary>
      <pre>{json.dumps(asdict(results), indent=2)}</pre>
    </details>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze roof cooling from Sahara desert ant hair-inspired optical properties.",
    )

    p.add_argument("--solar", type=float, default=900.0, help="Solar irradiance (W/m^2).")
    p.add_argument("--ambient-c", type=float, default=42.0, help="Ambient air temperature (C).")
    p.add_argument("--sky-c", type=float, default=15.0, help="Effective sky temperature (C).")
    p.add_argument("--h", type=float, default=8.0, help="Convective coefficient h (W/m^2K).")

    p.add_argument("--alpha-baseline", type=float, default=0.85, help="Baseline solar absorptance.")
    p.add_argument("--eps-baseline", type=float, default=0.90, help="Baseline IR emissivity.")

    p.add_argument("--alpha-ant", type=float, default=0.35, help="Ant-inspired solar absorptance.")
    p.add_argument("--eps-ant", type=float, default=0.95, help="Ant-inspired IR emissivity.")

    p.add_argument("--indoor-c", type=float, default=24.0, help="Indoor setpoint temperature (C).")
    p.add_argument("--u-value", type=float, default=1.2, help="Roof U-value (W/m^2K).")
    p.add_argument("--cop", type=float, default=3.2, help="Cooling COP.")

    p.add_argument("--day-hours", type=float, default=8.0, help="Equivalent high-load hours per day.")
    p.add_argument("--days-month", type=float, default=30.0, help="Days per month.")
    p.add_argument("--months-year", type=float, default=12.0, help="Months per year.")

    p.add_argument(
        "--html",
        type=Path,
        default=Path("ant_roof_cooling_report.html"),
        help="Output HTML report path.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

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
        daytime_hours_per_day=args.day_hours,
        days_per_month=args.days_month,
        months_per_year=args.months_year,
    )

    results = compute_results(inputs)
    render_html(inputs, results, args.html)

    print("Sahara ant hair roof cooling analysis")
    print(f"Baseline roof temperature: {results.baseline_roof_temp_c:.2f} C")
    print(f"Ant-inspired roof temperature: {results.ant_roof_temp_c:.2f} C")
    print(f"Temperature drop: {results.roof_temp_drop_c:.2f} C")
    print(f"Cooling electricity savings: {results.electric_savings_kwh_m2_month:.2f} kWh/m^2/month")
    print(f"Cooling electricity savings: {results.electric_savings_kwh_m2_year:.2f} kWh/m^2/year")
    print(f"HTML report: {args.html.resolve()}")


if __name__ == "__main__":
    main()
