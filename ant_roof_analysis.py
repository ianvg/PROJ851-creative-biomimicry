from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import plotly.graph_objects as go
import trimesh

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
    ground_temp_c: float = 42.0
    surrounding_temp_c: float = 42.0
    convective_h_w_m2k: float = 8.0
    view_factor_ground: float = 0.0
    view_factor_sky: float = 1.0
    view_factor_air: float = 0.0
    view_factor_surrounding: float = 0.0

    baseline_solar_absorptance: float = 0.85
    baseline_ir_emissivity: float = 0.90

    ant_solar_absorptance: float = 0.35
    ant_ir_emissivity: float = 0.95

    indoor_temp_c: float = 24.0
    roof_u_value_w_m2k: float = 1.2
    cooling_cop: float = 3.5
    building_width_m: float = 7.0
    building_length_m: float = 10.0
    east_wall_height_m: float = 2.5
    west_wall_height_m: float = 3.0
    window_u_value_w_m2k: float = 1.0 / 0.795
    door_u_value_w_m2k: float = 1.0 / 1.77
    wall_u_value_w_m2k: float = 1.0 / 0.965
    window_width_m: float = 1.0
    window_height_m: float = 2.0
    door_width_m: float = 1.0
    door_height_m: float = 2.0
    windows_north: int = 1
    windows_east: int = 1
    windows_west: int = 1
    windows_south: int = 0


@dataclass
class ModelResults:
    baseline_roof_temp_c: float
    ant_roof_temp_c: float
    roof_temp_drop_c: float
    cooling_power_saved_w_m2: float
    electric_savings_kwh_m2_hour: float
    baseline_building_cooling_w: float
    ant_building_cooling_w: float
    building_cooling_power_saved_w: float
    building_electric_savings_kwh_hour: float


def c_to_k(celsius: float) -> float:
    return celsius + 273.15


def k_to_c(kelvin: float) -> float:
    return kelvin - 273.15


def linearized_radiation_coefficient(
    ts_k: float,
    target_k: float,
    ir_emissivity: float,
    view_factor: float,
) -> float:
    if view_factor <= 0.0:
        return 0.0

    # Stable equivalent of (Ts^4 - Tj^4) / (Ts - Tj), including Ts == Tj.
    delta_t4_over_delta_t = (
        ts_k**3
        + ts_k**2 * target_k
        + ts_k * target_k**2
        + target_k**3
    )
    return ir_emissivity * SIGMA * view_factor * delta_t4_over_delta_t


def longwave_radiation_exchange(ts_k: float, ir_emissivity: float, inputs: ModelInputs) -> float:
    targets = (
        (c_to_k(inputs.ground_temp_c), inputs.view_factor_ground),
        (c_to_k(inputs.sky_temp_c), inputs.view_factor_sky),
        (c_to_k(inputs.ambient_temp_c), inputs.view_factor_air),
        (c_to_k(inputs.surrounding_temp_c), inputs.view_factor_surrounding),
    )

    return sum(
        linearized_radiation_coefficient(ts_k, target_k, ir_emissivity, view_factor) * (target_k - ts_k)
        for target_k, view_factor in targets
    )


def energy_balance(ts_k: float, solar_absorptance: float, ir_emissivity: float, inputs: ModelInputs) -> float:
    ta_k = c_to_k(inputs.ambient_temp_c)

    solar_gain = solar_absorptance * inputs.solar_irradiance_w_m2
    convection = inputs.convective_h_w_m2k * (ta_k - ts_k)
    thermal_radiation = longwave_radiation_exchange(ts_k, ir_emissivity, inputs)

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


def roof_area_m2(inputs: ModelInputs) -> float:
    roof_rise_m = inputs.west_wall_height_m - inputs.east_wall_height_m
    roof_slope_width_m = (inputs.building_width_m**2 + roof_rise_m**2) ** 0.5
    return roof_slope_width_m * inputs.building_length_m


def building_envelope_areas(inputs: ModelInputs) -> dict[str, float]:
    window_area_each_m2 = inputs.window_width_m * inputs.window_height_m
    total_window_area_m2 = window_area_each_m2 * (
        inputs.windows_north + inputs.windows_east + inputs.windows_west + inputs.windows_south
    )
    door_area_m2 = inputs.door_width_m * inputs.door_height_m

    north_wall_area_m2 = inputs.building_width_m * (inputs.east_wall_height_m + inputs.west_wall_height_m) * 0.5
    south_wall_area_m2 = north_wall_area_m2
    east_wall_area_m2 = inputs.building_length_m * inputs.east_wall_height_m
    west_wall_area_m2 = inputs.building_length_m * inputs.west_wall_height_m

    gross_wall_area_m2 = north_wall_area_m2 + south_wall_area_m2 + east_wall_area_m2 + west_wall_area_m2
    opaque_wall_area_m2 = gross_wall_area_m2 - total_window_area_m2 - door_area_m2

    return {
        "roof_area_m2": roof_area_m2(inputs),
        "gross_wall_area_m2": gross_wall_area_m2,
        "opaque_wall_area_m2": max(0.0, opaque_wall_area_m2),
        "window_area_m2": total_window_area_m2,
        "door_area_m2": door_area_m2,
    }


def building_heat_loss_coefficients(inputs: ModelInputs) -> dict[str, float]:
    areas = building_envelope_areas(inputs)
    roof_h_w_k = inputs.roof_u_value_w_m2k * areas["roof_area_m2"]
    wall_h_w_k = inputs.wall_u_value_w_m2k * areas["opaque_wall_area_m2"]
    window_h_w_k = inputs.window_u_value_w_m2k * areas["window_area_m2"]
    door_h_w_k = inputs.door_u_value_w_m2k * areas["door_area_m2"]
    return {
        "roof_h_w_k": roof_h_w_k,
        "wall_h_w_k": wall_h_w_k,
        "window_h_w_k": window_h_w_k,
        "door_h_w_k": door_h_w_k,
        "total_h_w_k": roof_h_w_k + wall_h_w_k + window_h_w_k + door_h_w_k,
    }


def building_cooling_load_w(roof_surface_c: float, inputs: ModelInputs) -> float:
    areas = building_envelope_areas(inputs)
    roof_delta_c = max(0.0, roof_surface_c - inputs.indoor_temp_c)
    ambient_delta_c = max(0.0, inputs.ambient_temp_c - inputs.indoor_temp_c)

    roof_load_w = inputs.roof_u_value_w_m2k * areas["roof_area_m2"] * roof_delta_c
    wall_load_w = inputs.wall_u_value_w_m2k * areas["opaque_wall_area_m2"] * ambient_delta_c
    window_load_w = inputs.window_u_value_w_m2k * areas["window_area_m2"] * ambient_delta_c
    door_load_w = inputs.door_u_value_w_m2k * areas["door_area_m2"] * ambient_delta_c
    return roof_load_w + wall_load_w + window_load_w + door_load_w


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

    baseline_building_cooling_w = building_cooling_load_w(baseline_ts_c, inputs)
    ant_building_cooling_w = building_cooling_load_w(ant_ts_c, inputs)
    building_cooling_power_saved_w = max(0.0, baseline_building_cooling_w - ant_building_cooling_w)
    building_electric_savings_kwh_hour = (building_cooling_power_saved_w / inputs.cooling_cop) / 1000.0

    return ModelResults(
        baseline_roof_temp_c=baseline_ts_c,
        ant_roof_temp_c=ant_ts_c,
        roof_temp_drop_c=temp_drop_c,
        cooling_power_saved_w_m2=cooling_power_saved,
        electric_savings_kwh_m2_hour=electric_savings_kwh_hour,
        baseline_building_cooling_w=baseline_building_cooling_w,
        ant_building_cooling_w=ant_building_cooling_w,
        building_cooling_power_saved_w=building_cooling_power_saved_w,
        building_electric_savings_kwh_hour=building_electric_savings_kwh_hour,
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
    monthly_baseline_kwh = defaultdict(float)
    monthly_ant_kwh = defaultdict(float)
    monthly_saved_kwh = defaultdict(float)
    hour_bins_temp = defaultdict(list)
    hour_bins_baseline_kwh = defaultdict(list)
    hour_bins_ant_kwh = defaultdict(list)
    hour_bins_saved_kwh = defaultdict(list)

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
            ground_temp_c=args.ground_c if args.ground_c is not None else ambient_c,
            surrounding_temp_c=args.surrounding_c if args.surrounding_c is not None else ambient_c,
            convective_h_w_m2k=h_dynamic,
            view_factor_ground=args.f_ground,
            view_factor_sky=args.f_sky,
            view_factor_air=args.f_air,
            view_factor_surrounding=args.f_surrounding,
            baseline_solar_absorptance=args.alpha_baseline,
            baseline_ir_emissivity=args.eps_baseline,
            ant_solar_absorptance=args.alpha_ant,
            ant_ir_emissivity=args.eps_ant,
            indoor_temp_c=args.indoor_c,
            roof_u_value_w_m2k=args.u_value,
            cooling_cop=args.cop,
        )

        result = compute_results(inputs)
        baseline_kwh_hour = (result.baseline_building_cooling_w / args.cop) / 1000.0
        ant_kwh_hour = (result.ant_building_cooling_w / args.cop) / 1000.0
        saved_kwh_hour = result.building_electric_savings_kwh_hour

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
                "electric_saved_kwh_m2_hour": result.electric_savings_kwh_m2_hour,
                "baseline_building_cooling_w": result.baseline_building_cooling_w,
                "ant_building_cooling_w": result.ant_building_cooling_w,
                "building_cooling_saved_w": result.building_cooling_power_saved_w,
                "baseline_building_electric_kwh_hour": baseline_kwh_hour,
                "ant_building_electric_kwh_hour": ant_kwh_hour,
                "building_electric_saved_kwh_hour": saved_kwh_hour,
            }
        )

        monthly_baseline_kwh[month] += baseline_kwh_hour
        monthly_ant_kwh[month] += ant_kwh_hour
        monthly_saved_kwh[month] += saved_kwh_hour
        hour_bins_temp[hour].append(result.roof_temp_drop_c)
        hour_bins_baseline_kwh[hour].append(baseline_kwh_hour)
        hour_bins_ant_kwh[hour].append(ant_kwh_hour)
        hour_bins_saved_kwh[hour].append(saved_kwh_hour)

    annual_baseline_kwh = sum(r["baseline_building_electric_kwh_hour"] for r in hourly_out)
    annual_ant_kwh = sum(r["ant_building_electric_kwh_hour"] for r in hourly_out)
    annual_saved_kwh = sum(r["building_electric_saved_kwh_hour"] for r in hourly_out)
    avg_drop = sum(r["temp_drop_c"] for r in hourly_out) / len(hourly_out)

    hourly_avg = []
    for hour in range(24):
        t_vals = hour_bins_temp.get(hour, [0.0])
        baseline_vals = hour_bins_baseline_kwh.get(hour, [0.0])
        ant_vals = hour_bins_ant_kwh.get(hour, [0.0])
        saved_vals = hour_bins_saved_kwh.get(hour, [0.0])
        hourly_avg.append(
            {
                "hour": hour,
                "avg_temp_drop_c": sum(t_vals) / len(t_vals),
                "avg_baseline_building_kwh_hour": sum(baseline_vals) / len(baseline_vals),
                "avg_ant_building_kwh_hour": sum(ant_vals) / len(ant_vals),
                "avg_building_saved_kwh_hour": sum(saved_vals) / len(saved_vals),
            }
        )

    summary = {
        "city": city,
        "annual_baseline_kwh_building": annual_baseline_kwh,
        "annual_ant_kwh_building": annual_ant_kwh,
        "annual_saved_kwh_building": annual_saved_kwh,
        "monthly_baseline_kwh_building": dict(sorted(monthly_baseline_kwh.items())),
        "monthly_ant_kwh_building": dict(sorted(monthly_ant_kwh.items())),
        "monthly_saved_kwh_building": dict(sorted(monthly_saved_kwh.items())),
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
            "<tr>"
            f"<td>{r['hour']:02d}:00</td>"
            f"<td>{r['avg_temp_drop_c']:.2f}</td>"
            f"<td>{r['avg_baseline_building_kwh_hour']:.4f}</td>"
            f"<td>{r['avg_ant_building_kwh_hour']:.4f}</td>"
            f"<td>{r['avg_building_saved_kwh_hour']:.4f}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _hourly_energy_plot(hourly_avg: list[dict[str, float]], city: str) -> str:
    hours = [f"{r['hour']:02d}:00" for r in hourly_avg]
    baseline = [r["avg_baseline_building_kwh_hour"] for r in hourly_avg]
    ant = [r["avg_ant_building_kwh_hour"] for r in hourly_avg]
    saved = [r["avg_building_saved_kwh_hour"] for r in hourly_avg]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=baseline, mode="lines", name="Before", line={"color": "#c26d1f", "width": 3}))
    fig.add_trace(go.Scatter(x=hours, y=ant, mode="lines", name="After", line={"color": "#1f7a8c", "width": 3}))
    fig.add_trace(go.Scatter(x=hours, y=saved, mode="lines", name="Savings", line={"color": "#2b8a3e", "width": 3}))
    fig.update_layout(
        title=f"{city.title()} Building Electricity By Hour",
        margin={"l": 40, "r": 20, "t": 48, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        xaxis={"title": "Hour of day"},
        yaxis={"title": "Electricity (kWh/h)", "rangemode": "tozero"},
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displaylogo": False})


def _mesh_from_faces(vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]) -> trimesh.Trimesh:
    obj_lines = [f"v {x} {y} {z}" for x, y, z in vertices]
    obj_lines.extend(f"f {i + 1} {j + 1} {k + 1}" for i, j, k in faces)
    obj_payload = "\n".join(obj_lines).encode("utf-8")
    return trimesh.load(io.BytesIO(obj_payload), file_type="obj", force="mesh")


def _mesh3d_trace(mesh: trimesh.Trimesh, color: str, name: str, opacity: float = 1.0) -> go.Mesh3d:
    vertices = mesh.vertices
    faces = mesh.faces
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        flatshading=True,
        hovertemplate=f"{name}<extra></extra>",
    )


def _opening_trace(points: list[tuple[float, float, float]], color: str, name: str) -> go.Scatter3d:
    x, y, z = zip(*points)
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={"color": color, "width": 8},
        name=name,
        hovertemplate=f"{name}<extra></extra>",
        showlegend=False,
    )


def _building_geometry_plot(inputs: ModelInputs) -> str:
    width = inputs.building_width_m
    length = inputs.building_length_m
    east_h = inputs.east_wall_height_m
    west_h = inputs.west_wall_height_m
    roof_rise = west_h - east_h
    roof_area = roof_area_m2(inputs)
    h_values = building_heat_loss_coefficients(inputs)

    west_south_bottom = (0.0, 0.0, 0.0)
    west_north_bottom = (0.0, length, 0.0)
    east_north_bottom = (width, length, 0.0)
    east_south_bottom = (width, 0.0, 0.0)
    west_south_top = (0.0, 0.0, west_h)
    west_north_top = (0.0, length, west_h)
    east_north_top = (width, length, east_h)
    east_south_top = (width, 0.0, east_h)

    shell_vertices = [
        west_south_bottom,
        west_north_bottom,
        east_north_bottom,
        east_south_bottom,
        west_south_top,
        west_north_top,
        east_north_top,
        east_south_top,
    ]
    shell_faces = [
        (0, 1, 5), (0, 5, 4),
        (3, 2, 6), (3, 6, 7),
        (1, 2, 6), (1, 6, 5),
        (0, 3, 7), (0, 7, 4),
        (0, 1, 2), (0, 2, 3),
    ]
    roof_faces = [(4, 5, 6), (4, 6, 7)]

    shell_mesh = _mesh_from_faces(shell_vertices, shell_faces)
    roof_mesh = _mesh_from_faces(shell_vertices, roof_faces)

    def rect_loop(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        return points + [points[0]]

    y_center = length * 0.5
    window_w = inputs.window_width_m
    window_h = inputs.window_height_m
    door_w = inputs.door_width_m
    door_h = inputs.door_height_m
    offset = 0.03

    west_window = rect_loop([
        (offset, y_center - window_w * 0.5, 0.7),
        (offset, y_center + window_w * 0.5, 0.7),
        (offset, y_center + window_w * 0.5, 0.7 + window_h),
        (offset, y_center - window_w * 0.5, 0.7 + window_h),
    ])
    east_window = rect_loop([
        (width - offset, y_center - window_w * 0.5, 0.4),
        (width - offset, y_center + window_w * 0.5, 0.4),
        (width - offset, y_center + window_w * 0.5, 0.4 + window_h),
        (width - offset, y_center - window_w * 0.5, 0.4 + window_h),
    ])
    north_window = rect_loop([
        (width * 0.28, length - offset, 0.6),
        (width * 0.28 + window_w, length - offset, 0.6),
        (width * 0.28 + window_w, length - offset, 0.6 + window_h),
        (width * 0.28, length - offset, 0.6 + window_h),
    ])
    south_door = rect_loop([
        (width * 0.55, offset, 0.0),
        (width * 0.55 + door_w, offset, 0.0),
        (width * 0.55 + door_w, offset, door_h),
        (width * 0.55, offset, door_h),
    ])

    fig = go.Figure()
    fig.add_trace(_mesh3d_trace(shell_mesh, "#d8dee4", "Building shell", opacity=0.92))
    fig.add_trace(_mesh3d_trace(roof_mesh, "#b8742d", "Roof", opacity=0.98))
    fig.add_trace(_opening_trace(west_window, "#3f8fc4", "West window"))
    fig.add_trace(_opening_trace(east_window, "#3f8fc4", "East window"))
    fig.add_trace(_opening_trace(north_window, "#3f8fc4", "North window"))
    fig.add_trace(_opening_trace(south_door, "#6b4022", "South door"))

    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        scene={
            "xaxis": {"title": "Width (m)", "backgroundcolor": "#f4f7f5", "gridcolor": "#d9e2dd"},
            "yaxis": {"title": "Length (m)", "backgroundcolor": "#f4f7f5", "gridcolor": "#d9e2dd"},
            "zaxis": {"title": "Height (m)", "backgroundcolor": "#f4f7f5", "gridcolor": "#d9e2dd"},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.7, "y": -1.8, "z": 1.2}},
        },
    )

    plot_div = fig.to_html(full_html=False, include_plotlyjs=True, config={"responsive": True, "displaylogo": False})
    return f"""
    <div class="card geometry-card">
      <div class="label">Interactive 3D building model</div>
      <div class="plotly-wrap">{plot_div}</div>
      <p class="subtitle" style="margin: 8px 0 0;">Length: {length:.1f} m, width: {width:.1f} m, west height: {west_h:.1f} m, east height: {east_h:.1f} m, roof rise: {roof_rise:.1f} m, sloped roof area: {roof_area:.1f} m^2.</p>
      <p class="subtitle" style="margin: 4px 0 0;">The mesh is loaded with trimesh and displayed with plotly. Window outlines are shown on the north, east, and west sides, with a south door outline.</p>
      <p class="subtitle" style="margin: 4px 0 0;">Building heat-loss coefficients: H_roof = {h_values['roof_h_w_k']:.2f} W/K, H_wall = {h_values['wall_h_w_k']:.2f} W/K, H_window = {h_values['window_h_w_k']:.2f} W/K, H_door = {h_values['door_h_w_k']:.2f} W/K, H_total = {h_values['total_h_w_k']:.2f} W/K.</p>
    </div>
    """




def single_run_summary(result: ModelResults, inputs: ModelInputs) -> dict[str, Any]:
    baseline_kwh_hour = (result.baseline_building_cooling_w / inputs.cooling_cop) / 1000.0
    ant_kwh_hour = (result.ant_building_cooling_w / inputs.cooling_cop) / 1000.0
    saved_kwh_hour = result.building_electric_savings_kwh_hour
    monthly_baseline_kwh = baseline_kwh_hour * 24.0 * 30.0
    monthly_ant_kwh = ant_kwh_hour * 24.0 * 30.0
    monthly_saved_kwh = saved_kwh_hour * 24.0 * 30.0
    annual_baseline_kwh = baseline_kwh_hour * 24.0 * 365.0
    annual_ant_kwh = ant_kwh_hour * 24.0 * 365.0
    annual_saved_kwh = saved_kwh_hour * 24.0 * 365.0
    hourly_avg = [
        {
            "hour": hour,
            "avg_temp_drop_c": result.roof_temp_drop_c,
            "avg_baseline_building_kwh_hour": baseline_kwh_hour,
            "avg_ant_building_kwh_hour": ant_kwh_hour,
            "avg_building_saved_kwh_hour": saved_kwh_hour,
        }
        for hour in range(24)
    ]
    h_values = building_heat_loss_coefficients(inputs)
    return {
        "city": "single case",
        "annual_baseline_kwh_building": annual_baseline_kwh,
        "annual_ant_kwh_building": annual_ant_kwh,
        "annual_saved_kwh_building": annual_saved_kwh,
        "monthly_baseline_kwh_building": {month: monthly_baseline_kwh for month in range(1, 13)},
        "monthly_ant_kwh_building": {month: monthly_ant_kwh for month in range(1, 13)},
        "monthly_saved_kwh_building": {month: monthly_saved_kwh for month in range(1, 13)},
        "avg_temp_drop_c": result.roof_temp_drop_c,
        "hourly_avg": hourly_avg,
        "roof_h_before_w_k": h_values["roof_h_w_k"],
        "roof_h_after_w_k": h_values["roof_h_w_k"],
        "wall_h_before_w_k": h_values["wall_h_w_k"],
        "wall_h_after_w_k": h_values["wall_h_w_k"],
        "window_h_before_w_k": h_values["window_h_w_k"],
        "window_h_after_w_k": h_values["window_h_w_k"],
        "door_h_before_w_k": h_values["door_h_w_k"],
        "door_h_after_w_k": h_values["door_h_w_k"],
        "total_h_before_w_k": h_values["total_h_w_k"],
        "total_h_after_w_k": h_values["total_h_w_k"],
    }
def _roof_assumptions_block(inputs: ModelInputs) -> str:
    eq_base_1 = rf"$$\alpha_{{base}} = {inputs.baseline_solar_absorptance:.2f}, \quad \varepsilon_{{base}} = {inputs.baseline_ir_emissivity:.2f}$$"
    eq_base_2 = rf"$$q_{{roof,base}} = {inputs.baseline_solar_absorptance:.2f}G + h(T_{{air}} - T_s) + q_{{LWR,base}}$$"
    eq_base_3 = rf"$$q_{{LWR,base}} = {inputs.baseline_ir_emissivity:.2f}\sigma [ F_{{gnd}}\frac{{T_s^4 - T_{{gnd}}^4}}{{T_s - T_{{gnd}}}}(T_{{gnd}} - T_s) + F_{{sky}}\frac{{T_s^4 - T_{{sky}}^4}}{{T_s - T_{{sky}}}}(T_{{sky}} - T_s) + F_{{air}}\frac{{T_s^4 - T_{{air}}^4}}{{T_s - T_{{air}}}}(T_{{air}} - T_s) + F_{{srd}}\frac{{T_s^4 - T_{{srd}}^4}}{{T_s - T_{{srd}}}}(T_{{srd}} - T_s) ]$$"
    eq_ant_1 = rf"$$\alpha_{{ant}} = {inputs.ant_solar_absorptance:.2f}, \quad \varepsilon_{{ant}} = {inputs.ant_ir_emissivity:.2f}$$"
    eq_ant_2 = rf"$$q_{{roof,ant}} = {inputs.ant_solar_absorptance:.2f}G + h(T_{{air}} - T_s) + q_{{LWR,ant}}$$"
    eq_ant_3 = rf"$$q_{{LWR,ant}} = {inputs.ant_ir_emissivity:.2f}\sigma [ F_{{gnd}}\frac{{T_s^4 - T_{{gnd}}^4}}{{T_s - T_{{gnd}}}}(T_{{gnd}} - T_s) + F_{{sky}}\frac{{T_s^4 - T_{{sky}}^4}}{{T_s - T_{{sky}}}}(T_{{sky}} - T_s) + F_{{air}}\frac{{T_s^4 - T_{{air}}^4}}{{T_s - T_{{air}}}}(T_{{air}} - T_s) + F_{{srd}}\frac{{T_s^4 - T_{{srd}}^4}}{{T_s - T_{{srd}}}}(T_{{srd}} - T_s) ]$$"
    eq_delta = rf"$$\Delta \alpha = \alpha_{{ant}} - \alpha_{{base}} = {inputs.ant_solar_absorptance - inputs.baseline_solar_absorptance:.2f},\quad \Delta \varepsilon = \varepsilon_{{ant}} - \varepsilon_{{base}} = {inputs.ant_ir_emissivity - inputs.baseline_ir_emissivity:.2f}$$"

    return f"""
    <div class="card" style="margin-top: 16px;">
      <div class="label">Roof assumptions used in the simulation</div>
      <p class="subtitle" style="margin: 8px 0 0;">Baseline roof is assumed to be conventional clay tiles. The ant-inspired case changes the roof optical properties only.</p>
      <p class="subtitle" style="margin: 4px 0 0;">Clay tile baseline</p>
      <div class="eq">{eq_base_1}</div>
      <div class="eq">{eq_base_2}</div>
      <div class="eq">{eq_base_3}</div>
      <p class="subtitle" style="margin: 12px 0 0;">Ant-inspired reflective roof</p>
      <div class="eq">{eq_ant_1}</div>
      <div class="eq">{eq_ant_2}</div>
      <div class="eq">{eq_ant_3}</div>
      <div class="eq">{eq_delta}</div>
    </div>
    """

def render_hourly_html(summaries: list[dict[str, Any]], output_path: Path, geometry_inputs: ModelInputs | None = None) -> None:
    eq1 = r"$$q_{LWR}=h_{r,gnd}(T_{gnd}-T_s)+h_{r,sky}(T_{sky}-T_s)+h_{r,air}(T_{air}-T_s)+h_{r,srd}(T_{srd}-T_s)$$"
    eq1b = r"$$h_{r,gnd}=\varepsilon\sigma F_{gnd}\frac{T_s^4-T_{gnd}^4}{T_s-T_{gnd}},\quad h_{r,sky}=\varepsilon\sigma F_{sky}\frac{T_s^4-T_{sky}^4}{T_s-T_{sky}}$$"
    eq1c = r"$$h_{r,air}=\varepsilon\sigma F_{air}\frac{T_s^4-T_{air}^4}{T_s-T_{air}},\quad h_{r,srd}=\varepsilon\sigma F_{srd}\frac{T_s^4-T_{srd}^4}{T_s-T_{srd}}$$"
    eq1d = r"$$\alpha G + h(T_{air} - T_s) + q_{LWR} = 0$$"
    eq2 = r"$$Q_{cool}=H_{roof}\,\mathrm{max}(0, T_s-T_{in}) + (H_{wall}+H_{win}+H_{door})\,\mathrm{max}(0, T_{air}-T_{in})$$"
    eq3 = r"$$Q_{saved}=Q_{cool,base}-Q_{cool,ant},\quad E_{saved}=\frac{Q_{saved}}{COP\cdot 1000}$$"
    geometry_inputs = geometry_inputs or ModelInputs()
    geometry_block = _building_geometry_plot(geometry_inputs)
    assumptions_block = _roof_assumptions_block(geometry_inputs)

    city_names = ", ".join(s["city"].title() for s in summaries)
    city_blocks = []
    for s in summaries:
        energy_plot_div = _hourly_energy_plot(s["hourly_avg"], s["city"])
        city_blocks.append(
            f"""
      <details>
        <summary>{s['city'].title()} - Hourly And Monthly Benefits</summary>
        <div class=\"grid\" style=\"margin-top:10px;\">
          <div class=\"card\">
            <div class=\"label\">Average roof temperature drop (all TMY hours)</div>
            <div class=\"value ok\">{s['avg_temp_drop_c']:.2f} deg C</div>
          </div>
          <div class=\"card\">
            <div class=\"label\">Annual building electricity before</div>
            <div class=\"value\">{s['annual_baseline_kwh_building']:.2f} kWh/year</div>
          </div>
          <div class=\"card\">
            <div class=\"label\">Annual building electricity after</div>
            <div class=\"value\">{s['annual_ant_kwh_building']:.2f} kWh/year</div>
          </div>
          <div class=\"card\">
            <div class=\"label\">Annual building electricity savings</div>
            <div class=\"value ok\">{s['annual_saved_kwh_building']:.2f} kWh/year</div>
          </div>
        </div>
        <h3>Monthly Building Electricity Savings (kWh)</h3>
        <table>
          <thead><tr><th>Month</th><th>kWh</th></tr></thead>
          <tbody>{_monthly_rows(s['monthly_saved_kwh_building'])}</tbody>
        </table>
        <h3>Average Building Electricity By Hour Of Day</h3>
        <div class=\"card\" style=\"margin-top:8px;\">
          <div class=\"label\">Before vs after vs savings on the same hourly graph</div>
          <div class=\"plotly-wrap\">{energy_plot_div}</div>
        </div>
        <table>
          <thead><tr><th>Hour</th><th>Avg Temp Drop (deg C)</th><th>Avg Before (kWh/h)</th><th>Avg After (kWh/h)</th><th>Avg Savings (kWh/h)</th></tr></thead>
          <tbody>{_hourly_rows(s['hourly_avg'])}</tbody>
        </table>
      </details>
"""
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Roof PVGIS Hourly Analysis</title>
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['$', '$'], ['\\(', '\\)']], displayMath: [['$$', '$$'], ['\\[', '\\]']] }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script defer src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js\"></script>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <script>
    window.addEventListener('load', function () {{
      if (window.MathJax && window.MathJax.typesetPromise) {{
        window.MathJax.typesetPromise();
      }}
    }});
  </script>
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
    .nav {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 18px; padding: 12px; background: #ffffff; border: 1px solid #d7e1dc; border-radius: 12px; }}
    .nav a {{ color: #0b7285; text-decoration: none; font-weight: 600; }}
    .nav a:hover {{ text-decoration: underline; }}
    .geometry-card {{ margin-top: 16px; }}
    .plotly-wrap {{ margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border-bottom: 1px solid #e5ece8; text-align: left; padding: 6px; font-size: 0.92rem; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"nav\"><a href=\"index.html\">Home</a><a href=\"interactive_roof_report.html\">Interactive Roof Report</a><a href=\"ant_roof_cooling_report.html\">Roof Cooling Report</a><a href=\"hat_analysis_report.html\">Hat Analysis Report</a></div>
    <div class=\"title\">Roof Hourly Benefit Analysis From PVGIS TMY</div>
    <p class=\"subtitle\">Cities: {city_names}. Hourly simulation uses PVGIS TMY solar irradiance, air temperature, and wind.</p>


    {geometry_block}
    {assumptions_block}
    <details open>
      <summary>Equations Used</summary>
      <div class=\"eq\">{eq1}</div>
      <div class=\"eq\">{eq1b}</div>
      <div class=\"eq\">{eq1c}</div>
      <div class=\"eq\">{eq1d}</div>
      <div class=\"eq\">{eq2}</div>
      <div class=\"eq\">{eq3}</div>
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
    p = argparse.ArgumentParser(description="Roof analysis with optional PVGIS TMY hourly mode.")
    p.add_argument("--use-pvgis", action="store_true", help="Fetch PVGIS TMY and run hourly analysis.")
    p.add_argument("--cities", default="marseille", help="Comma-separated city names (default: marseille).")
    p.add_argument("--pvgis-cache-dir", default="pvgis_cache")
    p.add_argument("--sky-offset-c", type=float, default=12.0, help="Sky temperature estimate offset from ambient.")

    p.add_argument("--solar", type=float, default=900.0)
    p.add_argument("--ambient-c", type=float, default=42.0)
    p.add_argument("--sky-c", type=float, default=15.0)
    p.add_argument("--ground-c", type=float, default=None, help="Ground radiant temperature. Defaults to ambient.")
    p.add_argument("--surrounding-c", type=float, default=None, help="Surrounding radiant temperature. Defaults to ambient.")
    p.add_argument("--h", type=float, default=8.0)
    p.add_argument("--f-ground", type=float, default=0.0, help="Longwave view factor to ground.")
    p.add_argument("--f-sky", type=float, default=1.0, help="Longwave view factor to sky.")
    p.add_argument("--f-air", type=float, default=0.0, help="Longwave view factor to air.")
    p.add_argument("--f-surrounding", type=float, default=0.0, help="Longwave view factor to surroundings.")

    p.add_argument("--alpha-baseline", type=float, default=0.85)
    p.add_argument("--eps-baseline", type=float, default=0.90)
    p.add_argument("--alpha-ant", type=float, default=0.35)
    p.add_argument("--eps-ant", type=float, default=0.95)

    p.add_argument("--indoor-c", type=float, default=24.0)
    p.add_argument("--u-value", type=float, default=1.2)
    p.add_argument("--cop", type=float, default=3.5)

    p.add_argument("--html", type=Path, default=None, help="Output HTML path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.use_pvgis:
        cities = [c.strip().lower() for c in args.cities.split(",") if c.strip()]
        summaries: list[dict[str, Any]] = []

        html_path = args.html or Path("Reports") / "ant_roof_pvgis_hourly_report.html"

        for city in cities:
            hourly_rows, summary = hourly_analysis_for_city(city, args)
            summaries.append(summary)
            csv_name = f"roof_hourly_{city}.csv"
            write_hourly_csv(Path(csv_name), hourly_rows)

        geometry_inputs = ModelInputs(
            building_width_m=7.0,
            building_length_m=10.0,
            east_wall_height_m=2.5,
            west_wall_height_m=3.0,
        )
        render_hourly_html(summaries, html_path, geometry_inputs)
        print("Roof PVGIS hourly analysis complete")
        for s in summaries:
            print(f"{s['city'].title()}: avg drop {s['avg_temp_drop_c']:.2f} C, annual savings {s['annual_saved_kwh_building']:.2f} kWh/year")
        print(f"HTML report: {html_path.resolve()}")
        return

    inputs = ModelInputs(
        solar_irradiance_w_m2=args.solar,
        ambient_temp_c=args.ambient_c,
        sky_temp_c=args.sky_c,
        ground_temp_c=args.ground_c if args.ground_c is not None else args.ambient_c,
        surrounding_temp_c=args.surrounding_c if args.surrounding_c is not None else args.ambient_c,
        convective_h_w_m2k=args.h,
        view_factor_ground=args.f_ground,
        view_factor_sky=args.f_sky,
        view_factor_air=args.f_air,
        view_factor_surrounding=args.f_surrounding,
        baseline_solar_absorptance=args.alpha_baseline,
        baseline_ir_emissivity=args.eps_baseline,
        ant_solar_absorptance=args.alpha_ant,
        ant_ir_emissivity=args.eps_ant,
        indoor_temp_c=args.indoor_c,
        roof_u_value_w_m2k=args.u_value,
        cooling_cop=args.cop,
    )
    result = compute_results(inputs)

    monthly = result.building_electric_savings_kwh_hour * 8.0 * 30.0
    yearly = monthly * 12.0

    print("Sahara ant hair roof cooling analysis")
    print(f"Baseline roof temperature: {result.baseline_roof_temp_c:.2f} C")
    print(f"Ant-inspired roof temperature: {result.ant_roof_temp_c:.2f} C")
    print(f"Temperature drop: {result.roof_temp_drop_c:.2f} C")
    print(f"Baseline building cooling load: {result.baseline_building_cooling_w:.2f} W")
    print(f"Ant-inspired building cooling load: {result.ant_building_cooling_w:.2f} W")
    h_values = building_heat_loss_coefficients(inputs)
    print(f"Building cooling load saved: {result.building_cooling_power_saved_w:.2f} W")
    print(f"H_roof before/after: {h_values['roof_h_w_k']:.2f} / {h_values['roof_h_w_k']:.2f} W/K")
    print(f"H_wall before/after: {h_values['wall_h_w_k']:.2f} / {h_values['wall_h_w_k']:.2f} W/K")
    print(f"H_window before/after: {h_values['window_h_w_k']:.2f} / {h_values['window_h_w_k']:.2f} W/K")
    print(f"H_door before/after: {h_values['door_h_w_k']:.2f} / {h_values['door_h_w_k']:.2f} W/K")
    print(f"H_total before/after: {h_values['total_h_w_k']:.2f} / {h_values['total_h_w_k']:.2f} W/K")
    print(f"Building electricity savings: {monthly:.2f} kWh/month")
    print(f"Building electricity savings: {yearly:.2f} kWh/year")

    html_path = args.html or Path("Reports") / "interactive_roof_report.html"
    render_hourly_html([single_run_summary(result, inputs)], html_path, inputs)
    print(f"HTML report: {html_path.resolve()}")


if __name__ == "__main__":
    main()
