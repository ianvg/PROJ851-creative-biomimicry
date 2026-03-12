# Roof Analysis

## Standard run
```bat
run_ant_roof_analysis.bat
```

## PVGIS hourly run (Marseille + Cairo)
```bat
run_roof_pvgis_hourly.bat
```

Outputs (PVGIS mode):
- `ant_roof_pvgis_hourly_report.html`
- `roof_hourly_marseille.csv`
- `roof_hourly_cairo.csv`

The hourly report includes LaTeX equations, monthly totals, and average benefit by hour-of-day.

The Python entry point is `simple_roof_analysis.py`.
