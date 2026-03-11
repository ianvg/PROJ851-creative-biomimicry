# Sahara Desert Ant Analysis

This analysis investigates how Sahara desert ant-inspired data and design principles can improve product efficiency.

## Objective
Estimate how much efficiency the product could improve, reported as:
- `kWh/m^2/month`
- `kWh/m^2/year`

## Interactive Map Analysis (Any Lat/Lon)
Use the map-based workflow to click any location, fetch PVGIS TMY data, and generate both roof + hat hourly analysis outputs.

Run:

```bat
run_interactive_map_analysis.bat
```

Then in the browser:
1. Click a point on the map.
2. Optionally enter a location name.
3. Click **Run Analysis**.

Outputs are generated automatically in:
- `roof analysis/` (HTML + CSV)
- `hat analysis/` (HTML + CSV)

## Core Question
How much additional energy efficiency can be achieved by applying Sahara desert ant-inspired strategies to the product?

## Notes
This is a first-order physics model (solar absorption + convection + thermal radiation + conduction/convection assumptions). For final engineering decisions, calibrate with measured data.
