# Sahara Desert Ant Analysis

This analysis investigates how Sahara desert ant-inspired data and design principles can improve product efficiency.

## Objective
Estimate how much efficiency the product could improve, reported as:
- `kWh/m^2/month`
- `kWh/m^2/year`

## Python Analysis Tool
Run the local model:

```bash
python ant_roof_analysis.py
```

The script prints:
- baseline roof temperature
- ant-inspired roof temperature
- roof temperature drop
- electricity savings (`kWh/m^2/month`, `kWh/m^2/year`)

It also creates a clickable HTML report:
- `ant_roof_cooling_report.html`

In the HTML report, click each section to view:
- equations used
- input values
- computed outputs

## Example custom run

```bash
python ant_roof_analysis.py --solar 1000 --ambient-c 45 --alpha-ant 0.30 --eps-ant 0.96
```

## Core Question
How much additional energy efficiency can be achieved by applying Sahara desert ant-inspired strategies to the product?

## Notes
This is a first-order physics model (solar absorption + convection + thermal radiation + roof conduction). For final engineering decisions, calibrate with measured roof and weather data.
