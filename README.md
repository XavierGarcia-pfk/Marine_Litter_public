# Marine Litter vs Precipitation Analysis

**Quantifying precipitation and river discharge effects on benthic marine litter accumulation in the NW Mediterranean**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Overview

This repository contains the analysis code for a systematic study examining how precipitation and river discharge influence benthic marine litter accumulation along the Catalan coast (NW Mediterranean, 2019-2024). The analysis uses negative binomial generalized linear models (GLMs) to test relationships between hydrological variables and litter abundance from bottom-trawl surveys.

**Key findings:**
- Rivers predominantly **remove** rather than deliver litter (5 of 6 ports show washout)
- Urban CSO systems are the **primary delivery mechanism** (7 of 9 ports show accumulation)
- **Station selection is critical**: within-port heterogeneity reaches 150 percentage points
- **Distinct temporal scales**: CSO effects peak at 2-6 weeks, river effects at 12-18 weeks

---

## Features

- **Multi-station sensitivity testing**: Systematically tests 51 station-ground combinations across 9 ports
- **Optimal window selection**: AIC-based comparison of 2-20 week cumulative precipitation windows
- **Negative binomial GLM**: Accounts for overdispersion in marine litter count data
- **River discharge analysis**: Tests effects at 6 hydrologically connected ports
- **Litter type partitioning**: Exploratory analysis by material categories (light-urban, heavy-urban, heavy-industrial)
- **Geographic controls**: Validates methodology using negative control sites

---

## Repository Structure

```
.
├── main.py                          # Main analysis script
├── test_multiple_stations.py        # Station sensitivity analysis wrapper
├── src/                             # Source code modules
│   ├── data_loading.py              # Data import and preprocessing
│   ├── model_fitting.py             # GLM implementation
│   ├── window_selection.py          # AIC-based window optimization
│   ├── station_selection.py         # Station testing framework
│   ├── river_analysis.py            # River discharge effects
│   ├── visualization.py             # Figure generation
│   └── utils.py                     # Helper functions
├── data/                            # Data directory (not included)
│   ├── hauls/                       # Trawl survey data (ICATMAR)
│   ├── precipitation/               # XEMA meteorological data
│   └── rivers/                      # River discharge data (ACA, SAIH)
├── outputs/                         # Analysis outputs
│   ├── figures/                     # Generated plots
│   ├── tables/                      # Statistical results
│   └── reports/                     # Station comparison reports
└── README.md                        # This file
```

---

## Installation

### Requirements

- Python 3.8 or higher
- Required packages:

```bash
pip install numpy pandas scipy statsmodels matplotlib seaborn
```

### Optional dependencies

For extended functionality:

```bash
pip install geopandas contextily  # For spatial visualization
```

---

## Usage

### Basic Analysis

Run the main analysis for a single port-station combination:

```bash
python main.py --port 64100 --station XU --window 2
```

**Arguments:**
- `--port`: Port ID (e.g., 64100 for Vilanova i la Geltrú)
- `--station`: XEMA meteorological station code (e.g., XU)
- `--window`: Precipitation window in weeks (2-20)

### Multi-Station Analysis

Test all station-ground combinations:

```bash
python test_multiple_stations.py --all-ports
```

This will:
1. Load all port-station combinations from configuration
2. Run GLM analysis for each combination
3. Compare AIC across 2-20 week windows
4. Generate station comparison reports
5. Identify optimal stations based on mechanistic criteria

**Options:**
- `--all-ports`: Test all 9 ports
- `--port <ID>`: Test specific port only
- `--windows 2,4,6,8,10,12,16,18,20`: Custom window set

### Output

Results are saved to `outputs/`:

**Figures:**
- `aic_profiles.pdf`: AIC comparison across windows
- `station_comparison.pdf`: Effect sizes for all stations
- `river_effects.pdf`: River discharge effects
- `effect_heatmap.pdf`: Temporal response patterns

**Tables:**
- `optimal_stations.csv`: Selected stations with effect sizes and p-values
- `window_selection.csv`: AIC values and optimal windows
- `river_results.csv`: River discharge analysis results

**Reports:**
- `station_comparison_report.txt`: Detailed station sensitivity analysis

---

## Data Requirements

### Input Data Format

**1. Trawl survey data** (`hauls.csv`):
```csv
haul_id,port_id,date,latitude,longitude,litter_kg,effort_hours
H001,64100,2019-03-15,41.223333,1.726667,2.5,3.2
...
```

**2. Precipitation data** (`precipitation_STATION.csv`):
```csv
date,precipitation_mm
2019-01-01,0.0
2019-01-02,12.3
...
```

**3. River discharge data** (`discharge_RIVER.csv`):
```csv
date,discharge_m3s
2019-01-01,45.2
2019-01-02,67.8
...
```

---

## Methodology

### Statistical Model

Negative binomial GLM with log link:

```
ln(μ) = β₀ + β₁·P_lag + β₂·sin(2π·week/52) + β₃·cos(2π·week/52) + Σβ_port·Port_i + log(E)
```

Where:
- `μ`: Expected litter mass (kg/hour)
- `P_lag`: Cumulative precipitation over optimal window
- `β₂, β₃`: Seasonal harmonic terms
- `β_port`: Port fixed effects
- `log(E)`: Effort offset

### Window Selection

Optimal precipitation windows selected via:
1. Fit GLM for each window (2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20 weeks)
2. Compare AIC values
3. Apply parsimony rule: when ΔAIC < 2, select shorter window

### Station Selection Criteria

**Hierarchical framework:**
1. **Consistent positive effects**: Select strongest positive station
2. **Consistent negative effects**: Select least negative station  
3. **Mixed signals**: Apply mechanistic criteria:
   - Prioritize urban CSO influence
   - Avoid river-influenced stations
   - Consider statistical robustness (p-values, AIC)

---

## Key Results

### Port Classification

**Type A (CSO delivery, n=7):**
- La Ràpita (+60.5%, 5 weeks)
- L'Ametlla de Mar (+3.4%, 16 weeks)
- Vilanova i la Geltrú (+88.4%, 2 weeks)
- Barcelona (+25.2%, 5 weeks)
- Arenys de Mar (+18.9%, 4 weeks)
- Palamós (+5.8%, 14 weeks)
- Roses (+99.3%, 2 weeks)

**Type B (River washout, n=2):**
- Blanes (-37.1%, 5 weeks)
- Tarragona (-10.9%, 18 weeks)

### River Effects

- **Washout dominant**: 5 of 6 rivers show negative effects
- **Strongest washout**: Roses-Muga (-99.9%)
- **Only positive**: Vilanova-Llobregat (+11.7%)
- **Controls validated**: Ebro shows negligible effects at distant ports

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{Garcia.ea:2026,
  title={Precipitation and river discharge effects on benthic marine litter: 
         Systematic station testing reveals contrasting pathways in the 
         NW Mediterranean},
  author={Garcia, Xavier Garcia and Neukirch, Maik and Iglesias, Jordi and Berdalet, Elisa and Ballabrera-Poy, Joaquim and Galiana, Savitri and Flo, Eva and Cabús, Bru and Ribera-Altimir, Jordi and Blanco, Marta and Galimany, Eve},
  journal={Waste Management},
  year={2026},
  status={In preparation}
}
```

---

## Data Sources

- **Marine litter**: ICATMAR bottom-trawl surveys (Institut Català de Recerca per a la Governança del Mar)
- **Precipitation**: XEMA network (Servei Meteorològic de Catalunya)
- **River discharge**: 
  - ACA (Agència Catalana de l'Aigua)
  - SAIH (Sistema Automático de Información Hidrológica, Confederación Hidrográfica del Ebro)

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Lead Researcher**: Xavier Garcia
**Institution**: Institut de Ciències del Mar (ICM-CSIC), Barcelona, Spain  
**Email**: xgarcia@icm.csic.es

For questions about the methodology or data access, please open an issue or contact the authors directly.

---

## Acknowledgments

- ICATMAR for providing bottom-trawl survey data
- Servei Meteorològic de Catalunya for XEMA precipitation data
- Agència Catalana de l'Aigua and Confederación Hidrográfica del Ebro for river discharge data
- All participating fishing ports and vessels

---

## Version History

- **v1.0.0** (2026-03): Initial release with manuscript submission
  - Multi-station sensitivity analysis
  - Optimal window selection via AIC
  - River discharge effects
  - Litter type partitioning

---

**Last updated**: March 2026
