# Ghana Streetscape Flood Vulnerability AI

AI-based flood vulnerability detection from street-level imagery in Accra, Ghana.

## Overview

This project builds a multimodal GeoAI pipeline that detects flood vulnerability
indicators from Google Street View imagery and integrates them with satellite data,
digital elevation models, drainage networks, and historical flood records to produce
a neighborhood-scale **Street-Level Flood Vulnerability Index (SLFVI)**.

## Study Area

Primary: Accra Metropolitan Area and Greater Accra Region  
Districts: Ga East, Ga West, Ga South, La Dade-Kotopon, Ledzokuku, Krowor, Tema,
Ashaiman, Weija-Gbawe

## Pipeline Phases

| Phase | Description |
|-------|-------------|
| 1 | Scoping, study area definition, and sampling frame |
| 2 | Street View metadata query and imagery catalogue |
| 3 | Image annotation |
| 4 | Computer vision model training |
| 5 | Geospatial feature engineering and fusion |
| 6 | Validation |
| 7 | Policy outputs and dashboard |

## Quick Start

```bash
conda env create -f environment.yml
conda activate streetscape-flood
```

## SLFVI Formula

```
SLFVI = 0.30H + 0.20E + 0.35S + 0.15A
```

- **H** — terrain and rainfall hazard score  
- **E** — population/building/road exposure score  
- **S** — streetscape sensitivity score from AI  
- **A** — inverse adaptive capacity score  

All components normalized to 0–1.

## Attribution

Street View imagery © Google. Usage subject to Google Maps Platform Terms of Service.
Derived features only are stored and redistributed.

## License

Research use. See LICENSE for details.
