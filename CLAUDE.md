# CLAUDE.md

# Project: AI-Based Flood Vulnerability Detection from Streetscapes in Ghana

## 1. Project Summary

This repository implements an AI and geospatial pipeline for detecting flood vulnerability indicators from street-level imagery in Ghana. The project combines Google Street View imagery, Street View metadata, satellite-derived flood proxies, digital elevation models, drainage networks, building footprints, land-cover products, historical flood reports, and machine learning models to produce a neighborhood-scale flood vulnerability index.

The core idea is that flood vulnerability is not only a hydrological phenomenon visible from above. It is also embedded in street-level urban conditions: blocked drains, open gutters, poor road surfaces, low-lying streets, encroached waterways, impervious surfaces, missing drainage infrastructure, informal structures in flood corridors, poor waste management, and limited pedestrian escape routes. These features can be detected from streetscape imagery using computer vision and linked to flood hazard layers derived from satellite and terrain data.

Primary study areas:
- Accra Metropolitan Area
- Ga East, Ga West, Ga South, La Dade-Kotopon, Ledzokuku, Krowor, Tema, Ashaiman, Weija-Gbawe
- Optional expansion to Kumasi, Tamale, Cape Coast, Sekondi-Takoradi, and Ho

Main output:
- Street-Level Flood Vulnerability Index (SLFVI)
- Drainage obstruction map
- Streetscape flood-risk typology
- Ward/neighborhood vulnerability dashboard
- Reproducible GeoAI research pipeline

---

## 2. Main Research Question

Can AI models trained on street-level imagery detect visible urban conditions associated with flood vulnerability in Ghanaian cities, and can these indicators improve conventional satellite- and DEM-based flood exposure mapping?

---

## 3. Specific Objectives

1. Build a georeferenced streetscape image dataset for flood-prone and non-flood-prone urban locations in Ghana.

2. Extract visible flood vulnerability features from streetscape imagery using computer vision models.

3. Integrate streetscape indicators with satellite, DEM, drainage, building, land-cover, and rainfall datasets.

4. Develop a Street-Level Flood Vulnerability Index at road-segment, grid-cell, neighborhood, and municipal scales.

5. Validate the index against historical flood records, local knowledge, social media reports, disaster reports, and observed flood-prone locations.

6. Produce decision-support outputs for local authorities, emergency planners, urban researchers, and climate adaptation actors.

---

## 4. Conceptual Framework

The project is based on the idea that flood vulnerability emerges from the interaction of:

1. Hazard
   - Rainfall intensity
   - Low elevation
   - Poor drainage gradients
   - Proximity to rivers, streams, wetlands, and depressions

2. Exposure
   - Buildings
   - Roads
   - Population
   - Businesses
   - Schools
   - Health facilities
   - Informal settlements

3. Sensitivity
   - Poor drainage
   - Blocked gutters
   - Open drains
   - Impervious surfaces
   - Dense built-up areas
   - Informal construction in flood paths

4. Adaptive capacity
   - Road access
   - Drainage maintenance
   - Solid-waste control
   - Emergency access
   - Local infrastructure quality

Street View imagery is most useful for measuring sensitivity and adaptive capacity because it shows the visible condition of infrastructure at human scale.

---

## 5. Data Sources

### 5.1 Street-Level Data

Use official Google Maps Platform services where permitted.

Recommended access route:
- Street View Metadata API
- Street View Static API

Metadata to store:
- panorama ID
- latitude
- longitude
- capture date
- heading
- field of view
- pitch
- image URL metadata
- copyright metadata
- API response status

Important:
- Do not scrape imagery from Google Maps webpages.
- Use API-based access.
- Keep usage compliant with Google Maps Platform terms, billing, attribution, storage, and display requirements.
- Where redistribution is restricted, store derived features instead of redistributing raw imagery.

Alternative or complementary sources:
- Mapillary, where coverage exists and licensing permits research use
- Locally collected GoPro or smartphone imagery
- Drone imagery where approved
- Municipal drainage inspection photos
- Community-based geo-tagged photographs

### 5.2 Hydrological and Terrain Data

Recommended datasets:
- FABDEM or Copernicus DEM
- MERIT Hydro
- HydroSHEDS
- JRC Global Surface Water
- Sentinel-1 flood extent products
- CHIRPS rainfall
- IMERG precipitation
- ERA5-Land rainfall and soil moisture
- Google FloodHub or Flood Forecasting API where available
- Ghana Meteorological Agency rainfall stations, if accessible
- Ghana Hydrological Authority or WRC river gauge data, if accessible

### 5.3 Urban Exposure Data

Recommended datasets:
- Google Open Buildings
- Microsoft Building Footprints
- OpenStreetMap roads, waterways, drains, bridges, schools, hospitals, markets
- Overture Maps building and road layers
- WorldPop population
- Ghana 2021 Population and Housing Census
- Nighttime lights
- GHSL built-up layers
- ESA WorldCover
- Dynamic World
- Landsat/Sentinel impervious surface proxies

### 5.4 Historical Flood and Validation Data

Recommended validation sources:
- NADMO flood reports
- News archives
- Ghana Meteorological Agency warnings
- Google FloodHub events where relevant
- Social media reports, used carefully and ethically
- Community flood mapping
- Municipal assembly records
- Drainage desilting records
- Field validation at selected hotspots
- Existing academic flood inventories for Accra

---

## 6. Sampling Design

### 6.1 Spatial Sampling Frame

1. Define study boundary:
   - Greater Accra Region or Accra Metropolitan Area

2. Build a road-segment dataset:
   - Download OSM roads
   - Clean road classes
   - Split roads into 50 m or 100 m segments
   - Generate sample points at segment centroids or regular intervals

3. Stratify by expected flood exposure:
   - High-risk flood depressions
   - Areas within 50 m or 100 m of drains/waterways
   - Low-lying areas
   - Historical flood locations
   - Non-flood reference areas

4. Add settlement type:
   - Formal residential
   - Informal settlement
   - Commercial district
   - Industrial area
   - Peri-urban expansion zone
   - Coastal settlement
   - Wetland-adjacent neighborhood

### 6.2 Street View Request Design

For each sample point:
- Query Street View metadata
- Retain only points with valid coverage
- Request four directional images:
  - heading 0
  - heading 90
  - heading 180
  - heading 270
- Optional additional headings:
  - road direction heading
  - opposite road direction heading
- Use consistent:
  - pitch = 0
  - field of view = 80 or 90
  - image size based on API limits and budget

### 6.3 Suggested Sample Size

Pilot:
- 5,000 road points
- 20,000 images using four headings

Full Accra study:
- 25,000 road points
- 100,000 images

National comparative study:
- 100,000 road points
- 400,000 images

---

## 7. Annotation Strategy

### 7.1 Target Labels

Create labels at image level, object level, and segment level.

Image-level labels:
- visible drain present
- open gutter present
- blocked drain present
- stagnant water visible
- poor road condition
- heavy impervious surface
- unpaved shoulder
- informal structure near drainage
- solid waste accumulation
- visible waterway or stream
- low-lying street form
- roadside erosion
- pedestrian exposure
- culvert or bridge visible
- no visible drainage

Object-level labels:
- drain
- gutter
- culvert
- water
- solid waste
- road
- sidewalk
- vegetation
- building
- kiosk/container
- bridge
- stream/channel

Segment-level labels:
- low flood vulnerability
- moderate flood vulnerability
- high flood vulnerability
- uncertain or requires field check

### 7.2 Annotation Tools

Recommended:
- CVAT
- Label Studio
- Roboflow
- QGIS annotation plugin
- GeoJSON-based review interface

### 7.3 Annotation Protocol

1. Develop a visual codebook with Ghana-specific examples.
2. Train annotators using 300 pilot images.
3. Double-code at least 15 percent of images.
4. Calculate inter-annotator agreement.
5. Review disagreements.
6. Use active learning to prioritize uncertain model predictions.
7. Use field photos to refine labels for confusing classes.

---

## 8. Computer Vision Models

### 8.1 Baseline Models

Image classification:
- ResNet50
- EfficientNet
- ConvNeXt

Semantic segmentation:
- DeepLabV3+
- U-Net
- SegFormer

Object detection:
- YOLOv8 or YOLOv10
- Faster R-CNN
- Grounding DINO

Foundation model assistance:
- Segment Anything Model
- CLIP
- DINOv2
- Grounded SAM

### 8.2 Recommended Model Stack

Use a hybrid model design:

1. Object detection model
   - Detects drains, gutters, solid waste, culverts, streams, potholes, erosion, roadside water.

2. Semantic segmentation model
   - Estimates percentage of visible surface types:
     - road
     - vegetation
     - water
     - building
     - drain
     - bare ground
     - waste
     - sidewalk

3. Vision-language model
   - Generates structured scene descriptions.
   - Example prompt:
     "Describe visible drainage, road quality, waste accumulation, and likely flood-related infrastructure problems in this image."

4. Risk classifier
   - Predicts low, moderate, or high streetscape vulnerability.

5. Multimodal fusion model
   - Combines streetscape features with satellite, DEM, rainfall, population, and building data.

---

## 9. Feature Engineering

### 9.1 Streetscape Features

For each road segment or image cluster, calculate:

Drainage indicators:
- drain presence probability
- blocked drain probability
- open gutter probability
- culvert presence probability
- visible water channel probability
- stagnant water probability

Surface indicators:
- impervious surface share
- unpaved surface share
- poor road-condition score
- pothole/erosion score

Waste indicators:
- solid waste presence probability
- waste volume proxy from segmentation area
- waste near drain indicator

Built-form indicators:
- building density proxy from visible building share
- informal structure probability
- kiosk/container density proxy
- encroachment near drainage probability

Vegetation indicators:
- vegetation share
- tree canopy proxy
- exposed soil share

Accessibility indicators:
- road width proxy
- sidewalk presence
- pedestrian escape/access route proxy

### 9.2 Remote Sensing and Terrain Features

For each sample point or grid cell:
- elevation
- slope
- topographic wetness index
- distance to stream
- distance to drain
- distance to wetland
- distance to coastline
- flow accumulation
- depression depth
- rainfall intensity
- historical surface-water occurrence
- built-up density
- impervious surface proxy
- building count
- population density

### 9.3 Spatial Aggregation

Aggregate features at:
- road segment
- 100 m grid cell
- 250 m grid cell
- electoral area
- community
- municipality

Recommended spatial unit for modeling:
- 100 m grid for high-resolution vulnerability mapping
- municipality or electoral area for policy summary outputs

---

## 10. Modeling Strategy

### 10.1 Baseline Statistical Model

Use logistic regression or random forest to predict whether a location is historically flood affected.

Input:
- DEM-based variables
- rainfall variables
- distance to drainage
- building density
- streetscape features

Output:
- flood-prone probability

### 10.2 Machine Learning Models

Recommended:
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- TabNet

### 10.3 Deep Learning Fusion

Recommended architecture:

1. Image encoder:
   - Vision Transformer or ConvNeXt

2. Spatial feature encoder:
   - Multilayer perceptron for tabular geospatial variables

3. Fusion layer:
   - concatenate image embedding and geospatial embedding

4. Output heads:
   - flood vulnerability score
   - feature-specific risks
   - uncertainty score

### 10.4 Spatial Model

Recommended:
- Graph neural network over road network
- Spatial lag model
- Bayesian hierarchical model
- Geographically weighted random forest

Purpose:
- capture spatial dependence between adjacent road segments and drainage-connected neighborhoods.

---

## 11. Street-Level Flood Vulnerability Index

### 11.1 Index Components

SLFVI = f(Hazard, Exposure, Streetscape Sensitivity, Adaptive Capacity)

Suggested weighting:
- Hazard: 30 percent
- Exposure: 20 percent
- Streetscape sensitivity: 35 percent
- Adaptive capacity: 15 percent

Weights may be:
- expert-defined
- learned through supervised modeling
- estimated using PCA
- calibrated against historical flood impacts

### 11.2 Example Index Formula

SLFVI = 0.30H + 0.20E + 0.35S + 0.15A

Where:
- H = terrain and rainfall hazard score
- E = population/building/road exposure score
- S = streetscape sensitivity score from AI
- A = inverse adaptive capacity score

Normalize all components to 0–1.

### 11.3 Classification

Suggested categories:
- 0.00–0.20: very low vulnerability
- 0.21–0.40: low vulnerability
- 0.41–0.60: moderate vulnerability
- 0.61–0.80: high vulnerability
- 0.81–1.00: very high vulnerability

---

## 12. Validation and Evaluation

### 12.1 Computer Vision Evaluation

Metrics:
- precision
- recall
- F1 score
- mean average precision
- intersection over union
- confusion matrix

Evaluate separately for:
- drain detection
- blocked drain detection
- solid waste detection
- water/stagnation detection
- poor road condition classification

### 12.2 Flood Model Evaluation

Metrics:
- ROC-AUC
- PR-AUC
- Brier score
- calibration curve
- spatial cross-validation accuracy
- recall at top 10 percent risk locations

Validation targets:
- known flood hotspots
- reported flood events
- field-verified flood-prone locations
- drainage maintenance records
- citizen reports

### 12.3 Spatial Validation

Use:
- block spatial cross-validation
- leave-one-municipality-out validation
- distance-based train/test splits

This is important to avoid overestimating performance due to spatial autocorrelation.

---

## 13. Explainability

Use:
- SHAP values for tabular models
- Grad-CAM for CNN models
- attention maps for transformer models
- object-level risk explanation
- local risk reports for each road segment

Example explanation:
"This road segment is classified as high vulnerability because the model detected blocked open drains, high impervious surface coverage, poor road condition, low elevation, and proximity to a historical flood depression."

---

## 14. Ethics, Privacy, and Governance

Key principles:
- Use official and permitted data access routes.
- Do not identify individuals.
- Do not publish recognizable faces, vehicle plates, or private properties unnecessarily.
- Store only derived features where possible.
- Follow platform terms for imagery use, storage, and display.
- Avoid neighborhood stigmatization.
- Validate model outputs with local experts.
- Present vulnerability as infrastructural risk, not as blame on residents.
- Include uncertainty in all public maps.

---

## 15. Repository Structure

Recommended repository layout:

```text
ghana-streetscape-flood-ai/
├── CLAUDE.md
├── README.md
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── external/
│   └── validation/
├── notebooks/
│   ├── 01_sampling_design.ipynb
│   ├── 02_streetview_metadata.ipynb
│   ├── 03_image_quality_control.ipynb
│   ├── 04_annotation_review.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_geospatial_fusion.ipynb
│   └── 07_vulnerability_mapping.ipynb
├── src/
│   ├── config/
│   ├── data/
│   ├── imagery/
│   ├── annotation/
│   ├── models/
│   ├── features/
│   ├── geospatial/
│   ├── evaluation/
│   ├── visualization/
│   └── utils/
├── configs/
│   ├── accra_pilot.yaml
│   ├── full_accra.yaml
│   └── national_expansion.yaml
├── models/
│   ├── detection/
│   ├── segmentation/
│   ├── classification/
│   └── fusion/
├── outputs/
│   ├── maps/
│   ├── figures/
│   ├── tables/
│   ├── dashboards/
│   └── reports/
├── tests/
├── environment.yml
├── requirements.txt
└── pyproject.toml
```

---

## 16. Implementation Milestones

### Phase 1: Scoping and Design

Tasks:
- define study area
- collect flood hotspot inventory
- download roads, buildings, DEM, hydrography, and rainfall data
- design road-segment sampling grid
- prepare ethics and data governance protocol

Deliverables:
- study boundary
- sampling frame
- data inventory
- annotation codebook draft

### Phase 2: Street View Metadata and Imagery

Tasks:
- query Street View metadata for sample points
- filter valid panoramas
- collect images using API-compliant methods
- store metadata and image references
- perform image quality control

Deliverables:
- streetscape image catalogue
- metadata table
- panorama coverage map

### Phase 3: Annotation

Tasks:
- prepare pilot annotation set
- train annotators
- annotate drainage, waste, water, road condition, and infrastructure labels
- review agreement
- refine codebook

Deliverables:
- labeled pilot dataset
- final annotation codebook
- benchmark dataset

### Phase 4: Computer Vision Model Development

Tasks:
- train baseline classifiers
- train object detection model
- train segmentation model
- evaluate performance
- run active learning loop

Deliverables:
- trained models
- prediction tables
- model performance report

### Phase 5: Geospatial Fusion

Tasks:
- calculate DEM and hydrological features
- derive satellite-based flood proxies
- join streetscape predictions to road segments
- aggregate features to grid cells and neighborhoods
- train vulnerability prediction models

Deliverables:
- fused geospatial dataset
- Street-Level Flood Vulnerability Index
- model comparison table

### Phase 6: Validation

Tasks:
- compare results to known flood hotspots
- conduct field checks for selected locations
- consult local experts
- test sensitivity to image date, road type, and sampling density
- perform spatial cross-validation

Deliverables:
- validation report
- uncertainty map
- corrected final index

### Phase 7: Policy Outputs

Tasks:
- produce hotspot maps
- create municipal risk profiles
- generate ranked drainage maintenance priorities
- develop interactive dashboard
- prepare manuscript and policy brief

Deliverables:
- web dashboard
- academic paper
- policy brief
- municipal hotspot maps
- GitHub repository

---

## 17. Suggested Python Stack

Core:
- Python
- GeoPandas
- Pandas
- NumPy
- Rasterio
- Xarray
- rioxarray
- PyProj
- Shapely
- OSMnx
- NetworkX

Remote sensing:
- Google Earth Engine
- geemap
- pystac-client
- planetary-computer
- sentinelhub, if available

Computer vision:
- PyTorch
- torchvision
- ultralytics
- transformers
- segment-anything
- timm
- scikit-image
- OpenCV

Machine learning:
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- SHAP

Visualization:
- Matplotlib
- Plotly
- Kepler.gl
- Folium
- Streamlit
- Dash
- Lonboard

Data validation:
- Great Expectations
- pandera
- pytest

Experiment tracking:
- MLflow
- Weights & Biases, optional

---

## 18. Key Tables to Produce

1. Street View metadata table
2. Sample-point coverage table
3. Annotation class distribution table
4. Computer vision performance table
5. Geospatial feature summary table
6. Flood validation dataset table
7. Model comparison table
8. Municipality vulnerability ranking table
9. Top 100 drainage maintenance priority road segments
10. Sensitivity analysis table

---

## 19. Key Figures to Produce

1. Study area map
2. Street View coverage map
3. Sampling design map
4. Example streetscape risk classes
5. Computer vision detection examples
6. Drainage obstruction probability map
7. DEM-derived flood depression map
8. Streetscape vulnerability map
9. Integrated flood vulnerability map
10. Validation map against historical flood hotspots
11. SHAP feature importance plot
12. Municipal vulnerability profile figure

---

## 20. Manuscript Outline

Suggested title:
AI-Based Flood Vulnerability Detection from Streetscapes: A Multimodal GeoAI Framework for Urban Ghana

Abstract:
- introduce problem
- explain streetscape opportunity
- describe multimodal data
- summarize AI methods
- present vulnerability index
- state policy relevance

Sections:
1. Introduction
2. Flood Vulnerability, Street-Level AI, and Urban Infrastructure
3. Study Area and Data
4. Methods
5. Results
6. Validation and Sensitivity Analysis
7. Discussion
8. Policy Implications
9. Limitations
10. Conclusion

Potential journals:
- International Journal of Disaster Risk Reduction
- Journal of Flood Risk Management
- Computers, Environment and Urban Systems
- Sustainable Cities and Society
- Landscape and Urban Planning
- ISPRS International Journal of Geo-Information
- Urban Climate
- Natural Hazards

---

## 21. Risks and Mitigation

Risk: Street View coverage gaps.
Mitigation: use metadata API first; supplement with Mapillary or local field imagery.

Risk: API cost.
Mitigation: begin with pilot sample; use stratified sampling; store derived features.

Risk: image dates vary.
Mitigation: include capture date as a model feature; run sensitivity analysis.

Risk: blocked drains are hard to label.
Mitigation: use local annotation codebook and active learning.

Risk: model overfits to neighborhood appearance.
Mitigation: use spatial cross-validation and leave-one-municipality-out tests.

Risk: ethical concern over stigmatizing informal areas.
Mitigation: frame results as infrastructural vulnerability and governance priority.

---

## 22. Minimum Viable Product

The MVP should deliver:

1. Accra pilot study with 5,000 road points.
2. Street View metadata coverage map.
3. 20,000 images across four headings.
4. Annotation of 2,000 images.
5. Baseline classifier for visible drainage and solid waste risk.
6. DEM-based flood depression overlay.
7. Road-segment vulnerability index.
8. Static map and short policy brief.

---

## 23. Commands for Claude Code

Use these tasks sequentially.

### Task 1: Create Repository Skeleton

Create the full repository structure listed in Section 15. Add placeholder README, requirements, config files, and src modules.

### Task 2: Build Sampling Frame

Write Python scripts to:
- download OSM roads for Greater Accra
- clean road classes
- split roads into 100 m segments
- generate sample points
- export sample points as GeoPackage and CSV

### Task 3: Query Street View Metadata

Write a compliant Street View metadata query module that:
- reads sample points
- queries metadata endpoint
- handles API errors
- stores panorama ID, image date, lat/lon, status, and copyright
- avoids duplicate panorama requests
- writes metadata to Parquet and GeoPackage

### Task 4: Prepare Imagery Catalogue

Write a script to:
- generate image request records for headings 0, 90, 180, and 270
- avoid downloading invalid records
- track API usage
- store image metadata
- create a reproducible image manifest

### Task 5: Build Annotation Dataset

Write a script to:
- sample images for annotation
- stratify by municipality, flood-risk class, road type, and image date
- export to CVAT or Label Studio format
- create annotation schema from the labels in Section 7

### Task 6: Train Baseline Models

Write training scripts for:
- image-level classification
- object detection
- semantic segmentation
- model evaluation
- confusion matrix and metrics export

### Task 7: Generate Streetscape Features

Write inference scripts to:
- run trained models over all images
- aggregate predictions by road segment
- compute streetscape vulnerability features
- export feature tables

### Task 8: Build Geospatial Features

Write geospatial processing scripts to:
- calculate elevation, slope, TWI, flow accumulation, depression depth
- calculate distance to drains, waterways, wetlands, and coastline
- join building and population exposure
- export road-segment and grid-cell features

### Task 9: Train Fusion Model

Write model scripts to:
- combine streetscape and geospatial features
- train random forest, XGBoost, and LightGBM models
- run spatial cross-validation
- evaluate against historical flood locations
- compute uncertainty

### Task 10: Produce Outputs

Write scripts to:
- generate vulnerability maps
- rank road segments for drainage maintenance
- produce municipal summaries
- create dashboard-ready GeoJSON
- export manuscript-ready figures and tables

---

## 24. Expected Final Deliverables

- Street-level flood vulnerability dataset
- Ghana streetscape image metadata catalogue
- Annotated computer vision benchmark dataset
- Trained AI models
- Street-Level Flood Vulnerability Index
- Drainage obstruction probability map
- Flood vulnerability dashboard
- Municipal maintenance priority list
- Academic manuscript
- Policy brief for NADMO and metropolitan assemblies

---

## 25. Core Principle

The project should not treat streetscape imagery as a decorative data source. It should use street-level AI to detect the everyday infrastructural conditions that make urban flooding more likely, more damaging, and harder to manage. The scientific contribution is the integration of human-scale urban evidence with satellite and hydrological flood modeling.
