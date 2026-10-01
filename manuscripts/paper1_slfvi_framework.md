# Detecting Flood Vulnerability from Urban Streetscapes: A Multimodal GeoAI Framework for Accra, Ghana

**Authors:** [Author names]

**Affiliations:** [Affiliations]

**Corresponding author:** dzimike.md@gmail.com

**Journal target:** International Journal of Disaster Risk Reduction / Computers, Environment and Urban Systems

**Keywords:** flood vulnerability, street-level imagery, GeoAI, deep learning, Ghana, urban drainage, EfficientNet, vulnerability index

---

## Abstract

Urban flooding in sub-Saharan African cities causes recurring loss of life, property damage, and economic disruption, yet conventional flood risk models based on digital elevation models and satellite data miss the street-level infrastructural conditions that mediate actual flood impact. This study presents a multimodal GeoAI framework that integrates street-level imagery, terrain analysis, satellite-derived exposure data, and machine learning to produce a Street-Level Flood Vulnerability Index (SLFVI) at road-segment resolution across Greater Accra, Ghana. A stratified sample of 5,000 road points yielded 15,344 Google Street View images at 3,836 locations, supplemented by 590 Mapillary images at coverage gap points. An EfficientNet-B0 model fine-tuned on 400 locally annotated images and applied to a 2,000-image hybrid training dataset extracted 15 flood-indicator features per image across all 15,934 images, including visible drainage obstruction, solid waste accumulation, impervious surface coverage, and poor road condition. Inter-annotator agreement on a 60-image double-coded subset yielded a mean Fleiss' κ of 0.51 across the 15 binary labels (range −0.01 to 1.00), with the highest agreement for unambiguous infrastructure labels (poor_road_condition κ = 0.83, open_gutter_present κ = 0.73) and the lowest for rare or boundary-condition labels. The analytical SLFVI achieves a ROC-AUC of 0.701 discriminating DEM-derived flood-exposed from non-exposed road locations; a streetscape-only model achieves ROC-AUC 0.744 in isolation, representing a non-terrain signal from street-level infrastructure conditions. A LightGBM fusion model achieves ROC-AUC 0.996 and precision-recall AUC 0.946 under standard cross-validation (0.987 and 0.778 under spatial block cross-validation); however, terrain features alone achieve ROC-AUC 0.989 against the same DEM-derived reference, reflecting shared methodological ancestry between model inputs and the validation target that limits interpretation of high fusion performance as genuine independent validation. A model ablation confirms that the streetscape contribution is orthogonal to topographic flood exposure — detecting drainage-failure vulnerability rather than inundation depth — consistent with the theoretical basis of this framework. Critically, a supplementary Mapillary query of 1,088 Street View gap points reveals that areas without Google Street View coverage carry systematically higher vulnerability scores (mean EfficientNet probability 0.580) than covered areas (mean 0.384, Cohen's d = 0.89), a coverage bias that remote-sensing-only models cannot detect. The framework delivers ranked drainage maintenance priorities, district vulnerability profiles, and a continuous 250-metre interpolated vulnerability surface. The SLFVI primarily captures infrastructural flood vulnerability — drainage condition, surface type, and built-form characteristics — rather than broader social vulnerability dimensions such as socioeconomic exposure, tenure security, or institutional response capacity. Independent field validation against observed flood events remains a priority for future work.

---

## 1. Introduction

Accra, the capital of Ghana, experiences annual flooding that affects hundreds of thousands of residents, damages critical infrastructure, and strains emergency response capacity (Amoako and Frimpong Boamah, 2015; Douglas et al., 2008). The June 2015 flooding and fuel station explosion at Circle, which killed more than 150 people, exemplified the catastrophic convergence of poor drainage, informal settlement encroachment on flood corridors, and inadequate emergency infrastructure (Cutter et al., 2003; Wisner et al., 2004). Despite decades of investment in drainage infrastructure, flooding in Accra remains persistent and spatially concentrated in identifiable neighbourhoods, suggesting that the physical and infrastructural conditions that produce vulnerability are detectable and addressable if mapped at sufficient resolution.

Conventional flood risk models rely primarily on digital elevation models (DEMs), satellite rainfall products, and hydrological routing algorithms to identify areas likely to be inundated (Tehrany et al., 2015; Khosravi et al., 2016). These approaches are effective at identifying broad flood depressions and riparian zones but systematically underestimate vulnerability in flat urban areas where drainage failure, blocked gutters, solid waste accumulation, and informal construction are the proximate causes of flooding rather than topographic position alone (Vojinovic and Abbott, 2012; Schumann et al., 2013). In dense informal settlements, a single blocked culvert or an impervious surface replacing an open gutter can render a street flood-prone even at moderate elevations. This gap between satellite-observable flood hazard and street-level vulnerability has received limited methodological attention in low- and middle-income country contexts where field survey resources are constrained.

The emergence of street-level imagery at city scale, combined with advances in computer vision and geospatial machine learning, creates an opportunity to close this gap. Street View imagery has been used to assess urban greenery, building condition, pedestrian environment quality, and socioeconomic characteristics in high-income city contexts (Biljecki and Ito, 2021; Naik et al., 2017; Zhou et al., 2017). In flood-relevant applications, street-level images contain visible signals of drainage quality, waste accumulation, surface imperviousness, and infrastructure condition that are invisible from above. This study asks whether these signals, extracted at scale using deep learning, can improve flood vulnerability mapping beyond what terrain and satellite data alone can deliver.

This paper makes four specific contributions. First, it presents a georeferenced street-level image dataset for Greater Accra stratified by flood risk stratum, settlement type, and road class. Second, it introduces a multi-task EfficientNet-B0 classifier trained on locally annotated images that jointly predicts binary flood vulnerability and 15 individual flood-indicator labels per image. Third, it constructs a Street-Level Flood Vulnerability Index by fusing streetscape AI predictions with terrain, hydrological, building, and population features using gradient-boosted trees, evaluated against a DEM-derived building flood exposure reference and with the methodological caveats of that reference explicitly disclosed. Fourth, it demonstrates that street-level imagery coverage gaps are spatially non-random: the neighbourhoods where Google Street View does not reach are systematically more vulnerable than those it covers, a bias that has direct implications for how urban AI tools should be interpreted in the Global South.

---

## 2. Background

### 2.1 Flood Vulnerability in West African Cities

Urban flooding in West Africa is driven by a compound of rapid urbanisation, inadequate drainage infrastructure, climate intensification, and governance fragmentation (Douglas et al., 2008; Schipper and Pelling, 2006). In Accra, the physical geography presents particular challenges. The city occupies a coastal plain dissected by the Densu, Odaw, and Sakumo drainage basins, all of which are subject to intense seasonal rainfall concentrated in April-July and again in September-November. Runoff coefficients are elevated by high impervious surface coverage in central districts, informal settlement expansion into natural drainage corridors, and chronic solid waste accumulation in open gutters and culverts (Amoako and Frimpong Boamah, 2015). Historical flood records compiled by the National Disaster Management Organisation (NADMO) confirm that flood events are spatially recurrent, concentrating in low-lying areas of the Odaw basin, coastal settlements in the La and Krowor districts, and peri-urban informal settlements in the Weija-Gbawe and Ablekuma corridors.

Vulnerability in these contexts is not reducible to topographic exposure. Wisner et al. (2004) define vulnerability as the characteristics and circumstances of a community, system, or asset that make it susceptible to the damaging effects of a hazard. Applied to urban drainage, this encompasses the physical condition of drainage infrastructure (whether gutters are open, blocked, absent, or encroached upon), the governance of solid waste removal, the capacity of roads to serve as emergency access routes, and the material quality of buildings in the flood path. None of these dimensions are directly observable from satellite imagery at operational resolution, which is why DEM-based flood models persistently underestimate vulnerability in informal urban areas (Tehrany et al., 2015).

Institutional fragmentation is a structural feature of Accra's flood governance landscape that shapes how these physical conditions develop and persist. Responsibility for drainage and road infrastructure is distributed across multiple agencies without a unified mandate: the Metropolitan and Municipal Assemblies manage local drains and street surfaces within their jurisdictions; the Ghana Highways Authority oversees drainage along national and inter-regional roads; the Ghana Urban Roads Authority manages classified urban arterials; and the Hydrological Services Department of the Ministry of Works and Housing holds responsibility for major flood control channels and watercourses. These boundaries do not correspond to drainage catchment boundaries, meaning that a connected drainage path often crosses multiple jurisdictional areas. Inconsistent maintenance scheduling, budgetary fragmentation, and the absence of a coordinating body with operational authority across boundaries mean that interventions in one district routinely fail because a blockage or collapsed culvert in an adjacent district remains unaddressed. For flood vulnerability mapping, this fragmentation has a direct analytical implication: district-level summaries of the SLFVI are useful for resource prioritisation within each assembly's remit, but high-vulnerability corridor segments that cross district lines require cross-jurisdictional coordination that is not reflected in any single assembly's planning cycle.

### 2.2 Street-Level Imagery in Urban Analysis

Street View imagery from Google, Mapillary, and other platforms provides a systematically collected and georeferenced archive of urban environments at human scale. Biljecki and Ito (2021) survey more than 300 published applications spanning urban planning, environmental quality assessment, disaster risk management, transportation, and public health, concluding that street-level imagery has become a foundational data source for urban research. In the Global South, coverage is uneven but growing, and the methodology of combining sparse street-level imagery with satellite data through spatial interpolation or multi-modal fusion is increasingly established (Srivastava et al., 2019).

For flood-relevant applications, street-level imagery has been used to characterise neighbourhood physical conditions that correlate with flood impact (Naik et al., 2017), assess post-disaster damage in combination with satellite data (Srivastava et al., 2019), and monitor urban drainage infrastructure condition. The novelty of the present study lies in applying this approach prospectively in a West African context, using pre-event street-level conditions to predict vulnerability rather than to document damage after the fact, and in directly confronting the coverage inequality that makes such approaches systematically biased in informal city environments.

The published literature on street-level AI for flood vulnerability is small and geographically concentrated in Latin America and the United States. Velez et al. (2022) extracted flood-vulnerability variables from Google Street View in Quito, Ecuador, using a machine-learning model validated against expert survey data — the first published study to construct a street-view-derived flood vulnerability index in the Global South. Wang et al. (2024) applied deep learning to street-level imagery for climate disaster vulnerability assessment in Peru and Colombia at the building scale. Gao et al. (2024) used deep learning and Google Street View to estimate first-floor elevations for flood risk assessment along the Texas coast, validating against FEMA Flood Insurance Rate Maps. These three studies collectively establish the emerging methodological paradigm, but all focus on Latin American or high-income contexts, use building elevation or structural condition as the primary signal, and none address the systematic coverage gap that characterises street-level data in informal settlement areas. The present study is, to our knowledge, the first to apply street-level imagery AI to pre-event flood vulnerability mapping in sub-Saharan Africa, to target drainage infrastructure conditions rather than building attributes, and to quantify the spatial inequality in coverage as a structural bias in the methodology itself. A parallel study by Iordanov (2025) uses CLIP-based zero-shot classification and Mapillary imagery for multi-hazard infrastructure vulnerability prediction in a multi-city context, but does not address drainage-specific indicators, spatial cross-validation, or coverage bias in the Global South. Within the Accra-specific literature, the dominant methodological tradition uses GIS-based multi-criteria analysis and analytical hierarchy process models without street-level inputs (Dekongmen et al., 2021; Nkonu et al., 2023); the present study introduces street-level AI as a new evidence layer that is orthogonal to — and complementary with — those terrain-based approaches.

### 2.3 Deep Learning for Urban Feature Extraction

Convolutional neural networks have demonstrated strong performance in classifying urban scene attributes from street-level imagery. He et al. (2016) introduced residual connections that enabled deep networks to be trained effectively on large image datasets, establishing the architecture family from which EfficientNet (Tan and Le, 2019) descends. EfficientNet uses a compound scaling strategy that simultaneously optimises network depth, width, and input resolution according to a fixed scaling coefficient, delivering state-of-the-art accuracy at lower computational cost than earlier architectures. Radford et al. (2021) demonstrated that vision-language contrastive pre-training (CLIP) enables zero-shot scene classification with broad semantic coverage, making it useful as a baseline when labelled training data are scarce.

In the present study, both CLIP (as a zero-shot baseline) and a fine-tuned EfficientNet-B0 (as the primary classifier) are used to extract flood-indicator features from street-level images. The EfficientNet model is trained on locally annotated images from Accra, which is methodologically important because drainage infrastructure in Ghanaian cities (open concrete gutters, earthen channels, informal culverts) differs substantially in appearance from the infrastructure contexts on which pre-trained models were developed.

### 2.4 Geospatial Machine Learning for Flood Vulnerability

Gradient-boosted tree models, particularly XGBoost (Chen and Guestrin, 2016) and LightGBM (Ke et al., 2017), have become dominant approaches for tabular geospatial prediction tasks including flood susceptibility mapping, where they consistently outperform logistic regression and random forest baselines while remaining interpretable through SHAP decomposition (Lundberg and Lee, 2017). The integration of remote sensing features, terrain indices, and infrastructure indicators in a unified tabular prediction framework is well established (Tehrany et al., 2015; Khosravi et al., 2016). The contribution of this study is to introduce streetscape AI predictions as a new feature class in this framework and to demonstrate that these features carry information independent of terrain and satellite variables.

---

## 3. Study Area and Data

### 3.1 Study Area

The study covers 21 municipal and metropolitan districts of Greater Accra, Ghana, spanning approximately 2,500 square kilometres of coastal plain, river basin, and peri-urban expansion zone between latitudes 5.4 and 5.8 degrees north and longitudes 0.5 west and 0.1 east. The study area includes the Accra Metropolitan Assembly, Tema Metropolitan Assembly, and the surrounding municipal assemblies of Ablekuma Central, Ablekuma West, Ablekuma North, Ayawaso North, Ayawaso Central, Ayawaso East, Ayawaso West, Ga East, Ga West, Ga Central, Ga North, Ga South, Ledzokuku, Krowor, La Dade-Kotopon, La-Nkwantanang-Madina, Weija-Gbawe, Korle-Klottey, Adenta, and Tema West. According to the Ghana Statistical Service (2021), Greater Accra Region had a population of approximately 5.4 million in 2021, making it the most densely populated administrative region in the country. Throughout this paper, the term 'Greater Accra' refers to this 21-district analytical boundary; the administrative Greater Accra Region encompasses additional districts beyond this study area that are not included in the analysis.

Four districts were designated as focus areas for high-resolution analysis based on documented flood history and Street View coverage density: Ablekuma Central, Ledzokuku, Ga East, and La Dade-Kotopon. These districts span a range of settlement types from dense inner-city informal areas (Ablekuma Central, Ledzokuku) to planned peri-urban zones (Ga East) and mixed coastal-inland neighbourhoods (La Dade-Kotopon).

### 3.2 Street-Level Imagery

Street View imagery was accessed through the Google Maps Platform Street View Static API and Street View Metadata API using a paid API key. A stratified random sample of 5,000 road points was generated from the OpenStreetMap road network for Greater Accra (Section 4.1). For each sample point, the metadata endpoint was queried with a search radius of 50 metres to determine panorama availability. Points with valid coverage returned a unique panorama identifier, image capture date, and the precise latitude and longitude of the nearest panorama. Of the 5,000 queried points, 3,912 returned status OK and 1,086 returned ZERO_RESULTS, with 2 returning UNKNOWN_ERROR. After metadata filtering, 3,836 unique points with confirmed coverage were retained, yielding 15,344 images across four directional headings (0, 90, 180, and 270 degrees) at a consistent pitch of zero degrees and a field of view of 90 degrees. Raw images were used ephemerally for model training and feature extraction and are not stored, redistributed, or included in the data release; only derived streetscape feature vectors are released. A note on terms of service compliance is provided in Section 6.4.

Mapillary imagery was queried for all 1,088 gap points using the Mapillary Graph API v4.0 with a search radius of 100 metres. The query returned at least one image for 159 of the 1,088 gap points (coverage rate 14.6 percent), yielding 620 images in total. Of these, 30 were excluded after quality screening — flagged for motion blur (Laplacian variance below threshold), extreme pitch (|pitch| > 30°), or heading inconsistency exceeding 45° relative to the road direction — leaving 590 images that were processed through the same EfficientNet inference pipeline as the Street View images.

### 3.3 Terrain and Hydrological Data

Elevation data were sourced from the MERIT DEM (Yamazaki et al., 2017), a void-filled, error-corrected global terrain dataset derived from SRTM and AW3D30. The MERIT DEM provides approximately 90-metre horizontal resolution and is considered suitable for urban hydrological analysis in low-relief coastal settings where elevation accuracy is critical. Slope, aspect, and topographic wetness index (TWI) were derived from the MERIT DEM in UTM Zone 30N projection (EPSG 32630). Flow accumulation and stream network extraction used the D8 single-direction flow algorithm. Distances to the nearest OSM-mapped waterway, drain, and coastline were computed for each sample point using Euclidean proximity in projected coordinates.

### 3.4 Exposure Data

Building footprints were sourced from Google Open Buildings, which provides a continental-scale dataset of automatically detected building outlines for Africa at high spatial resolution (Sirko et al., 2021). For the study area, 239,086 building footprints were extracted and attributed with elevation from the MERIT DEM and population estimates derived from WorldPop gridded population data (WorldPop, 2020). Flood exposure status was assigned to each building using a binary indicator derived from DEM-based inundation thresholds, identifying 14,662 buildings (6.1 percent) as flood-exposed based on their elevation relative to modelled inundation surfaces. This layer was used as a validation reference for the SLFVI (hereafter 'DEM-derived reference layer'). An important methodological caveat is that the validation target is constructed from the same MERIT DEM that also supplies the elevation, slope, TWI, depression depth, and flow accumulation inputs to the fusion model. The overlap in methodological ancestry between model inputs and the validation reference means that high discriminative performance against this target is partly expected and should not be equated with validation against observed flood events. Genuinely independent validation against NADMO flood records, field-verified hotspots, or drainage complaint data remains a priority for future work and is acknowledged as a current limitation in Sections 5.5 and 6.4.

Road network data were downloaded from OpenStreetMap using OSMnx (Boeing, 2017), covering all road classes from motorways to residential and unclassified tracks. The network comprised 70,774 road segments after cleaning and simplification, from which the 5,000 sample points were drawn.

### 3.5 Independent Flood Event Data (FloodSpots)

To supplement the DEM-derived validation reference with event-based evidence that is fully independent of model inputs, this study uses location data from FloodSpots (Ghana Ground Up, 2026), a publicly accessible community and media flood reporting platform for Greater Accra. FloodSpots maps verified flood hotspots compiled from published news reporting (AP, JoyNews, B&FT, Modern Ghana, and others), on-record resident interviews, and community tips reviewed by a moderator before inclusion; each record is tagged by origin and event date. At the time of analysis, the platform had mapped 23 verified flood locations in Greater Accra, comprising 16 locations from the June 2026 flood event (which produced 593.2 mm of rainfall — the highest June total since 1995 — and affected an estimated 38,802 people), 6 chronic flood zones with recurrent multi-year records, and 1 emerging first-time location. All 23 locations are publicly named on the platform. June 2026 event locations (16): Achimota, Agbogbloshie, Darkuman Junction, Dzorwulu, Jubilee House vicinity, Kaneshie, Kasoa (Accra-Kasoa stretch), Kwame Nkrumah Interchange/Circle, Madina (Atomic), Mallam, Nungua, Shiashie, Spintex, Tema, Teshie, and Weija. Chronic flood zones (6): Adabraka, Alajo/Abossey Okai belt, Circle/Odaw River corridor, Kaneshie Market area, Klagon, and Odawna. Emerging/first-time location (1): Mallam residential streets.

This dataset differs from the DEM-derived reference in three important ways. First, it is based on reported events rather than modelled inundation thresholds, providing independent evidence of where flooding actually occurs. Second, it does not share any methodological ancestry with the MERIT DEM inputs to the fusion model. Third, it explicitly includes informal settlement flooding caused by drainage failure, which is precisely the vulnerability mode that the SLFVI is designed to detect. The platform notes that its data are guidance rather than survey-grade and that absence of a report does not imply safety; these caveats are acknowledged in the validation interpretation below (Section 5.5). With only 23 locations, this dataset is insufficient for statistical ROC analysis and is used as a qualitative spot-check of face validity — checking whether the SLFVI correctly identifies the areas that the journalistic and community record confirms as flood hotspots — rather than as a primary quantitative benchmark.

---

## 4. Methods

### 4.1 Methodological Overview

Figure 1 presents the seven-stage processing pipeline. Stage one generates a stratified road-segment sample from the OSM network. Stage two queries Street View and Mapillary metadata to establish imagery coverage. Stage three extracts annotated labels for model training. Stage four trains and applies computer vision models to extract streetscape flood indicators. Stage five computes terrain, hydrological, and exposure features for each sample point. Stage six trains the fusion model and computes the SLFVI. Stage seven performs spatial interpolation and produces output maps and rankings.

![Figure 1](outputs/figures/fig1_workflow.png)

**Figure 1.** Methodological workflow. Rounded boxes represent processing steps; rectangular boxes represent data inputs or intermediate outputs; the green terminal box represents final decision outputs. Arrows show data flow direction.

### 4.2 Sampling Design

The OSM road network for Greater Accra was downloaded and simplified using OSMnx. Roads were classified into six functional categories (motorway, primary, secondary, tertiary, residential, and unclassified) and split into 100-metre segments. Sample points were generated at the centroid of each segment and stratified across four flood exposure strata defined by distance to the nearest mapped waterway or drain (under 50 metres, 50 to 100 metres, 100 to 250 metres, and over 250 metres) and three settlement types (formal urban, informal settlement, and peri-urban) derived from the Ghana 2021 Population and Housing Census spatial layers and expert delineation. Final sample sizes across strata were proportional to stratum area.

### 4.3 Annotation Protocol

A hybrid training dataset of 2,000 images was constructed through a two-stage process. In the first stage, 400 images were manually annotated by three trained annotators using a codebook developed specifically for drainage and flood-vulnerability features observable in Ghanaian urban streetscapes. Annotators assigned binary labels for 15 image-level categories (Table 1) and an overall vulnerability class (low, moderate, or high). Inter-annotator agreement was computed on a 60-image double-coded subset independently annotated by all three raters. Fleiss' κ was calculated per label across all three raters; pairwise Cohen's κ was computed for each of the three rater pairs and averaged. The mean Fleiss' κ across the 15 binary labels was 0.51 (range −0.01 to 1.00), indicating moderate overall agreement (Landis and Koch, 1977). Agreement was highest for stagnant_water_visible (κ = 1.00, prevalence 2%), poor_road_condition (κ = 0.83), open_gutter_present (κ = 0.73), and visible_drain_present (κ = 0.65), reflecting high rater confidence for visually unambiguous conditions. Agreement was fair for pedestrian_exposure (κ = 0.37), informal_structure_near_drainage (κ = 0.35), heavy_impervious_surface (κ = 0.35), roadside_erosion (κ = 0.35), and low_lying_street_form (κ = 0.31), where codebook boundary conditions were less clearly defined. The lowest agreement was for visible_waterway_or_stream (κ = −0.01), which had near-zero prevalence in the double-coding subset (0.6%) and is therefore statistically unreliable at this sample size. Moderate-prevalence labels with κ below 0.40 were subject to codebook clarification after the double-coding round; the refined guidance is incorporated in the final annotation schema released with the benchmark dataset. In the second stage, a teacher EfficientNet-B0 model trained on all 400 human-labelled images was used to generate pseudo-labels for a further 1,600 images, which were reviewed and corrected at a 20 percent sampling rate. The 1,600 pseudo-labelled images then formed the exclusive training set for the final model; the 400 human-labelled images were split into 200 test and 200 validation images for evaluating the final model. An important caveat is that the teacher model used to generate pseudo-labels was trained on all 400 human-labelled images, including the 200 images that subsequently formed the test set. The teacher's learned representations therefore inform the pseudo-labels that shaped the final student model's training, creating an indirect evaluation dependency between the teacher's training data and the student's test set. Reported EfficientNet classification metrics (Section 5.1) consequently represent an optimistic upper bound rather than a clean independent estimate. A design that avoids this dependency would hold out a spatially separated subset of the human-labelled images before training any model; this is the recommended approach for a subsequent annotated benchmark release. The dataset is more accurately described as a hybrid human-annotated and model-pseudo-labelled dataset rather than a fully hand-labelled benchmark; label noise introduced by pseudo-labelling is acknowledged as a limitation in Section 6.4.

**Table 1.** Annotation labels and definitions used in the EfficientNet training set.

| Label | Definition |
|-------|-----------|
| visible_drain_present | A surface drain, gutter, or channel is visible in the image |
| open_gutter_present | An open, uncovered gutter is visible |
| blocked_drain_present | A drain or gutter with visible obstruction by waste or sediment |
| stagnant_water_visible | Pooled or standing water visible on road or adjacent surface |
| poor_road_condition | Visible potholes, cracking, rutting, or unpaved surface |
| heavy_impervious_surface | More than 60 percent of visible ground area is paved or concrete |
| unpaved_shoulder | Road shoulder is bare earth or gravel |
| informal_structure_near_drainage | Kiosk, container, or informal structure within visible drain corridor |
| solid_waste_accumulation | Visible refuse heap or scattered waste near drainage |
| visible_waterway_or_stream | Natural stream or channel visible in image |
| low_lying_street_form | Road profile appears lower than surrounding terrain |
| roadside_erosion | Visible soil erosion or undermining at road edge |
| pedestrian_exposure | Absence of sidewalk or pedestrian barrier adjacent to road |
| culvert_or_bridge_visible | A culvert crossing or bridge structure is visible |
| no_visible_drainage | No drainage infrastructure of any kind is visible in the image |

### 4.4 Computer Vision Model

#### 4.4.1 EfficientNet-B0 Architecture

The primary computer vision model was an EfficientNet-B0 backbone with two task-specific output heads. EfficientNet (Tan and Le, 2019) uses a compound scaling rule to balance network depth d, width w, and input resolution r according to

$$d = \alpha^\phi, \quad w = \beta^\phi, \quad r = \gamma^\phi$$

subject to the constraint $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ and the condition that $\alpha \geq 1$, $\beta \geq 1$, $\gamma \geq 1$, where $\phi$ is a user-defined compound coefficient. The B0 variant is the baseline network at $\phi = 0$; the scaling coefficients $\alpha = 1.2$, $\beta = 1.1$, $\gamma = 1.15$ were determined by a grid search on B0 and are then applied for $\phi = 1, 2, \ldots$ to produce progressively larger B1 through B7 variants. EfficientNet-B0 has approximately 5.3 million parameters, making it suitable for training on a dataset of the present size without extensive GPU resources.

The EfficientNet backbone was modified by replacing its default classification head with an identity layer, producing a 1,280-dimensional feature vector. Two linear output heads were attached: a binary vulnerability head with two output neurons (non-vulnerable and vulnerable) and a multi-label head with 15 output neurons corresponding to the annotation labels in Table 1. The three-level annotator label (low, moderate, high) was collapsed to binary for the vulnerability head by treating moderate and high as the positive class (vulnerable = 1) and low as the negative class (non-vulnerable = 0), reflecting the operational decision that any detectable elevation above baseline vulnerability warrants flagging for infrastructure review. Formally, for an input image $\mathbf{x}$, the backbone produces feature representation $\mathbf{f} = g(\mathbf{x}) \in \mathbb{R}^{1280}$, and the outputs are

$$\hat{y}_{vuln} = \text{softmax}(\mathbf{W}_{vuln} \mathbf{f} + \mathbf{b}_{vuln})$$

$$\hat{\mathbf{y}}_{labels} = \sigma(\mathbf{W}_{labels} \mathbf{f} + \mathbf{b}_{labels})$$

where $\sigma(\cdot)$ denotes the element-wise sigmoid function. The binary vulnerability prediction uses the positive-class softmax probability, while each of the 15 label predictions is an independent sigmoid probability.

Training used a combined loss function

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{BCE}$$

where $\mathcal{L}_{CE}$ is categorical cross-entropy on the binary vulnerability head, $\mathcal{L}_{BCE}$ is the mean binary cross-entropy across the 15 label heads, and $\lambda = 1.0$ is a weighting coefficient. Class imbalance in the binary head was addressed using a positive class weight of 2.5, computed from the inverse class frequency in the training data. The model was trained for 20 epochs using the AdamW optimiser with a learning rate of $10^{-4}$ and a cosine annealing schedule.

#### 4.4.2 CLIP Baseline

OpenAI CLIP (Radford et al., 2021) was used as a zero-shot baseline to generate initial vulnerability scores before annotated training data became available. Fifteen text prompts corresponding to the annotation labels in Table 1 were constructed in natural language form (for example, "a blocked drain with solid waste" and "a road with no visible drainage infrastructure"). For each image, CLIP computed the cosine similarity between the image embedding and each prompt embedding in the shared vision-language space, producing a soft-label score for each category.

#### 4.4.3 Training and Evaluation

The final EfficientNet model was trained on the 1,600 pseudo-labelled images and evaluated on the 200-image test set and 200-image validation set drawn from the 400 human-annotated images. No pseudo-labelled image enters the evaluation split directly. However, as noted in Section 4.3, the teacher model that generated the pseudo-labels was trained on all 400 human-labelled images including the test and validation sets; reported classification metrics are therefore subject to indirect evaluation dependency and should be interpreted as upper-bound estimates. All input images were resized to 224 by 224 pixels and normalised using ImageNet mean and standard deviation values. Data augmentation during training included random horizontal flipping, random rotation up to 15 degrees, colour jitter with brightness and contrast variation, and random Gaussian blur. Model performance was evaluated using ROC-AUC, precision-recall AUC, F1 score, and per-label average precision.

After training, the fine-tuned EfficientNet-B0 model was applied in inference mode to all 15,344 Street View images and all 590 Mapillary images, producing per-image vulnerability probabilities and 15-label indicator scores.

### 4.5 Geospatial Feature Engineering

#### 4.5.1 Terrain Features

The topographic wetness index (TWI) was computed as

$$\text{TWI} = \ln\!\left(\frac{a}{\tan\beta}\right)$$

where $a$ is the upslope contributing area per unit contour length in square metres per metre and $\beta$ is the local slope angle in radians (Beven and Kirkby, 1979). Higher TWI values indicate topographic positions prone to moisture accumulation. Slope was computed in degrees using the third-order finite difference method. Depression depth was computed as the difference between the filled DEM and the original DEM, identifying enclosed topographic depressions likely to accumulate runoff.

#### 4.5.2 Distance and Proximity Features

Euclidean distances to the nearest OSM-mapped waterway, open drain, wetland, and coastline were computed in EPSG 32630 for each sample point. A binary indicator was set to one for points within 100 metres of a mapped waterway or drain, following the 100-metre buffer threshold used in the sampling stratification. Flow accumulation values from the D8 routing were extracted at each point to characterise upstream catchment size.

#### 4.5.3 Exposure Features

Building density was computed as the count of building footprints within a 100-metre buffer of each sample point, using the Google Open Buildings dataset. Population density was interpolated from the WorldPop 100-metre grid. Nighttime lights intensity was extracted from VIIRS monthly composites to serve as a proxy for commercial activity and infrastructure investment.

### 4.6 Streetscape Feature Aggregation

For each of the 3,836 sample points, predictions from the four directional images (headings 0, 90, 180, and 270 degrees) were aggregated to a single point-level feature vector by taking the mean and maximum across headings for each of the 15 label probabilities and the overall vulnerability probability. This produced 32 streetscape-derived features per point (2 statistics by 15 labels plus the mean and maximum vulnerability probability). Of these 32, six features were excluded prior to fusion model training due to near-zero variance in the aggregated road-segment dataset (fewer than 0.5 percent of points scoring above 0.1 on the relevant label), leaving 26 streetscape features for model input. The six excluded features are identified in the supplementary data release. A composite streetscape sensitivity score was computed as a weighted mean of the most drainage-relevant label probabilities:

$$S = \frac{1}{Z} \sum_{k=1}^{K} w_k \cdot p_k$$

where $p_k$ is the mean heading-averaged probability for label $k$, $w_k$ is the pre-specified weight for that label (with drain-related labels weighted 1.5 and road-condition labels weighted 1.0), and $Z$ is a normalisation constant ensuring $S \in [0, 1]$.

### 4.7 Street-Level Flood Vulnerability Index

Table 1a defines the four SLFVI components, the observable inputs used to operationalise each, and the rationale for the assigned weight.

**Table 1a.** SLFVI component definitions and operationalisation.

| Component | Conceptual definition | Operational inputs | Weight |
|-----------|----------------------|-------------------|--------|
| Hazard (H) | Terrain conditions that concentrate surface water | TWI, flow accumulation, depression depth, inverse distance to waterway | 0.30 |
| Exposure (E) | Population and assets at risk in the flood path | Building count (100 m buffer), population density | 0.20 |
| Sensitivity (S) | Visible street-level conditions that amplify flood impact given hazard | EfficientNet composite (drain obstruction, solid waste, imperviousness, road condition) | 0.35 |
| Inverse adaptive capacity (A) | Lack of physical and institutional capacity to reduce flood impact; expressed as a vulnerability contribution (higher = lower capacity = greater vulnerability) | Road width class (emergency access proxy, inverted), normalised distance from district boundary centre (service proximity proxy, inverted) | 0.15 |

Sensitivity receives the highest weight because it represents the novel observational contribution of the framework and because drainage failure rather than topographic position is the proximate flood mechanism in Accra's flat coastal districts. The adaptive capacity proxy is explicitly provisional; road class and boundary distance are weak substitutes for direct measures of drainage maintenance coverage or emergency response proximity.

The SLFVI integrates four component scores using a weighted linear formula inspired by the IPCC vulnerability framework (IPCC, 2022) and the Cutter et al. (2003) social vulnerability index tradition:

$$\text{SLFVI} = 0.30 \cdot H + 0.20 \cdot E + 0.35 \cdot S + 0.15 \cdot A$$

where $H$ is the terrain and rainfall hazard score, $E$ is the exposure score derived from building count and population density, $S$ is the streetscape sensitivity score from the EfficientNet model, and $A$ is the inverse adaptive capacity score — operationalised as a vulnerability contribution rather than a capacity measure, so that higher values indicate lower adaptive capacity and greater vulnerability (narrow road class and greater distance from the district boundary centre). All four components are independently normalised to the unit interval using

$$\tilde{x} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

The weights (0.30, 0.20, 0.35, 0.15) reflect informed prior judgements grounded in the conceptual framework: hazard receives the largest single weight among the three conventional components because elevation and drainage gradients are well-established flood predictors; sensitivity receives the highest overall weight because drainage failure rather than topographic inundation is the proximate flood mechanism in Accra's flat coastal districts; adaptive capacity receives the lowest weight given the limited quality of available proxies. These weights were not derived from a formal expert-elicitation protocol with a scored panel; future work should apply analytic hierarchy process or participatory weighting methods with local practitioners to produce empirically grounded component weights. A sensitivity analysis across four alternative weight specifications (equal weights 0.25/0.25/0.25/0.25; hazard-dominant 0.40/0.20/0.25/0.15; streetscape-dominant 0.20/0.15/0.50/0.15; no adaptive capacity 0.375/0.25/0.375/0.00) shows Spearman rank correlations with the original ordering of 0.934, 0.981, 0.984, and 0.829 respectively, and class-level agreement of 87.6, 83.8, 74.1, and 83.7 percent. The rank ordering of road segments is therefore robust to substantial weight perturbations; the class assignments — particularly the distinction between moderate and high — are more sensitive to weight changes in the streetscape-dominant specification. The data-driven slfvi_final variant (Section 4.8) jointly optimises all feature weights against the DEM-derived flood exposure labels and represents a fifth alternative specification for sensitivity comparison.

The hazard score $H$ was constructed as the mean of normalised values for TWI, flow accumulation, depression depth, and inverse distance to the nearest waterway. The exposure score $E$ combined normalised building count and population density; in the current 3,836-point pilot dataset, normalised building count and population density exhibited very limited within-dataset variation, causing the $E$ component to take a near-constant value across all road points (0.50 ± 0.00). The exposure term therefore contributes a uniform additive constant to all SLFVI values in this pilot and does not differentiate between road segments; collecting a larger, more spatially diverse sample or using a richer exposure index would be required to make this component discriminating. The sensitivity score $S$ was the streetscape composite described in Section 4.6. The inverse adaptive capacity score $A$ was approximated by road width class (as a proxy for emergency access route quality) and normalised distance from the district boundary centre (as a rough proxy for service proximity), expressed such that higher values indicate lower capacity and greater vulnerability. This proxy is explicitly provisional: road class and boundary distance are weak substitutes for direct measures of drainage maintenance service coverage, emergency response proximity, or social coping capacity, all of which would require data not currently available at the required spatial resolution. The component weight of 0.15 reflects the limited confidence placed in this proxy.

A data-driven variant (slfvi_final) was also computed using the predicted flood probability from the LightGBM fusion model (Section 4.8), which jointly optimised all feature weights against the DEM-derived flood exposure labels. This supervised variant is reported alongside the analytical SLFVI for comparison and to assess the information content of the feature set, but the primary index for planning interpretation is the analytical SLFVI because its component weights are transparent and independently auditable. The LightGBM probability is better understood as a model output calibrated to the DEM-derived exposure target than as a general flood vulnerability score.

SLFVI values were classified into five vulnerability categories:

| Category | Range |
|----------|-------|
| Very low | 0.00 to 0.20 |
| Low | 0.20 to 0.40 |
| Moderate | 0.40 to 0.60 |
| High | 0.60 to 0.80 |
| Very high | 0.80 to 1.00 |

### 4.8 Fusion Model

A gradient-boosted tree model was trained to predict binary flood exposure status using the combined feature set of streetscape indicators, terrain variables, and exposure features. The predicted probability output of this model is referred to in this paper as both slfvi_final (in Tables 4, 5, and 6, where it is used to classify road segments) and ml_flood_prob (in Figure 3 and Table 2, where it is compared against other model variants); these are identical values reported under two naming conventions used in the analysis code and results tables respectively. Three algorithms were compared: LightGBM (Ke et al., 2017), XGBoost (Chen and Guestrin, 2016), and Random Forest (Breiman, 2001). Hyperparameters were tuned using five-fold cross-validation with stratification on the flood exposure label. To account for spatial autocorrelation and avoid overly optimistic accuracy estimates, a spatial cross-validation scheme was also applied (Roberts et al., 2017). Spatial folds were constructed by assigning each road point to a spatial block defined by the combination of its latitude quintile and longitude quintile, producing up to 25 unique spatial blocks. These blocks were then randomly assigned to five folds of approximately equal size. For each fold, all points in the held-out blocks formed the test set and all points in the remaining blocks formed the training set, with no additional distance buffer enforced within the block assignment. All four directional images from a single road point were kept in the same fold, preventing image-level leakage across fold boundaries. The minimum effective spatial separation between training and test points under this scheme depends on block size but typically exceeds 500 metres in the study area, which spans approximately 51 kilometres east-west and 39 kilometres north-south. A supplementary figure showing the spatial distribution of the five cross-validation fold assignments across the study area is provided in the supplementary materials.

Model interpretability was assessed using SHAP (SHapley Additive exPlanations) values (Lundberg and Lee, 2017), which decompose each prediction into additive feature contributions based on cooperative game theory. SHAP values allow identification of which features drive individual predictions, and the mean absolute SHAP value across all predictions provides a global feature importance ranking.

### 4.9 Spatial Interpolation

The point-level SLFVI was interpolated to a continuous 250-metre grid covering the study area using two complementary methods.

Inverse distance weighting (IDW) was applied using the formula

$$\hat{z}(\mathbf{s}_0) = \frac{\sum_{i=1}^{k} w_i(\mathbf{s}_0) \cdot z(\mathbf{s}_i)}{\sum_{i=1}^{k} w_i(\mathbf{s}_0)}$$

where $w_i(\mathbf{s}_0) = d(\mathbf{s}_0, \mathbf{s}_i)^{-p}$, $d(\mathbf{s}_0, \mathbf{s}_i)$ is the Euclidean distance from the prediction location $\mathbf{s}_0$ to the $i$-th known point, $p = 2$ is the distance decay power, and $k = 12$ is the number of nearest neighbours. The distance floor of $10^{-6}$ metres prevents division by zero at coincident locations.

Ordinary kriging was applied as a geostatistically optimal alternative. The kriging system solves for weights $\lambda_i$ that minimise the prediction variance subject to the unbiasedness constraint:

$$\sum_{j=1}^{n} \lambda_j \cdot \gamma(\mathbf{s}_i - \mathbf{s}_j) + \mu = \gamma(\mathbf{s}_i - \mathbf{s}_0) \quad \forall i$$

$$\sum_{i=1}^{n} \lambda_i = 1$$

where $\gamma(\mathbf{h})$ is the experimental semivariogram modelled as a spherical function with nugget, sill, and range parameters estimated by weighted least squares from the 3,836 observed points, and $\mu$ is the Lagrange multiplier enforcing the unbiasedness condition (Cressie, 1993). The kriging surface additionally provides a kriging variance surface quantifying prediction uncertainty at each grid cell.

Both interpolation methods were applied after clipping the grid to the study area boundary, yielding 16,271 grid cells covering the 21-district study area.

### 4.10 Ethics, Privacy, and Data Use

Street View imagery was accessed through the Google Maps Platform Street View Static API and Street View Metadata API under the terms of service for research and educational use. No Street View imagery is reproduced in this publication; all figures displaying street-level scenes use Mapillary imagery only (Section 3.2). Mapillary images are reproduced under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) licence, with panorama identifiers displayed in each figure tile as required by the licence terms. Derived prediction outputs (EfficientNet probabilities and SLFVI scores) are stored and reported at road-segment and grid-cell resolution; raw Street View images are not redistributed. Privacy-sensitive content is managed through the automatic face and vehicle licence-plate blurring applied by both the Google Maps Platform and Mapillary before imagery is served; no additional manual redaction was required. The study collects no personal data and does not identify individual persons, properties, or residences. Community-level outputs (district vulnerability summaries, drainage maintenance priority lists) are framed as infrastructural risk assessments rather than characterisations of residents, consistent with the ethical guidance of Wisner et al. (2004) and the study's governance principle that vulnerability is primarily a function of infrastructure condition and institutional capacity rather than resident behaviour.

---

## 5. Results

### 5.1 Street View Coverage and the Mapillary Gap Analysis

Of the 5,000 stratified road points queried, 3,912 (78.2 percent) returned valid Street View metadata (status OK). After removing duplicate panoramas, 3,836 unique points with confirmed coverage were retained, corresponding to a unique-point coverage rate of 76.7 percent. These two figures measure different things: the 78.2 percent rate captures metadata availability, while the 76.7 percent rate captures distinct road locations retained for analysis. Coverage was highest in primary and secondary road classes (greater than 90 percent) and lowest in unclassified tracks and informal-settlement access ways (below 40 percent). The 1,088 points without Street View coverage were spatially concentrated in peripheral informal settlements, particularly in Weija-Gbawe, outer Ablekuma, and peri-urban corridors south of Ashaiman.

Mapillary imagery was found within 100 metres of 159 of the 1,088 gap points (14.6 percent). After quality screening, 590 of the 620 returned images were retained for inference. EfficientNet inference on the 590 Mapillary images produced a mean vulnerability probability of 0.580, compared with 0.384 for the 15,344 Street View images (difference 0.196, bootstrap 95% CI [0.171, 0.221], Cohen's d = 0.89). The proportion of Mapillary images classified as high risk (probability above 0.5) was 59.7 percent, compared with approximately 38 percent for the Street View dataset. The top-scoring features in Mapillary images were no_visible_drainage (mean probability 0.576), pedestrian_exposure (0.514), poor_road_condition (0.475), and informal_structure_near_drainage (0.473).

This finding indicates a systematic coverage bias: the streets that Street View does not cover are on average more flood-vulnerable than those it does cover. A chi-squared test of independence between coverage status (covered or not covered) and flood stratum (high-risk or low-risk, defined by distance to waterway) confirmed that coverage gaps are concentrated in high-risk flood strata (chi-squared statistic 47.3, p less than 0.001). 
### 5.2 Computer Vision Performance

The fine-tuned EfficientNet-B0 model, evaluated on the 200-image human-labelled held-out test set, achieved a binary vulnerability ROC-AUC of 0.713 and an F1 score of 0.560, compared with a CLIP zero-shot baseline ROC-AUC of 0.621. As noted in Section 4.3, the teacher model used to generate pseudo-labels for the training set was trained on all 400 human-labelled images including the test set; these metrics are therefore optimistic upper bounds rather than clean independent estimates, and should be read accordingly. The improvement of 0.092 ROC-AUC units over the zero-shot CLIP baseline nevertheless confirms that locally supervised fine-tuning on Ghanaian streetscape images adds discriminative power beyond general-purpose vision-language pre-training, since the CLIP baseline has no access to any labelled training data and is therefore unaffected by this dependency.

Per-label average precision varied substantially across the 15 categories. The highest detection performance was achieved for heavy_impervious_surface (AP 0.81), poor_road_condition (AP 0.77), and pedestrian_exposure (AP 0.74), reflecting the relatively clear visual signal for these features. The lowest performance was for blocked_drain_present (AP 0.43) and stagnant_water_visible (AP 0.38), where the relevant feature is often small, partially occluded, or only distinguishable from clean drainage through close visual inspection, a challenge documented in similar annotation tasks (Biljecki and Ito, 2021).

### 5.3 Fusion Model Performance

Table 2 presents the performance of the three fusion model algorithms on both standard five-fold cross-validation and spatial cross-validation against the flood exposure target.

**Table 2.** Fusion model algorithm comparison — full feature set (34 features; flood_exposed_count_100m excluded to prevent direct target leakage), standard 5-fold stratified CV and spatial block CV against the DEM-derived building flood exposure reference.

| Model | ROC-AUC (standard CV) | PR-AUC (standard CV) | ROC-AUC (spatial CV) | PR-AUC (spatial CV) |
|-------|----------------------|---------------------|---------------------|---------------------|
| Random Forest | 0.993 ± 0.002 | 0.876 ± 0.043 | 0.982 ± 0.015 | 0.715 ± 0.240 |
| XGBoost | 0.996 ± 0.002 | 0.949 ± 0.025 | 0.984 ± 0.023 | 0.816 ± 0.187 |
| LightGBM | 0.996 ± 0.002 | 0.946 ± 0.021 | 0.987 ± 0.016 | 0.778 ± 0.216 |

XGBoost achieved the highest PR-AUC under spatial CV (0.816); LightGBM achieved the highest ROC-AUC under spatial CV (0.987). The two models perform comparably and both outperform Random Forest on spatial PR-AUC. All three models show a 0.01 to 0.16 AUC reduction from standard to spatial cross-validation, confirming the presence of spatial autocorrelation and the importance of using spatially separated folds for honest performance estimation.

SHAP analysis of the best LightGBM model identified terrain and exposure features as the dominant predictors, followed by EfficientNet-derived streetscape features (Figure 2). The top ten SHAP contributors in order of mean absolute SHAP value were: elevation (2.56), building count within 100 metres (1.07), slope (0.51), distance to nearest waterway (0.37), local depression depth (0.30), enet_no_visible_drainage (0.22), bare-ground surface fraction (0.20), distance to nearest drain (0.20), enet_heavy_impervious_surface (0.15), and building count within 250 metres (0.12). The two EfficientNet features in the top ten contributed signal beyond the terrain variables. The full ablation in Table 2b shows, however, that streetscape features do not improve discrimination against the DEM-derived target when terrain and exposure variables are already included — a finding that is itself informative: the streetscape signal is orthogonal to topographic flood exposure rather than correlated with it, which is the theoretical basis of this paper's contribution.

**Table 2b.** LightGBM ablation study: performance by feature subset under standard stratified 5-fold CV and spatial block CV. The flood_exposed_count_100m column was excluded from all runs to avoid direct target leakage. Values are mean across folds.

| Feature set | N feats | ROC-AUC (std CV) | PR-AUC (std CV) | ROC-AUC (sp CV) | PR-AUC (sp CV) |
|---|---|---|---|---|---|
| Terrain only | 5 | 0.989 ± 0.007 | 0.838 ± 0.053 | 0.976 ± 0.023 | 0.639 ± 0.243 |
| Terrain + exposure | 8 | 0.996 ± 0.003 | 0.951 ± 0.026 | 0.987 ± 0.015 | 0.801 ± 0.186 |
| Streetscape only | 26 | 0.744 ± 0.029 | 0.175 ± 0.052 | 0.674 ± 0.068 | 0.113 ± 0.061 |
| Terrain + exposure + streetscape | 34 | 0.996 ± 0.002 | 0.946 ± 0.021 | 0.987 ± 0.016 | 0.778 ± 0.216 |

The ablation results reveal two important patterns. First, terrain features alone achieve ROC-AUC 0.989 and PR-AUC 0.838 against the DEM-derived flood exposure target, confirming that high performance is substantially attributable to the shared methodological ancestry between the model inputs and the validation reference (see Section 3.4). Second, adding streetscape features to the terrain-plus-exposure baseline does not improve PR-AUC (0.946 versus 0.951 under standard CV; 0.778 versus 0.801 under spatial CV), indicating that the streetscape signal is largely orthogonal to the DEM-based target rather than additive to it. This is conceptually consistent with the central argument of this paper: streetscape AI detects drainage failure conditions that are different from topographic flood exposure. The streetscape-only model achieves ROC-AUC 0.744 and PR-AUC 0.175 in isolation, representing the non-terrain streetscape signal from street-level infrastructure conditions relative to the DEM-derived reference.

![Figure 2](outputs/figures/shap_importance.png)

**Figure 2.** Mean absolute SHAP values for the top ten predictors in the LightGBM fusion model. EfficientNet-derived features (enet_no_visible_drainage, enet_heavy_impervious_surface) appear among the top ten alongside terrain variables, indicating that the model draws on street-level infrastructure information alongside terrain and hydrological inputs. The ablation results in Table 2b show that these features do not improve discrimination against the DEM-derived target once terrain and exposure variables are included, consistent with the interpretation that streetscape AI detects drainage-failure vulnerability orthogonal to topographic inundation. Feature names prefixed with "enet_" are derived from street-level image inference; all others are terrain, hydrological, or exposure variables.

### 5.4 Street-Level Flood Vulnerability Index

The analytical SLFVI was computed for all 3,836 sample points. Table 3 summarises the distribution of SLFVI scores by vulnerability class.

**Table 3.** Distribution of SLFVI scores across the study area.

| Class | Points | Percentage | Mean SLFVI |
|-------|--------|-----------|-----------|
| Very low (0.00 to 0.20) | 804 | 21.0% | 0.142 |
| Low (0.20 to 0.40) | 2,325 | 60.6% | 0.313 |
| Moderate (0.40 to 0.60) | 545 | 14.2% | 0.496 |
| High (0.60 to 0.80) | 133 | 3.5% | 0.663 |
| Very high (0.80 to 1.00) | 29 | 0.8% | 0.841 |

The distribution is right-skewed, with the majority of road points falling in the low vulnerability class. The 162 points classified as high or very high (4.2 percent of sampled road locations) are spatially concentrated in the low-lying Odaw basin, coastal fringe, and inner informal settlements of Ablekuma and Ledzokuku.

### 5.5 Validation against Flood Exposure Data

Table 4 presents discrimination metrics for three SLFVI variants against the DEM-derived building flood exposure reference.

**Table 4.** Discrimination performance of SLFVI variants against a DEM-derived building flood exposure reference (3,836 road points, 211 near DEM-defined flood-exposed buildings within 100-metre buffer). Values are computed on the full dataset. Note: the supervised slfvi_final and ml_flood_prob variants share methodological ancestry with terrain inputs used to construct the reference; their high AUC values reflect internal consistency with the DEM-derived target rather than validation against observed flood events.

| Score | ROC-AUC | PR-AUC |
|-------|---------|--------|
| slfvi (analytical formula) | 0.701 | 0.121 |
| slfvi_final (LightGBM predicted probability) | 0.990 | 0.914 |
| ml_flood_prob (raw model output) | 0.999 | 0.993 |

The monotonic relationship between SLFVI class and flood exposure rate is presented in Table 5.

**Table 5.** Flood-exposed building rate by SLFVI vulnerability class.

| SLFVI Class | N Points | Near Flood-Exposed Buildings (%) |
|-------------|---------|--------------------------------|
| Very low | 804 | 0.0% |
| Low | 2,325 | 0.1% |
| Moderate | 545 | 10.6% |
| High | 133 | 91.0% |
| Very high | 29 | 100.0% |

ROC and precision-recall curves for all three SLFVI variants are shown in Figure 3; Figure 4 presents the by-class bar chart.

The monotonic progression from 0.0 percent in very-low-scoring areas to 100.0 percent in very-high-scoring areas (n = 29; Clopper–Pearson 95% CI: 88.1%–100.0%) confirms that the SLFVI is internally consistent with the DEM-derived exposure layer: higher-scoring road locations are systematically closer to DEM-defined flood-exposed buildings. This result should be interpreted as evidence of alignment with terrain-based exposure estimates rather than as definitive proof of real-world flood prediction accuracy, because the validation target and several model inputs share the same MERIT DEM ancestry (see Section 3.4). The near-perfect discrimination of the supervised slfvi_final variant (ROC-AUC 0.990) reflects this shared ancestry more than it reflects independent predictive validity. The analytical SLFVI achieves moderate discriminative ability (ROC-AUC 0.701) against the same target, which is more informative about the index's likely performance against genuinely independent flood data.

**Supplementary spot-check against FloodSpots event data.** As an independent, non-DEM validation check, the 23 verified flood hotspot locations from the FloodSpots platform (Section 3.5) were cross-referenced against the SLFVI surface. Of the 23 named locations, 19 fall within the study area boundary (four locations — Kasoa, Tema, Spintex, and Jubilee House vicinity — are on the study area margins or lie outside the core 21-district zone). Of those 19, 15 (78.9 percent) extract at moderate or above (SLFVI ≥ 0.40) under the neighbourhood-level extraction method described in Table S1. Two caveats affect this count. First, Mallam (row 2) and Mallam residential streets (row 22) are recorded at identical geocoded coordinates and represent the same street network extent; treating them as a single location yields 18 unique in-study locations with 14 matching at moderate or above (77.8 percent). Second, Kaneshie (row 18) and Kaneshie Market area (row 20) are approximately 200 metres apart and their 1.5 km extraction radii overlap substantially, meaning they draw on largely the same road-segment pool; their SLFVI scores are accordingly identical (0.671). The deduplicated rate (14 of 18 unique locations, 77.8 percent) is used as the primary figure in the remainder of this paragraph. For comparison, only 7.6 percent of all 16,271 grid cells in the study area (1,235 of 16,271: 934 moderate, 298 high, and 3 very high) are classified at moderate or above under the IDW surface (Table 7); the 77.8 percent match rate is therefore substantially above the level expected by chance and is consistent with the index capturing real spatial variation in flood risk exposure. Eleven locations extract as High (SLFVI ≥ 0.60) in the deduplicated set: Agbogbloshie, Mallam, Weija, Teshie, Madina (Atomic), Achimota, Kaneshie, Kwame Nkrumah Interchange/Circle, Alajo/Abossey Okai, Kaneshie Market area, and Odawna. Three locations — Darkuman Junction, Adabraka, and Circle/Odaw River corridor — extract as Moderate (0.40–0.60). The four locations falling in low-vulnerability SLFVI cells are: Dzorwulu and Shiashie (elevated planned residential areas where June 2026 flooding may have reflected localised obstruction events not captured at 250-metre resolution), and Nungua and Klagon (coastal communities at the eastern study area periphery where road-segment coverage is sparse, limiting the reliability of the SLFVI extraction). Taken together, this spot-check provides face validity from an independent, event-based source: the areas the SLFVI identifies as high risk substantially overlap with the areas that journalistic reporting and community testimony confirm flooded in the most recent major event. Because this validation is qualitative and based on named neighbourhood matches rather than coordinate-level comparison, it supplements rather than replaces quantitative validation; NADMO event coordinate data and field-survey verification remain a priority for future work.

Table S1 (supplementary) presents the full cross-reference for all 23 named FloodSpots locations, including SLFVI scores derived from the 95th-percentile road-segment value within a 1.5-kilometre radius of each neighbourhood centroid (a method that captures the most flood-prone corridor within each community rather than the centroid value, which can fall on higher ground within a large neighbourhood). All 23 locations are publicly named on the platform.

**Table S1.** FloodSpots verified flood locations cross-referenced against the SLFVI surface. SLFVI score method: 'p95 (1.5 km)' = 95th-percentile analytical SLFVI across all road-segment points within 1.5 km of geocoded neighbourhood centroid; 'centroid' = IDW raster value at geocoded centroid, used for planned residential areas (Dzorwulu, Shiashie) and peripheral locations with fewer than three road points in radius. Coordinates are approximate OpenStreetMap Nominatim centroids, not precision-survey locations. Event types follow platform classifications: June 2026 = location reported during June 2026 event; Chronic = recurrent multi-year flood zone; Emerging = first-time recorded location.

| # | Location | Event type | Date | Source | Coordinates (approx.) | SLFVI score | Method | SLFVI class |
|---|---|---|---|---|---|---|---|---|
| 1 | Agbogbloshie | June 2026 | June 2026 | News + Community | 5.5545°N, 0.2275°W | 0.708 | p95 (1.5 km) | High |
| 2 | Mallam | June 2026 | June 2026 | News | 5.5708°N, 0.2854°W | 0.634 | p95 (1.5 km) | High |
| 3 | Darkuman Junction | June 2026 | June 2026 | News | 5.5903°N, 0.2520°W | 0.576 | p95 (1.5 km) | Moderate |
| 4 | Adabraka | Chronic | Recurrent | News | 5.5625°N, 0.2129°W | 0.595 | p95 (1.5 km) | Moderate |
| 5 | Alajo/Abossey Okai | Chronic | Recurrent | News | 5.5591°N, 0.2336°W | 0.670 | p95 (1.5 km) | High |
| 6 | Weija | June 2026 | June 2026 | News + Community | 5.5620°N, 0.2850°W | 0.701 | p95 (1.5 km) | High |
| 7 | Teshie | June 2026 | June 2026 | News + Community | 5.5822°N, 0.1135°W | 0.749 | p95 (1.5 km) | High |
| 8 | Madina (Atomic) | June 2026 | June 2026 | News | 5.6762°N, 0.1556°W | 0.671 | p95 (1.5 km) | High |
| 9 | Dzorwulu | June 2026 | June 2026 | Community | 5.6123°N, 0.2038°W | 0.352 | centroid | Low |
| 10 | Shiashie | June 2026 | June 2026 | Community | 5.6289°N, 0.1697°W | 0.340 | centroid | Low |
| 11 | Nungua | June 2026 | June 2026 | News + Community | 5.6011°N, 0.0704°W | 0.259 | centroid | Low |
| 12 | Achimota | June 2026 | June 2026 | News | 5.6242°N, 0.2277°W | 0.676 | p95 (1.5 km) | High |
| 13 | Klagon | Chronic | Recurrent | Community | 5.6637°N, 0.0533°W | 0.274 | centroid | Low |
| 14 | Kasoa † | June 2026 | June 2026 | News | 5.5326°N, 0.4375°W | — | — | Outside core zone |
| 15 | Tema † | June 2026 | June 2026 | News + Community | 5.6596°N, 0.0097°W | — | — | Outside core zone |
| 16 | Spintex † | June 2026 | June 2026 | News | 5.6288°N, 0.0902°W | — | — | Outside core zone |
| 17 | Jubilee House vicinity † | June 2026 | June 2026 | News | 5.5797°N, 0.1884°W | — | — | Outside core zone |
| 18 | Kaneshie | June 2026 | June 2026 | News | 5.5666°N, 0.2345°W | 0.671 | p95 (1.5 km) | High |
| 19 | Kwame Nkrumah Interchange/Circle | June 2026 | June 2026 | News | 5.5696°N, 0.2154°W | 0.625 | p95 (1.5 km) | High |
| 20 | Kaneshie Market area | Chronic | Recurrent | Community | 5.5648°N, 0.2367°W | 0.671 | p95 (1.5 km) | High |
| 21 | Odawna | Chronic | Recurrent | Community | 5.5587°N, 0.2161°W | 0.690 | p95 (1.5 km) | High |
| 22 | Mallam residential streets | Emerging | June 2026 | Community | 5.5708°N, 0.2854°W | 0.634 | p95 (1.5 km) | High |
| 23 | Circle/Odaw River corridor | Chronic | Recurrent | Community | 5.5620°N, 0.2070°W | 0.581 | p95 (1.5 km) | Moderate |

*† Location on study area margin or outside core 21-district zone; excluded from the in-study-area count. Source types: News = published news reporting (AP, JoyNews, B&FT, Modern Ghana); Community = resident interview or community tip verified by platform moderator; News + Community = both. Nungua and Klagon extract as Low, reflecting sparse road-segment coverage at the eastern study area periphery (coastal communities with fewer than three road points within 1.5 km of geocoded centroid); centroid IDW values are used as fallback and carry higher uncertainty. Note: rows 2 (Mallam) and 22 (Mallam residential streets) share identical geocoded coordinates and are treated as one location in the match-rate calculation (18 unique in-study locations); rows 18 (Kaneshie) and 20 (Kaneshie Market area) are approximately 200 m apart with overlapping extraction radii and identical SLFVI scores. Coordinate-level extraction against actual FloodSpots event coordinates is recommended to resolve these cases in future work.*

![Figure 3](outputs/figures/validation_roc_pr.png)

**Figure 3.** ROC curves (left panel) and precision-recall curves (right panel) for the three SLFVI variants against the DEM-derived building flood exposure reference. The supervised slfvi_final and ml_flood_prob curves (upper bound) should be read with the methodological caveat in Table 4; the analytical SLFVI curve (ROC-AUC 0.701) is the more conservative and policy-relevant estimate. Jaggedness in the SLFVI-formula precision–recall curve reflects the discrete score distribution of the linear index (a weighted sum of a small number of continuous components) rather than numerical instability.

![Figure 4](outputs/figures/validation_by_class.png)

**Figure 4.** Percentage of road points located within 100 metres of a DEM-defined flood-exposed building, by analytical SLFVI vulnerability class (n = 3,836 points). The monotonic pattern confirms internal consistency with the DEM-derived exposure reference. Bars represent the five SLFVI classes; percentages are as reported in Table 5.

### 5.6 Focus District Analysis

Table 6 summarises the vulnerability results for the four focus districts, including both observed (Street View) and predicted (interpolated) points along the full road network within each district. Observed points are the 3,836 sampled Street View locations with directly computed analytical SLFVI values. Predicted points are unsampled 100-metre road-segment centroids throughout each district that received analytical SLFVI values by extracting from the IDW-interpolated 250-metre grid surface described in Section 5.7. The large ratio of predicted to observed points reflects the density of the full district road network relative to the 5,000-point stratified sample; SLFVI values for predicted points carry the interpolation uncertainty associated with the 250-metre grid and should be interpreted as spatially smoothed estimates rather than directly measured values.

**Table 6.** Focus district vulnerability summary (combined observed and predicted points along full road network).

| District | Total Points | Observed | Predicted | Mean SLFVI | High+ (%) |
|----------|------------|---------|----------|-----------|----------|
| Ledzokuku | 8,330 | 278 | 8,052 | 0.615 | 71.2% |
| Ablekuma Central | 5,323 | 194 | 5,129 | 0.592 | 38.2% |
| Ga East | 10,341 | 440 | 9,901 | 0.566 | 15.4% |
| La Dade-Kotopon | 11,141 | 372 | 10,769 | 0.288 | 0.3% |

Ledzokuku emerges as the highest-vulnerability district with a mean SLFVI of 0.615 and 71.2 percent of road network points classified as high or very high. This is consistent with documented flood history in the district, which encompasses densely built coastal communities including Teshie. (Nungua is administratively within Krowor Municipal Assembly and is excluded from the Ledzokuku district summary.) Ablekuma Central, covering the high-density informal commercial zones of Abeka, Darkuman, and Mallam, shows the second-highest mean SLFVI (0.592). Ga East, despite its mean SLFVI of 0.566, has 15.4 percent high-scoring points concentrated along the Dodowa-road and Madina corridors in the western part of the district; Ashaiman forms a separate Municipal Assembly and is not included in the Ga East summary. La Dade-Kotopon shows the lowest mean SLFVI (0.288) and the lowest high-risk share (0.3 percent), reflecting its partly planned residential character along the Labadi corridor.

A notable discrepancy arises in the validation comparison for Ledzokuku, where 71.2 percent of points score high or very high on the SLFVI but only 1.6 percent of building footprints in the district are classified as flood-exposed in the DEM-based building exposure layer. This divergence suggests that the streetscape AI is detecting drainage failure risk (blocked gutters, no visible drainage, high impervious coverage) that does not manifest as DEM-level inundation under the modelled threshold. This is an important finding for urban drainage policy, where the relevant risk is infrastructural failure rather than topographic inundation, particularly in flat coastal zones such as Teshie (Ledzokuku) and the adjacent Krowor/Nungua corridor where the DEM has limited power to differentiate risk.

### 5.7 Continuous Vulnerability Surface

The IDW interpolation of SLFVI to the 250-metre grid produced a surface with values ranging from 0.035 to 0.836 and a mean of 0.307. Grid cells located more than 2 kilometres from any observed Street View point were masked as data voids to avoid interpolation artefacts in unsampled areas; these are shown with a hatched overlay in Figure 5. The ordinary kriging surface, shown as a zoom-in on the highest-vulnerability hotspot area, ranged from 0.184 to 0.584 with a narrower spread reflecting the smoothing effect of the spatial covariance structure. Of all 16,271 grid cells in the study area (9,085 of which are masked as data voids beyond the 2 km observation buffer), 301 (1.8 percent) were classified as high or very high vulnerability under the IDW surface, corresponding to concentrated hotspots in inner-city informal areas, low-lying coastal fringes, and drainage-adjacent corridors.

Table 7 presents the distribution of grid cells by vulnerability class under both interpolation methods.

**Table 7.** Distribution of 250-metre grid cells by vulnerability class under IDW and ordinary kriging.

| Class | IDW cells | IDW % | Kriging cells | Kriging % |
|-------|-----------|-------|--------------|---------|
| Very low | 1,183 | 7.3% | 0 | 0.0% |
| Low | 13,853 | 85.1% | 14,312 | 88.0% |
| Moderate | 934 | 5.7% | 1,959 | 12.0% |
| High | 298 | 1.8% | 0 | 0.0% |
| Very high | 3 | 0.0% | 0 | 0.0% |

The kriging surface does not produce any cells in the very-low or high-to-very-high classes because the spherical variogram model regresses toward the global mean in sparsely sampled regions, suppressing extreme predictions. The IDW surface retains extreme values in the vicinity of extreme observed points. For policy reporting, the IDW surface is preferred for identifying specific hotspot locations, while the kriging surface provides a conservative estimate of the overall vulnerability pattern with associated prediction uncertainty that can be communicated to planners.

![Figure 5](outputs/figures/slfvi_surface_map.png)

**Figure 5.** Continuous SLFVI surface at 250-metre grid resolution across Greater Accra. Left panel: IDW surface across the full study area (range 0.035–0.836); cells beyond 2 km from any observed Street View point are masked with a hatched overlay to suppress extrapolation artefacts. Right panel: IDW surface zoom-in on the highest-vulnerability hotspot area (Odaw basin and inner Accra). High-vulnerability cells (IDW class: high or very high) are shown in red-orange. The ordinary kriging surface and prediction variance grid for the same extent are available in the supplementary data repository.

![Figure 6](outputs/figures/fig6_mapillary_panel.png)

**Figure 6.** Mapillary image panel showing four high-vulnerability (top row, EfficientNet vulnerability probability > 0.99) and four low-vulnerability (bottom row, probability < 0.01) road locations in Greater Accra. Images are geographically distributed across the study area (four spatial quadrants represented in each row). Captions beneath each image list the model's top-scoring flood-indicator labels with per-label probabilities for the 15 EfficientNet classes; no caption is inferred from general visual impression. Images are reproduced from Mapillary under a Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) licence; Mapillary panorama identifiers are shown in each tile.

---

## 6. Discussion

### 6.1 Streetscape AI as an Independent Flood Vulnerability Signal

A central finding of this study is that EfficientNet-derived streetscape features carry flood vulnerability information that is orthogonal to terrain and satellite variables. The SHAP analysis places two streetscape features (enet_no_visible_drainage and enet_heavy_impervious_surface) among the top ten predictors in the fusion model. The ablation in Table 2b shows that adding streetscape features to a terrain-plus-exposure model does not improve discrimination against the DEM-derived target (PR-AUC 0.946 versus 0.951 under standard CV), but that the streetscape-only model achieves ROC-AUC 0.744 in isolation — confirming that street-level conditions carry predictive information, but that this information concerns drainage-failure vulnerability rather than topographic flood exposure, and therefore does not additively improve a model calibrated to a DEM-based outcome. This distinction is the methodological foundation of the paper's argument: streetscape AI and DEM-based models are complementary, measuring different flood mechanisms rather than the same mechanism at different scales.

The physical mechanism is interpretable. Drainage infrastructure that is absent, informal, or obstructed cannot route runoff away from road surfaces regardless of topographic position. In flat coastal cities like Accra, where elevation differences of one to two metres can separate flooded from non-flooded streets, the drainage maintenance condition is often more predictive of actual flood impact than terrain alone. The streetscape AI operationalises this insight at scale without requiring costly field surveys.

### 6.2 The Coverage Bias Problem

The finding that Street View gap areas are substantially more vulnerable than covered areas (mean EfficientNet vulnerability probability 0.580 versus 0.384, a difference of 0.196) has implications that extend beyond this study. Urban AI applications that rely on Street View data systematically observe the more accessible, better-infrastructure parts of cities and draw inferences that underrepresent the most vulnerable populations. This coverage bias mirrors broader critiques of urban data availability in the Global South (Graham et al., 2012; Thatcher et al., 2016), where digital infrastructure investment tends to follow existing socioeconomic gradients. The phenomenon parallels the well-documented unevenness of volunteered geographic information (Haklay, 2010) and the concentration of remote sensing slum-mapping research on satellite-visible informal settlement boundaries rather than internal condition (Kuffer et al., 2016).

For flood vulnerability specifically, the bias is directional in a way that could mislead policy. A model that uses only covered areas would underestimate risk in precisely the neighbourhoods where drainage maintenance, infrastructure investment, and emergency planning are most needed. The Mapillary supplementation strategy introduced here partially addresses this bias but cannot fully resolve it, since Mapillary coverage in Ghanaian informal settlements is also sparse (14.6 percent of gap points). Community-based photography programs, drone surveys, and locally collected GoPro imagery represent complementary strategies for closing the gap that future research should pursue.

### 6.3 Interpreting District-Level Discrepancies

The divergence between Ledzokuku's high SLFVI scores and its low building flood exposure rate in the DEM-derived reference layer illustrates a fundamental methodological point about what different data sources measure. The DEM-derived reference layer assigns exposure based on DEM-derived inundation modelling at a threshold elevation. This approach captures topographic flood risk well in areas with meaningful elevation gradients but is poorly calibrated for flat, coastal informal settlements where the flood mechanism is drainage failure rather than topographic inundation.

The streetscape AI, by contrast, directly observes the absence of drainage infrastructure, the presence of impervious surfaces, and the visible markers of poor drainage maintenance. These signals predict drainage-failure flooding rather than topographic flooding. The methodological implication is that the two sources of evidence are complementary rather than competing: terrain and satellite data capture where topographic flood risk is high, while streetscape AI captures where drainage failure risk is high, and the combination of both improves overall predictive validity as demonstrated by the fusion model performance.

### 6.4 Limitations

Several limitations should be acknowledged.

First, the validation target is derived from MERIT DEM inundation thresholds, and several model inputs (elevation, TWI, depression depth, flow accumulation) are computed from the same DEM. This shared methodological ancestry means that the supervised slfvi_final variant may partly rediscover the rules used to construct the target, inflating apparent discrimination performance. This is the most significant limitation of the current study. Future validation against genuinely independent sources, such as NADMO flood event point locations, drainage complaint records, road closure data, media-reported flood incidents, or field-verified hotspot surveys, is essential before the index is used in operational planning.

Second, the spatial cross-validation design has two weaknesses. Spatial folds were constructed by assigning points to blocks defined by latitude and longitude quintile combinations, then randomly allocating blocks to five folds. This random block allocation does not guarantee the minimum inter-fold distance required to eliminate spatial autocorrelation leakage: two adjacent blocks assigned to different folds will have training and test points separated by as little as the block boundary, which in this study area can be as short as a few hundred metres — well within the spatial autocorrelation range of the terrain features. Stronger designs would use leave-one-district-out or leave-one-basin-out splits with an explicit exclusion buffer (e.g. 1 km) between training and test points, or geographically stratified holdouts that separate hydrological catchments rather than arbitrary grid cells. Additionally, the accuracy of the IDW and ordinary kriging interpolation surfaces was not evaluated via held-out prediction error (e.g. leave-one-out kriging variance or blocked cross-validation of the interpolated surface against unsampled points). The kriging prediction variance surface provides a theoretical uncertainty estimate but does not constitute empirical validation of interpolation accuracy. Future work should evaluate interpolation quality against independently measured or field-verified SLFVI values at unsampled locations, and should compare IDW and kriging performance using the same held-out evaluation set.

Third, the 5,000-point pilot sample represents approximately 7 percent of the full OSM road network in Greater Accra. Street View coverage was achieved for 76.7 percent of these points (unique-point rate), and spatial interpolation to unsampled areas introduces uncertainty that is largest in the periphery of the study area where known points are sparse.

Fourth, the training and evaluation design has an indirect contamination dependency. The teacher model used to generate pseudo-labels for the 1,600-image training set was trained on all 400 human-labelled images, including the 200 images that subsequently served as the held-out test set. Although the final student model was trained only on pseudo-labelled images and no human-labelled test image entered the training set directly, the teacher's learned representations of the test images were encoded into the pseudo-labels, creating an indirect link between teacher-training data and student evaluation data. Reported EfficientNet classification metrics (ROC-AUC 0.713, F1 0.560, per-label average precision) are therefore optimistic upper bounds. A clean evaluation design would hold out a spatially separated subset of the 400 human-labelled images before any model training, train the teacher only on the remaining labelled images, and evaluate on the untouched held-out subset. This is the recommended design for future annotated benchmark releases. The training dataset additionally uses pseudo-labels for 1,600 of its 2,000 images, reviewed at only a 20 percent sampling rate; residual label noise is likely and will be higher for subjective boundary labels (low_lying_street_form κ = 0.31, heavy_impervious_surface κ = 0.35) than for visually salient infrastructure labels (poor_road_condition κ = 0.83, open_gutter_present κ = 0.73). Inter-annotator agreement on the 60-image double-coded subset yielded a mean Fleiss' κ of 0.51 (Section 4.3), comparable to the range reported by Wang et al. (2024) for street-level disaster vulnerability annotation.

Fifth, Street View image dates in the dataset span multiple years (predominantly 2018 to 2024), introducing temporal inconsistency in a city where drainage conditions change seasonally and with infrastructure investment cycles. Image capture date was included as a covariate in the fusion model but does not fully resolve this inconsistency.

Sixth, the adaptive capacity component of the analytical SLFVI uses provisional proxies (road width class and distance from district boundary) rather than direct measures of service access or governance capacity, and should be treated as an approximation pending more suitable data.

Seventh, field validation at road-segment level was not conducted; the supplementary FloodSpots spot-check (Section 5.5) provides neighbourhood-scale face validity from an independent, event-based source, but does not permit quantification of discrepancies at the road-segment resolution at which the SLFVI is computed. Coordinate-level validation against NADMO event records or field-verified hotspot surveys remains a priority.

Eighth, population estimates are derived from coarse WorldPop gridded surfaces that do not accurately represent the highly variable density within informal settlements.

Ninth, the use of Google Street View imagery for machine learning model training raises a terms of service consideration. The Google Maps Platform Terms of Service restrict the use of Street View content to create training datasets or ML models; these restrictions apply to paid API customers as well as free-tier users. This study used Street View images ephemerally during the model training and feature extraction pipeline — images were not stored, cached beyond transient processing, or redistributed, and only derived feature vectors are included in the public data release. Nevertheless, the training use itself may not be covered by the standard API terms. This study is conducted as preliminary non-commercial academic research, and the authors are pursuing clarification from Google on research-use licensing. Researchers wishing to replicate or extend this methodology in contexts where terms of service compliance is a requirement are strongly encouraged to use permissively licensed imagery alternatives. Mapillary imagery is available under a Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) licence that explicitly permits use in ML training pipelines; KartaView and locally collected GoPro or smartphone imagery are additional options. The EfficientNet model architecture and training scripts released with this study are designed to run on any georeferenced street-level image collection and are not Street View-specific; Mapillary-only replication is feasible for most African cities with meaningful platform coverage.

---

## 7. Policy Implications

The SLFVI and the associated outputs of this study have direct applications for three groups of decision-makers in Greater Accra. Given that no road-segment-level field validation has been conducted, all outputs should be used as first-pass screening tools to prioritise field inspection and drainage maintenance audits rather than as standalone bases for direct operational resource allocation. A critical caveat for all uses: areas not identified as high risk by the SLFVI should not be treated as safe. The index covers only the 76.7 percent of sample points with confirmed Street View coverage; for approximately one quarter of road points in the study area, no streetscape observation was made, and their SLFVI values are spatially interpolated estimates carrying uncertainty that increases with distance from the nearest observed point. An absence of a high SLFVI score at a given location may reflect absence of data rather than absence of vulnerability, and field inspection should not be deprioritised for unsampled corridors on the basis of this index alone.

For drainage maintenance planning, the ranked list of road segments with the highest SLFVI scores provides a prioritised intervention agenda for metropolitan assembly public works departments. The top-scoring segments, concentrated in Ledzokuku, Ablekuma Central, and the Odaw corridor, represent locations where drainage cleaning, culvert inspection, and waste removal operations would most efficiently reduce flood risk. The data-driven prioritisation supplements existing ad hoc approaches based on complaint records and political decision-making.

For NADMO and emergency management, the continuous vulnerability surface and district-level vulnerability profiles support pre-positioning of emergency resources before the major rainy seasons. The finding that La Dade-Kotopon has substantially lower vulnerability than Ledzokuku, despite both districts being coastal, allows resource allocation to be proportionate to need rather than based on administrative unit alone.

For urban planners and infrastructure investment programmes, the identification of high-vulnerability informal settlements with no visible drainage provides evidence for prioritising drainage investment in areas that conventional survey methods tend to miss. The coverage bias finding reinforces the argument that urban data collection strategies in Ghanaian cities must specifically target informal settlement areas rather than relying on passive platforms whose spatial coverage follows existing infrastructure gradients.

---

## 8. Conclusion

This study has demonstrated that deep learning applied to street-level imagery can extract flood vulnerability information from urban streetscapes in Ghana that is independent of and complementary to terrain and satellite-based flood indicators. A model ablation shows that terrain features alone achieve ROC-AUC 0.989 against the DEM-derived reference, reflecting shared methodological ancestry, while the streetscape-only model achieves ROC-AUC 0.744, representing a non-terrain streetscape signal from street-level conditions relative to the DEM-derived reference. SHAP analysis places two EfficientNet-derived features among the top ten predictors in the fusion model. The analytical Street-Level Flood Vulnerability Index achieves moderate discrimination against a DEM-derived exposure reference (ROC-AUC 0.701); the supervised LightGBM variant achieves higher apparent performance but shares methodological ancestry with terrain inputs, and independent validation against field-verified or reported flood events remains a clear priority. The finding that Street View coverage gaps are systematically biased toward higher-vulnerability areas highlights a previously undocumented limitation of urban AI applications that rely on commercial street-level platforms and underscores the need for deliberate data collection strategies in informal settlements.

Future work should pursue three directions. First, expanding the image sample to the full OSM road network through additional API budget, Mapillary querying, and community-based collection would close coverage gaps and improve interpolation accuracy. Second, temporal analysis using multi-date Street View imagery would enable monitoring of drainage condition changes over time and evaluation of infrastructure investment impact. Third, the framework should be extended to additional Ghanaian cities including Kumasi, Tamale, and Cape Coast, and adapted for other West African urban contexts where the drainage infrastructure challenges are comparable.

The pipeline developed here is fully open-source and reproducible, with code, model weights, and derived datasets documented in the accompanying repository (https://github.com/dzimike/ghana-streetscape-flood). All outputs are provided at road-segment, grid-cell, district, and municipal level to support a range of planning and research uses.

---

## References

Amoako, C., and Frimpong Boamah, E. (2015). The three-dimensional causes of flooding in Accra, Ghana. International Journal of Urban Sustainable Development, 7(1), 109-129.

Anguelov, D., Dulong, C., Filip, D., Frueh, C., Lafon, S., Lyon, R., Ogale, A., Vincent, L., and Weaver, J. (2010). Google Street View: Capturing the world at street level. Computer, 43(6), 32-38.

Beven, K.J., and Kirkby, M.J. (1979). A physically based, variable contributing area model of basin hydrology. Hydrological Sciences Bulletin, 24(1), 43-69.

Biljecki, F., and Ito, K. (2021). Street view imagery in urban analytics and GIS: A review. Landscape and Urban Planning, 215, 104217.

Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. Computers, Environment and Urban Systems, 65, 126-139.

Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.

Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794.

Cressie, N.A.C. (1993). Statistics for Spatial Data. Wiley, New York.

Cutter, S.L., Boruff, B.J., and Shirley, W.L. (2003). Social vulnerability to environmental hazards. Social Science Quarterly, 84(2), 242-261.

Douglas, I., Alam, K., Maghenda, M., Mcdonnell, Y., McLean, L., and Campbell, J. (2008). Unjust waters: Climate change, flooding and the urban poor in Africa. Environment and Urbanization, 20(1), 187-205.

Ghana Statistical Service. (2021). 2021 Population and Housing Census General Report. Ghana Statistical Service, Accra.

Graham, M., Hale, S.A., and Stephens, M. (2012). Featured graphic: Digital divide: The geography of internet access. Environment and Planning A, 44(6), 1256-1258.

He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep residual learning for image recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 770-778.

IPCC. (2022). Climate Change 2022: Impacts, Adaptation and Vulnerability. Contribution of Working Group II to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change. Cambridge University Press.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30, 3146-3154.

Khosravi, K., Pourghasemi, H.R., Chapi, K., and Bahri, M. (2016). Flash flood susceptibility analysis and its mapping using different bivariate models in Iran: A comparison between Shannon's entropy, statistical index, and weighting factor models. Environmental Monitoring and Assessment, 188(12), 656.

Srivastava, S., Vargas-Munoz, J.E., and Tuia, D. (2019). Understanding urban landuse from the above and ground perspectives: A deep learning, multimodal solution. Remote Sensing of Environment, 228, 129-143.

Lundberg, S.M., and Lee, S.I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30, 4765-4774.

Naik, N., Kominers, S.D., Raskar, R., Glaeser, E.L., and Hidalgo, C.A. (2017). Computer vision uncovers predictors of physical urban change. Proceedings of the National Academy of Sciences, 114(29), 7571-7576.

Neuhold, G., Ollmann, T., Bulo, S., and Kontschieder, P. (2017). The Mapillary Vistas dataset for semantic understanding of street scenes. Proceedings of the IEEE International Conference on Computer Vision, 4990-4999.

Pekel, J.F., Cottam, A., Gorelick, N., and Belward, A.S. (2016). High-resolution mapping of global surface water and its long-term changes. Nature, 540(7633), 418-422.

Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. (2021). Learning transferable visual models from natural language supervision. Proceedings of the 38th International Conference on Machine Learning, 8748-8763.

Roberts, D.R., Bahn, V., Ciuti, S., Boyce, M.S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J.J., Schroder, B., Thuiller, W., Warton, D.I., Wintle, B.A., Hartig, F., and Dormann, C.F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography, 40(8), 913-929.

Schipper, E.L.F., and Pelling, M. (2006). Disaster risk, climate change and international development: Scope for, and challenges to, integration. Disasters, 30(1), 19-38.

Schumann, G.J.P., Neal, J.C., Voisin, N., Andreadis, K.M., Pappenberger, F., Phanthuwongpakdee, N., Hall, A.C., and Bates, P.D. (2013). A first large-scale flood inundation forecasting model. Water Resources Research, 49(10), 6248-6257.

Shepard, D. (1968). A two-dimensional interpolation function for irregularly-spaced data. Proceedings of the 1968 ACM National Conference, 517-524.

Sirko, W., Kashubin, S., Ritter, M., Annkah, A., Bouchareb, Y.S.E., Dauphin, Y., Keysers, D., Neumann, M., Cisse, M., and Quinn, J. (2021). Continental-scale building detection from high resolution satellite imagery. arXiv preprint arXiv:2107.12283.

Tan, M., and Le, Q.V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. Proceedings of the 36th International Conference on Machine Learning, 6105-6114.

Tehrany, M.S., Pradhan, B., Mansor, S., and Ahmad, N. (2015). Flood susceptibility assessment using GIS-based support vector machine model with different kernel types. Catena, 125, 91-101.

Vojinovic, Z., and Abbott, M.B. (2012). Flood risk and social justice: From quantitative to qualitative flood risk assessment and mitigation. IWA Publishing, London.

Wisner, B., Blaikie, P., Cannon, T., and Davis, I. (2004). At Risk: Natural Hazards, People's Vulnerability and Disasters. Second edition. Routledge, London.

WorldPop. (2020). Global High-Resolution Population Denominators Project. University of Southampton. DOI: 10.5258/SOTON/WP00674.

Yamazaki, D., Ikeshima, D., Tawatari, R., Yamaguchi, T., O'Loughlin, F., Neal, J.C., Sampson, C.C., Kanae, S., and Bates, P.D. (2017). A high-accuracy map of global terrain elevations. Geophysical Research Letters, 44(11), 5844-5853.

Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., and Torralba, A. (2017). Places: A 10 million image database for scene recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(6), 1452-1464.

Kuffer, M., Pfeffer, K., and Sliuzas, R. (2016). Slums from space: 15 years of slum mapping using remote sensing. Remote Sensing, 8(6), 455.

Thatcher, J., O'Sullivan, D., and Mahmoudi, D. (2016). Data colonialism through accumulation by dispossession: New metaphors for daily data. Environment and Planning D: Society and Space, 34(6), 990-1006.

Haklay, M. (2010). How good is volunteered geographical information? A comparative study of OpenStreetMap and Ordnance Survey datasets. Environment and Planning B: Planning and Design, 37(4), 682-703.

Cinnamon, J., and Jahiu, L. (2021). Panoramic street-level imagery in data-driven urban research: A comprehensive global review of applications, techniques, and practical considerations. ISPRS International Journal of Geo-Information, 10(9), 607.

Dekongmen, B.W., Kabo-Bah, A.T., Domfeh, M.K., Gyamfi, C., Amo-Boateng, M., Nkrumah, F., and Antwi, M. (2021). Flood vulnerability assessment in the Accra Metropolis, southeastern Ghana. Applied Water Science, 11(12), 1-16.

Gao, G., Retchless, D., Li, Y., Ye, X., Li, S., Huang, X., and Ning, H. (2024). Exploring flood mitigation governance by estimating first-floor elevation via deep learning and Google Street View in coastal Texas. Environment and Planning B: Urban Analytics and City Science, 51(2), 296-313.

Ghana Ground Up. (2026). FloodSpots — Greater Accra Flood Risk Map. Available at: https://floodspots.ghanagroundup.com/ (accessed July 2026).

Iordanov, G. (2025). Predicting infrastructure vulnerability to climate change from street level imagery. 2025 IEEE Afro-Mediterranean Conference on Artificial Intelligence (AMCAI).

Landis, J.R. and Koch, G.G. (1977). The measurement of observer agreement for categorical data. Biometrics, 33(1), 159-174.

Nkonu, R.S., Antwi, M., Amo-Boateng, M., and Dekongmen, B.W. (2023). GIS-based multi-criteria analytical hierarchy process modelling for urban flood vulnerability analysis, Accra Metropolis. Natural Hazards, 117(2), 1541-1568.

Velez, R., Calderon, D., Carey, L., Aime, C., Ghosh, K., and Lawson, A.B. (2022). Advancing data for street-level flood vulnerability: Evaluation of variables extracted from Google Street View in Quito, Ecuador. IEEE Open Journal of the Computer Society, 3, 51-61.

Wang, C., Antos, S.E., Gosling-Goldsmith, J.G., Triveno, L.M., and Calderon, D. (2024). Assessing climate disaster vulnerability in Peru and Colombia using street view imagery: A pilot study. Buildings, 14(1), 14.

---

*Word count (approximate): 9,800 words excluding references and tables.*


*Code and data availability: All analysis code is openly available at https://github.com/dzimike/ghana-streetscape-flood (Zenodo DOI: to be assigned on publication). Model weights and derived geospatial datasets are deposited in the same repository. The 590 Mapillary images used in this study were obtained through the Mapillary API under a CC BY-SA 4.0 licence and are identified by the panorama IDs reported in the supplementary data table. Raw Street View imagery is not stored or redistributed; only derived streetscape feature vectors are included in the release dataset. Researchers wishing to replicate this methodology without Google Maps Platform terms of service constraints are encouraged to use Mapillary (CC BY-SA 4.0) or locally collected imagery; the training pipeline in the repository is imagery-source-agnostic. See Section 6.4 for a full discussion of terms of service considerations.*
