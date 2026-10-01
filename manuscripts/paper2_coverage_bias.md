# Street-Level Data Voids and Flood Vulnerability Inequality in Urban Ghana

**[Author names]**
[Affiliations]
Corresponding author: dzimike.md@gmail.com

---

## Abstract

Street-level imagery platforms offer fine-grained evidence for urban flood vulnerability assessment, yet systematic gaps in platform coverage may concentrate analytical blind spots in the very neighbourhoods at greatest risk. Using a stratified sample of 5,000 road segments drawn from Greater Accra, Ghana, this paper quantifies the spatial structure of Google Street View coverage gaps and assesses whether those gaps introduce systematic bias in flood vulnerability estimation. Of the 5,000 queried points, 1,088 (21.8%; bootstrap 95% CI 20.7 to 22.9%) returned no panorama. Coverage deficits were strongly concentrated by road class: service roads recorded a gap rate of 32.0% and residential roads 24.1%, whereas trunk roads and motorways had no gaps. A chi-square test confirmed that the gap distribution differs significantly across road types (chi-square = 276.63, df = 7, p < 0.001). Geospatial comparison revealed that gap areas are on average 235 metres closer to waterways than covered areas (672.8 m versus 907.7 m; Cohen's d = 0.317), and have 32% higher surrounding building density (46.3 versus 35.2 buildings per 100 m radius). These structural properties suggest that the uncovered road network segment is more flood-exposed, not less, than the covered segment. Supplementary queries to Mapillary provided partial imagery for 159 of the 1,088 gap points (14.6%). Inverse-distance-weighted spatial imputation using vulnerability scores from the five nearest covered neighbours produced imputed mean values indistinguishable from the covered set (bias index B = −0.010; d = 0.04), confirming that IDW smoothing suppresses rather than resolves the gap problem. We propose a coverage bias index and a gap-flagged uncertainty layer that accompany the Street-Level Flood Vulnerability Index to make the limitation visible to decision-makers. The findings contribute to the growing literature on digital spatial inequality and motivate explicit gap characterisation protocols for urban AI applications in the Global South.

**Keywords:** street view coverage, spatial bias, flood vulnerability, digital inequality, urban Ghana, Accra, IDW imputation

---

## 1. Introduction

Platforms that deliver street-level imagery at continental scale have transformed the evidence base available to urban planners, public health researchers, and climate adaptation practitioners. Google Street View, first deployed in 2007, and community-sourced platforms such as Mapillary now provide multi-directional photographs for billions of road locations worldwide (Anguelov et al., 2010; Warburg et al., 2020). These platforms have been used to estimate poverty rates from infrastructure quality, to audit pedestrian accessibility, to measure urban greenery, and, most recently, to derive flood vulnerability indicators from visible drainage, road surface condition, solid waste accumulation, and informal infrastructure near drainage channels (Gebru et al., 2017; Biljecki and Ito, 2021). Each application shares a common assumption: that imagery is sufficiently representative of the road network to support inference.

That assumption is not neutral with respect to geography. Coverage has been shown to be correlated with road class, urban density, administrative priority, and the commercial logic of platforms that weight high-traffic corridors (Haklay, 2010). Studies of OpenStreetMap completeness in sub-Saharan Africa have documented systematic under-representation of informal settlements, secondary roads, and peri-urban areas (Haklay and Weber, 2008; Mooney and Corcoran, 2012). If street-level imagery replicates these patterns, analytical models trained and applied only on covered road segments may systematically underestimate flood risk in the very communities that are most vulnerable.

This paper addresses three questions. First, what fraction of a stratified urban road sample in Greater Accra has no Street View panorama, and how is that fraction distributed across road class and flood exposure stratum? Second, do gap areas differ from covered areas in their geospatial flood hazard properties? Third, can supplementary sources and spatial imputation close the gap, or do they introduce further distortions?

The contribution is methodological and diagnostic. We do not claim to have solved the coverage problem; we provide a rigorous characterisation of it, a formal bias index, and a set of recommendations for communicating coverage-conditioned uncertainty in urban vulnerability analyses. The work is grounded in Greater Accra but the methods generalise to any city where street-level imagery is used as a flood or infrastructure intelligence source.

---

## 2. Background

### 2.1 Street View Coverage and Its Determinants

Google Street View collects imagery using a fleet of camera-equipped vehicles that traverse road networks according to proprietary routing schedules (Anguelov et al., 2010). Coverage is determined by whether a vehicle has driven a road segment within the platform's operating history. In high-income countries with dense formal road networks, coverage reaches near-completeness for all road classes above private driveways. In sub-Saharan African cities, coverage is fragmentary below the primary and secondary road hierarchy and is absent from many informal lanes, unpaved service tracks, and alleyways that carry the majority of local pedestrian and commercial traffic.

Biljecki and Ito (2021) reviewed 641 studies employing street view imagery across disciplines and found that geographic representation was concentrated in North America, Western Europe, Japan, and major Chinese cities. Studies from low-income countries were sparse, and methodological discussions of representativeness were largely absent. The review identified coverage gaps as a key research frontier for the field.

For VGI platforms such as Mapillary, coverage reflects the spatial distribution of contributors rather than corporate routing. Warburg et al. (2020) documented that Mapillary's spatial coverage tracks population density and cycling infrastructure density, creating complementary rather than identical gaps to Google Street View. In informal urban areas of West Africa, both platforms tend to underperform formal road sensors.

### 2.2 Spatial Bias and Inequality in Urban Data

The concept of spatial data inequality draws on Haklay's (2010) demonstration that OpenStreetMap quality in England was inversely correlated with deprivation, reflecting the geographic distribution of volunteer contributors. Graham et al. (2014) extended this argument to show that internet-based knowledge production is geographically skewed toward urban cores, English-speaking countries, and wealthy neighbourhoods.

In the context of urban flood risk, the consequences of spatial data gaps are not merely analytical. Flood exposure analyses that draw on biased spatial samples will produce maps that systematically understate risk in under-mapped areas. Rentschler and Salhab (2020) estimated that 1.81 billion people in 188 countries are exposed to 1-in-100-year floods, with the burden concentrated in South and Southeast Asia and sub-Saharan Africa. If data pipelines used to identify and serve these populations are systematically blind to the most informal and vulnerable road segments, the analytical gap compounds the social gap.

### 2.3 Coverage Bias in AI-Assisted Vulnerability Assessment

When computer vision models are applied to street-level imagery, coverage gaps produce a specific failure mode: the model is not wrong about the images it sees, but it is silent about the places it cannot see. Silence is not the same as low risk. In flood vulnerability estimation, the absence of a prediction may indicate that a road segment is well-maintained and low-risk, or it may indicate that the imaging platform has never visited it. Without explicit coverage tracking, these two conditions are indistinguishable in the output.

Gebru et al. (2017) demonstrated that deep learning models applied to Google Street View imagery can estimate neighbourhood income levels with surprising accuracy, but their training and evaluation were restricted to US cities with near-complete coverage. The present study extends this line of inquiry to a West African informal city context in which coverage gaps are not incidental but systematic, and where the areas absent from the imaging platform are precisely the areas of greatest analytical interest for flood vulnerability mapping.

---

## 3. Study Area and Data

### 3.1 Study Area

The study covers the Accra Metropolitan Area and surrounding municipalities in the Greater Accra Region of Ghana: Ga East, Ga West, Ga South, La Dade-Kotopon, Ledzokuku, Krowor, Tema, Ashaiman, and Weija-Gbawe. The region has an estimated population of approximately 5.3 million and is one of the fastest-growing metropolitan areas in West Africa. It has experienced recurrent and severe urban flooding, with major events recorded in 2011, 2015, 2018, and 2021. Informal settlements occupy significant portions of low-lying and flood-prone land, and the formal road network is overlaid by dense secondary and service road networks that serve these areas.

### 3.2 Road Segment Sampling

The sampling frame was derived from OpenStreetMap road data downloaded for the study boundary. Roads were cleaned by class, split into 100-metre segments, and 5,000 sample points were generated at segment centroids. Points were stratified across two flood exposure strata and three settlement types. Flood strata were defined using a composite of DEM-derived local depressions, distance to OSM waterways and drains, and FABDEM elevation. The two strata were high-risk proximity (n = 3,000) and low-risk reference (n = 2,000). The three settlement types were formal residential, informal high-density, and commercial or mixed-use.

### 3.3 Street View Metadata Query

Each of the 5,000 sample points was submitted to the Google Street View Metadata API. The API returns status codes indicating whether a panorama exists within a defined search radius (set to 50 metres). Panoramas with status OK were retained; those with status ZERO_RESULTS or UNKNOWN_ERROR were classified as gap points. Metadata for OK panoramas included panorama ID, latitude, longitude, capture date, copyright, and heading.

### 3.4 Mapillary Supplementary Query

Gap points were queried against the Mapillary API using a bounding-box search with a 100-metre radius. Mapillary imagery was used as a supplementary coverage source; it was not integrated into the main EfficientNet-based vulnerability assessment due to differences in image quality, camera parameters, and heading consistency. The Mapillary query returned 620 images from 159 distinct gap-point locations. Of these, 590 were retained after quality screening (motion blur, extreme pitch, or heading inconsistency exceeding 45°) and were used for EfficientNet-based vulnerability inference in a related study; the full 620-image count is used here solely to characterise Mapillary coverage extent.

### 3.5 Geospatial Features

For all 5,000 sample points, the following geospatial features were derived: elevation from FABDEM, local terrain depression depth, slope, distance to the nearest OSM waterway, building count within a 100-metre radius from Google Open Buildings, and population density from WorldPop. These features were used to characterise the geospatial properties of gap versus covered areas.

---

## 4. Methods

### 4.1 Coverage Gap Rate and Bootstrap Confidence Interval

The observed gap rate is defined as

$$\hat{g} = \frac{n_{\text{gap}}}{n_{\text{total}}}$$

where $n_{\text{gap}}$ is the count of points with status ZERO_RESULTS or UNKNOWN_ERROR and $n_{\text{total}} = 5{,}000$. A non-parametric bootstrap confidence interval for $\hat{g}$ was computed by resampling the binary coverage vector 10,000 times with replacement and taking the 2.5th and 97.5th percentiles of the resulting distribution (Efron and Tibshirani, 1993).

### 4.2 Coverage Probability Model

To identify covariates associated with coverage probability, a logistic regression model was fitted:

$$\log \frac{P(\text{covered}_i)}{1 - P(\text{covered}_i)} = \beta_0 + \sum_{j=1}^{J} \beta_j \, x_{ij}$$

where the binary outcome (covered = 1, gap = 0) was regressed on road-class dummy variables (reference category: motorway) and a binary flood-stratum indicator (high-risk proximity versus low-risk reference). A Pearson chi-square test was used to assess whether the observed coverage distribution across road classes differed from the expected uniform distribution.

### 4.3 Geospatial Comparison and Cohen's d

For each geospatial feature $f$, the effect size of the covered-versus-gap contrast was quantified using Cohen's d:

$$d_f = \frac{\bar{f}_{\text{gap}} - \bar{f}_{\text{covered}}}{\hat{\sigma}_{\text{pooled}}}$$

where $\hat{\sigma}_{\text{pooled}} = \sqrt{(\hat{\sigma}_{\text{gap}}^2 + \hat{\sigma}_{\text{covered}}^2) / 2}$.

By convention, d less than 0.20 is negligible, 0.20 to 0.50 is small, 0.50 to 0.80 is medium, and greater than 0.80 is large (Cohen, 1988).

### 4.4 Coverage Bias Index

The coverage bias index $B$ is defined as the mean difference between IDW-imputed vulnerability scores for gap points and observed vulnerability scores for covered points. IDW imputation assumes that closer observations are more informative, a formalisation of Tobler's (1970) first law of geography:

$$B = \bar{p}_{\text{gap}}^{\text{IDW}} - \bar{p}_{\text{covered}}$$

where $\bar{p}_{\text{gap}}^{\text{IDW}}$ is computed by inverse-distance-weighted interpolation from the $k = 5$ nearest covered neighbours. Positive $B$ indicates that gap areas have higher inferred vulnerability than covered areas; negative $B$ indicates the reverse. A $B$ close to zero indicates that IDW imputation is approximately unbiased but does not confirm that gap areas have similar vulnerability to covered areas, because IDW smoothing suppresses local heterogeneity.

$$\hat{p}_{\text{gap},i}^{\text{IDW}} = \frac{\sum_{j=1}^{k} w_{ij} \, p_j}{\sum_{j=1}^{k} w_{ij}}, \quad w_{ij} = \frac{1}{d_{ij} + \epsilon}$$

where $d_{ij}$ is the Euclidean distance between gap point $i$ and covered neighbour $j$, and $\epsilon = 10^{-6}$ prevents division by zero.

### 4.5 Adjusted Uncertainty Layer

The kriging prediction variance from the main vulnerability mapping pipeline was used as the base uncertainty estimate. For grid cells whose centroids fall within 200 metres of at least one gap point and more than 200 metres from any covered point, an additional gap penalty was applied to the prediction variance:

$$\hat{\sigma}^2_{\text{adj},c} = \hat{\sigma}^2_{\text{krig},c} \times (1 + \alpha \, g_c)$$

where $g_c \in \{0, 1\}$ is a binary indicator for gap-adjacent grid cells and $\alpha = 0.5$ is a conservatively chosen penalty factor.

---

## 5. Results

### 5.1 Overall Coverage and Gap Rate

Of the 5,000 queried road points, 3,912 (78.2%) returned a valid Street View panorama (status OK), 1,086 (21.7%) returned ZERO_RESULTS, and 2 (0.04%) returned UNKNOWN_ERROR. The total gap count was 1,088, yielding an observed gap rate of 21.8% (bootstrap 95% CI 20.7 to 22.9%).

### 5.2 Coverage Inequality by Road Class

Figure 2 shows the spatial distribution of covered (blue) and gap (red) road points across the metropolitan area. Gaps are visually concentrated in the inner residential and peri-urban zones rather than being randomly dispersed. Figure 3 presents gap rates by road class in ranked order. Table 1 presents the same data numerically. The distribution of gaps was highly unequal across road classes. Trunk roads and motorways recorded zero gaps. Tertiary and secondary roads recorded gap rates below 2%. Residential roads, which account for 59.6% of all sampled points (n = 2,978), recorded a gap rate of 24.1%. Service roads, which carry local access traffic to informal lanes, recorded the highest gap rate at 32.0%.

**Table 1. Street View coverage by road class.**

| Highway class | n | Covered | Gap | Gap rate (%) |
|---|---|---|---|---|
| Trunk | 111 | 111 | 0 | 0.0 |
| Motorway | 13 | 13 | 0 | 0.0 |
| Secondary | 166 | 164 | 2 | 1.2 |
| Tertiary | 264 | 261 | 3 | 1.1 |
| Primary | 164 | 160 | 4 | 2.4 |
| Unclassified | 241 | 220 | 21 | 8.7 |
| Residential | 2,978 | 2,260 | 718 | 24.1 |
| Service | 1,063 | 723 | 340 | 32.0 |
| **Total** | **5,000** | **3,912** | **1,088** | **21.8** |

The Pearson chi-square test was highly significant (chi-square = 276.63, df = 7, p < 0.001), confirming that gap rates differ non-randomly across road classes. The logistic regression confirmed road class as the dominant predictor of coverage probability. Relative to motorway roads (the reference category), residential roads had a log-odds coefficient of 1.792 and service roads had a coefficient of 2.180, both indicating lower coverage probability. The flood-stratum coefficient was small (log-odds 0.146), indicating that the high-risk versus low-risk sampling stratum contributed little additional explanatory power beyond road class.

Coverage by flood stratum was 77.7% for high-risk proximity points and 79.1% for low-risk reference points, a difference of 1.4 percentage points that is not substantively meaningful.

### 5.3 Geospatial Properties of Gap Areas

Figure 4 presents side-by-side box plots of five geospatial features for covered versus gap points. Table 2 provides the corresponding summary statistics. The most pronounced difference was in distance to the nearest waterway. Gap areas were, on average, 235 metres closer to waterways than covered areas (672.8 m versus 907.7 m; Cohen's d = 0.317). Gap areas also had higher surrounding building density (46.3 versus 35.2 buildings per 100-metre radius; d = 0.121). Elevation and slope were similar between groups (d = 0.071 and 0.054 respectively).

**Table 2. Geospatial properties of covered versus gap road points.**

| Feature | Covered mean | Gap mean | Cohen's d |
|---|---|---|---|
| Elevation (m) | 32.4 | 31.0 | 0.071 |
| Slope (degrees) | 2.6 | 2.7 | 0.054 |
| Local depression (m) | 0.74 | 0.80 | 0.028 |
| Building count (100 m radius) | 35.2 | 46.3 | 0.121 |
| Distance to waterway (m) | 907.7 | 672.8 | 0.317 |
| Population density (per km²) | 11,291 | 8,609 | 0.042 |

The direction of the waterway distance effect is important. Gap areas are systematically closer to waterways than covered areas. In Greater Accra, proximity to waterways is a known flood risk factor; areas within 200 to 500 metres of channels, streams, and drainage networks are among the most flood-prone zones in the metropolitan area (NADMO, 2021). The coverage gap therefore concentrates analytical blind spots in areas where proximity to surface water drainage is highest.

The lower population density in gap areas (8,609 versus 11,291 per km²) initially appears contradictory: if informal dense areas are under-covered, one might expect higher density in gap areas. Inspection reveals that the highest-density sub-areas in the sample are well-covered commercial corridors with formal road classifications, while many informal peri-urban areas have large parcels with moderate density but unpaved service roads, which carry the highest gap rates.

### 5.4 Mapillary Supplementary Coverage

Of the 1,088 gap points, 159 (14.6%) were within 100 metres of at least one Mapillary image. The 620 Mapillary images came from 159 unique spatial locations in the gap set. Mapillary coverage was distributed across the gap set without a strong spatial pattern, reflecting the opportunistic nature of community-sourced coverage rather than systematic routing. In several waterway-adjacent gap clusters, no Mapillary images were present. The partial fill offered by Mapillary is therefore insufficient to resolve the systematic bias identified in Section 5.3.

### 5.5 IDW Spatial Imputation and the Bias Index

Inverse-distance-weighted imputation using the five nearest covered neighbours produced a mean imputed vulnerability probability of 0.374 for gap points, compared to 0.384 for covered points (B = −0.010; d = 0.04). Figure 5 maps the spatial distribution of IDW-imputed vulnerability probability for the 1,088 gap points alongside their covered neighbours (panel a) and the per-gap-point coverage bias index B (panel b). The near-zero global bias index and negligible effect size reflect the well-known smoothing property of IDW interpolation: spatially autocorrelated values tend to converge at interpolated points toward their neighbours' values, suppressing any genuine difference between gap and covered areas. IDW imputation does not resolve the gap problem because it cannot distinguish between a truly low-risk location and one that happens to be surrounded by low-risk covered neighbours.

A critical examination reveals the limitation. Gap areas are demonstrably closer to waterways (d = 0.317) and have higher building density (d = 0.121), suggesting that true vulnerability in gap areas may be higher than IDW imputation implies. The disconnect between the spatial property comparison and the IDW result occurs because IDW draws on vulnerability scores that were generated by the EfficientNet model from visual evidence in the images. In gap areas, visual evidence is absent; the spatial signal of proximity to waterways and dense building fabric is not captured in the model's predictions.

### 5.6 Adjusted Uncertainty Layer

Application of the adjusted uncertainty formula increased prediction variance by 50% (alpha = 0.5) for the 342 grid cells that were gap-adjacent and more than 200 metres from any covered point. These cells are predominantly located in service-road corridors adjacent to waterway channels and in the most informal peri-urban settlement zones. The adjusted uncertainty layer is exported as part of the standard output package and is recommended for display alongside any SLFVI vulnerability map.

---

## 6. Discussion

### 6.1 The Structural Nature of the Coverage Gap

The gap structure identified in this study is not random noise. It is a systematic feature of how Street View coverage is assembled: vehicles follow navigable roads, and navigability is correlated with road class, pavement, width, and institutional recognition. Service roads in informal Accra are often unpaved, narrow, and unmapped in Google's routing database. Residential roads in peri-urban expansion zones may be newly formed and absent from vehicle routing schedules. The result is a coverage pattern that mirrors urban inequality: formal roads at the top of the hierarchy are fully covered, while the informal road network that serves the majority of low-income residents carries the highest gap rates.

This pattern has a direct consequence for flood vulnerability assessment. Street-level flood indicators, including drainage blockages, solid waste near drains, unpaved road shoulders, and informal structures encroaching on drainage channels, are most prevalent in the informal residential and service road corridors that carry the highest gap rates. A model that is silent about these road segments does not produce low-risk estimates for them; it produces no estimate at all. When gap points are omitted from vulnerability maps without explicit flagging, the practical effect is that high-vulnerability informal areas become white space on the map.

### 6.2 Waterway Proximity as a Risk Amplifier

The finding that gap areas are 235 metres closer to waterways than covered areas (d = 0.317) is the most policy-relevant finding of this study. In the Accra urban flood context, waterway proximity is the strongest terrain-based predictor of flood exposure (Rentschler and Salhab, 2020; NADMO, 2021). Service and residential roads adjacent to watercourses, seasonal streams, and open drainage channels are precisely the roads that carry the highest gap rates. This convergence suggests that standard flood vulnerability mapping using Street View imagery may systematically underestimate risk in the areas of greatest actual exposure.

The building density finding reinforces this interpretation. Gap areas have 32% higher building density than covered areas. High building density in informal settlements adjacent to waterways is a well-documented flood risk amplifier: impervious surface increases surface runoff, informal structures reduce effective channel width, and dense settlement patterns reduce pedestrian escape routes. The combination of proximity to waterways and high building density in gap areas should be treated as a signal of elevated, not negligible, flood risk.

### 6.3 Limitations of Supplementary Data Sources

Mapillary provided partial coverage for only 14.6% of gap points, and those images were unevenly distributed across the gap set. Community-sourced platforms are valuable supplements but cannot substitute for systematic coverage collection. Their distribution reflects contributor geography rather than the geography of flood risk.

IDW spatial imputation is a widely used and technically defensible gap-filling approach, but the near-zero bias index conceals rather than resolves the underlying problem. The IDW-imputed vulnerability values are essentially the local spatial mean of the covered points, which suppresses heterogeneity and does not reflect the physical properties of gap areas. Future work should explore model-based imputation that conditions on geospatial covariates, including waterway proximity and building density, to produce more defensible gap estimates.

### 6.4 Comparison with Global Findings

The 21.8% gap rate observed in this Greater Accra sample is substantially higher than rates reported for formal road networks in European or North American cities, where gaps are typically limited to gated communities, new developments, and private access roads. It is broadly consistent with gap rates reported for informal settlement road networks in studies of VGI completeness in sub-Saharan African cities, where road map completeness for residential and service classes has been estimated at 60 to 80% of the true network (Haklay, 2010; Mooney and Corcoran, 2012). The Street View API gap rate measures a different property (whether an image exists, not whether a road is mapped), but the magnitude and road-class gradient are consistent with the structural patterns documented in the VGI completeness literature.

### 6.5 Implications for Practice

Three practical recommendations follow from the findings. First, any vulnerability analysis that uses street-level imagery should report gap rates by road class and flood stratum as standard output alongside the vulnerability map. The gap rate alone contextualises what fraction of the analysis is based on actual visual evidence. Second, gap-adjacent grid cells should be flagged with elevated uncertainty in any spatial output provided to decision-makers. The adjusted uncertainty layer described in Section 4.5 is a minimum standard. Third, supplementary data collection for gap areas should be prioritised using the most flood-exposed gap clusters identified by waterway proximity and building density. Community-based geo-tagged photography, drone imagery where permitted, and targeted Mapillary campaigns offer practical routes to closing the most consequential gaps.

---

## 7. Conclusion

This paper has documented a systematic and spatially structured coverage gap in Google Street View imagery for a stratified road segment sample in Greater Accra, Ghana. The 21.8% gap rate is concentrated in service and residential road classes, which are also the road types most closely associated with informal settlement patterns. Gap areas are closer to waterways and have higher building density than covered areas, suggesting that the uncovered road network is, on average, more flood-exposed than the covered one. IDW spatial imputation produces a near-zero bias index but suppresses the spatial heterogeneity that makes gap areas genuinely different from their neighbours. Mapillary supplementation addresses only 14.6% of gaps and is not spatially targeted to the most flood-exposed locations.

The central contribution is a formal framework for coverage-conditioned uncertainty reporting, including the coverage bias index $B$, the gap-flagged uncertainty layer, and a set of audit statistics that allow future studies to characterise their own coverage limitations. In the Global South, where informal road networks carry the largest populations at flood risk, street-level data voids are not a peripheral concern but a first-order methodological challenge for AI-assisted urban vulnerability assessment.
---

## Acknowledgements

[To be completed on submission.]

---

## References

Anguelov, D., Dulong, C., Filip, D., Frueh, C., Lafon, S., Lyon, R., Ogale, A., Vincent, L., and Weaver, J. (2010). Google Street View: Capturing the world at street level. Computer, 43(6), 32–38.

Biljecki, F., and Ito, K. (2021). Street view imagery in urban analytics and GIS: A review. Landscape and Urban Planning, 215, 104217.

Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Erlbaum, Mahwah, NJ.

Efron, B., and Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman and Hall, New York.

Gebru, T., Krause, J., Wang, Y., Chen, D., Deng, J., Aiden, E. L., and Fei-Fei, L. (2017). Using deep learning and Google Street View to estimate the demographic makeup of neighborhoods across the United States. Proceedings of the National Academy of Sciences, 114(50), 13108–13113.

Graham, M., Hogan, B., Straumann, R. K., and Medhat, A. (2014). Uneven geographies of user-generated information: Patterns of increasing informational poverty. Annals of the Association of American Geographers, 104(4), 746–764.

Haklay, M. (2010). How good is volunteered geographical information? A comparative study of OpenStreetMap and Ordnance Survey datasets. Environment and Planning B: Planning and Design, 37(4), 682–703.

Haklay, M., and Weber, P. (2008). OpenStreetMap: User-generated street maps. IEEE Pervasive Computing, 7(4), 12–18.

Mooney, P., and Corcoran, P. (2012). The annotation process in OpenStreetMap. Transactions in GIS, 16(4), 561–580.

NADMO (2021). Ghana National Disaster Management Organisation Annual Flood Situation Report 2021. Government of Ghana, Accra.

Rentschler, J., and Salhab, M. (2020). People in Harm's Way: Flood Exposure and Poverty in 189 Countries. World Bank Policy Research Working Paper No. 9447. World Bank, Washington, DC.

Tobler, W. R. (1970). A computer movie simulating urban growth in the Detroit region. Economic Geography, 46(Supplement), 234–240.

Warburg, P., Weinzaepfel, P., and Sattler, T. (2020). Mapillary Street-Level Sequences: A dataset for lifelong place recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2013–2022.

WorldPop (2020). WorldPop Open Population Repository, School of Geography, University of Southampton. DOI: 10.5258/SOTON/WP00645.

---

## Figure Captions

![Figure 1](outputs/figures/fig1_paper2_workflow.png)

**Figure 1.** Methodological workflow for the coverage gap analysis. Rounded boxes represent processing steps; rectangular boxes represent data or intermediate outputs; the purple terminal box represents final outputs. Arrows show data flow direction. Stage labels (S1–S5) correspond to the stages described in the Methods section.

![Figure 2](outputs/figures/fig2_paper2_coverage_map.png)

**Figure 2.** Spatial distribution of Street View coverage across the Greater Accra road-segment sample (n = 5,000 points). Blue points indicate valid panoramas (covered; n = 3,912; 78.2%). Red points indicate coverage gaps, where the Metadata API returned ZERO_RESULTS or UNKNOWN_ERROR (n = 1,088; 21.8%). Gap points are plotted at a larger size for visibility.

![Figure 3](outputs/figures/fig3_paper2_gap_rates.png)

**Figure 3.** Street View coverage gap rate by OpenStreetMap road class, ranked from highest to lowest gap rate. The dashed line marks the overall mean gap rate of 21.8%. Bar colour denotes gap severity: red greater than 15%, orange 5–15%, blue below 5%. Sample size per class is annotated. Chi-square test across all classes: chi-square = 276.63, df = 7, p < 0.001.

![Figure 4](outputs/figures/fig4_paper2_geo_comparison.png)

**Figure 4.** Box plots comparing five geospatial properties between covered road points (blue; n = 3,912) and coverage-gap road points (red; n = 1,088). Each panel reports Cohen's d for the gap-minus-covered contrast and the conventional effect-size label. The distance-to-waterway panel (d = 0.317, small-to-medium) is the most policy-relevant finding: gap areas are systematically closer to waterways than covered areas.

![Figure 5](outputs/figures/fig5_paper2_idw_bias_index.png)

**Figure 5.** IDW spatial imputation and the coverage bias index for gap points across Greater Accra. Panel (a) shows all 5,000 road points: covered points (circles) are coloured by their actual EfficientNet vulnerability probability; gap points (triangles, n = 1,088) are coloured by their IDW-imputed vulnerability probability derived from the five nearest covered neighbours weighted by inverse distance. Both panels use the same RdYlBu_r colormap (0 to 1). Panel (b) isolates the coverage bias index B_i = p̂_gap,i − p̄_covered for each gap point using a diverging red-blue colormap centred at zero; covered points are shown in light grey as a spatial reference. Red triangles indicate gap locations where imputed vulnerability exceeds the covered-point mean (positive bias); blue triangles indicate locations where imputed vulnerability falls below the mean (negative bias). The annotation box reports the global statistics: mean B = −0.010, covered mean = 0.384, IDW gap mean = 0.374. The near-zero global bias and the spatially mixed pattern of positive and negative residuals confirm that IDW suppresses spatial heterogeneity rather than recovering the true vulnerability contrast between gap and covered areas.

---

*Code and data availability: All analysis code is openly available at https://github.com/dzimike/ghana-streetscape-flood (Zenodo DOI: to be assigned on publication). The coverage-gap analysis and bias index outputs reported in this paper are reproducible from the scripts in the repository. Raw Street View imagery is not redistributed in accordance with Google Maps Platform terms of service.*

*Manuscript prepared 2026-07-18. Target journals: Nature Cities; Environment and Planning B: Urban Analytics and City Science; Big Data and Society.*



