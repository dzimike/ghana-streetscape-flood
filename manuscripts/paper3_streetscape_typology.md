# Streetscape Archetypes of Urban Flood Vulnerability in Greater Accra, Ghana

**[Author names]**
[Affiliations]
Corresponding author: dzimike.md@gmail.com

---

## Abstract

Understanding why some urban areas flood while physically adjacent areas do not requires moving beyond composite vulnerability indices toward the identification of recurring spatial patterns that reflect distinct combinations of infrastructure failure. This paper derives six streetscape archetypes of urban flood vulnerability in Greater Accra, Ghana, using unsupervised k-means clustering applied to segment-level EfficientNet-B0 predictions across 15 binary flood-indicator labels for 3,836 road points. Silhouette and Calinski-Harabasz validation indicated an optimal partition at k = 2, but a k = 6 solution was retained for policy utility, yielding a global silhouette coefficient of 0.2301. The six archetypes differ meaningfully in flood vulnerability profile, spatial distribution, and intervention priority. The highest-vulnerability archetype (Type I: Dense Impervious Drainage Void; 19.3% of segments; mean vulnerability probability 0.734) is characterised by heavy impervious surface coverage (mean 0.784) and the near-complete absence of visible drainage infrastructure (no-drainage indicator mean 0.718), despite relatively good road condition. The lowest-vulnerability archetype (Type VI: Low-Risk Formal Corridor; 28.2% of segments; mean probability 0.150) comprises well-maintained paved roads on the primary and trunk network. Intermediate archetypes reflect compound drainage failure with co-occurring erosion and solid waste (Type II), waterway-adjacent infrastructure with mixed drainage health (Type III), open-gutter waste accumulation (Type IV), and residential roads showing moderate maintenance deficits (Type V). The archetype framework provides a theoretically grounded and operationally tractable basis for prioritising drainage maintenance, infrastructure investment, and community engagement by road-segment type rather than by aggregate vulnerability score.

**Keywords:** streetscape archetypes, k-means clustering, flood vulnerability, urban typology, EfficientNet, Greater Accra, Ghana

---

## 1. Introduction

Composite flood vulnerability indices — such as the Street-Level Flood Vulnerability Index (SLFVI), which combines terrain, hydrological, exposure, and streetscape indicators into a single road-segment score — aggregate multiple dimensions of hazard, exposure, sensitivity, and adaptive capacity. Composite scores are useful for ranking locations and producing vulnerability maps at municipal scale, but they obscure the distinct causal pathways through which vulnerability arises. A road segment with an SLFVI score of 0.65 might have reached that score because its drainage is visibly absent, because its road surface is severely degraded, because it sits in a topographic depression, or because all three conditions co-occur. These distinctions matter for intervention: drainage installation, road resurfacing, and channel maintenance are different engineering responses directed at different actors in different budget categories.

Typological approaches address this limitation by grouping locations according to their pattern of indicator co-occurrence rather than their aggregate score. In sociology and urban planning, geodemographic classification has been used for decades to group census areas by their social profile (Singleton and Spielman, 2014). In ecology and sustainability science, the concept of archetypes has been used to identify recurrent configurations across case studies (Oberlack et al., 2019). In flood risk research, similar pattern-recognition approaches have been applied to damage curves, exposure profiles, and socioeconomic vulnerability components, but rarely to street-level infrastructure condition.

This paper asks whether unsupervised clustering of EfficientNet-B0 flood-indicator predictions at road-segment level produces an interpretable and policy-relevant typology of streetscape vulnerability in Greater Accra. The specific objectives are: (1) to determine the optimal number of clusters using silhouette and Calinski-Harabasz criteria; (2) to characterise each archetype by its mean label profile and flood vulnerability score; (3) to describe the spatial distribution of archetypes across the metropolitan area; and (4) to link each archetype to specific infrastructure intervention priorities.

The contribution is both methodological and empirical. Methodologically, the paper demonstrates that segment-level vision model outputs are sufficiently structured to support meaningful unsupervised classification. Empirically, it provides the first archetype-based typology of flood-vulnerable streetscapes for a major West African city.

---

## 2. Background

### 2.1 Urban Flood Vulnerability and Infrastructure Typology

Urban flood vulnerability in sub-Saharan African cities arises from the interaction of rapid urbanisation, inadequate formal drainage infrastructure, high-intensity seasonal rainfall, and low adaptive capacity in informal settlements (Douglas et al., 2008; Rentschler and Salhab, 2020). The spatial pattern of flood risk is therefore shaped not only by topography and rainfall but also by the visible condition of drainage infrastructure and road surfaces at street level.

Hallegatte et al. (2013) estimated that average annual losses from urban flooding in major coastal cities would reach USD 1 trillion by 2050 without adaptation, with the burden concentrated in cities undergoing rapid informal urbanisation. Winsemius et al. (2016) demonstrated that the key drivers of future flood exposure diverge between high-income and low-income countries: in low-income cities, socioeconomic exposure growth and informal urban expansion dominate, while in high-income cities, climate change effects are relatively more important. These findings motivate the development of fine-grained urban infrastructure typologies that can guide targeted adaptation investments in informal settings.

### 2.2 Cluster Analysis in Urban Analytics

K-means clustering is the most widely applied unsupervised partitioning algorithm in geographic data science, valued for its computational efficiency and interpretability (MacQueen, 1967). It has been used to derive neighbourhood typologies from census variables, urban form classifications from street morphology metrics, and land use categories from remote sensing spectral signatures. The algorithm minimises the within-cluster sum of squared distances:

$$\min_{\{C_k\}} \sum_{k=1}^{K} \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$$

where $C_k$ is the set of points assigned to cluster $k$ and $\boldsymbol{\mu}_k$ is the cluster centroid.

A well-known limitation of k-means in urban applications is sensitivity to the choice of $k$. Validation indices are used to guide selection. The silhouette coefficient (Rousseeuw, 1987) measures how similar each point is to its own cluster compared to the next-closest cluster:

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

where $a(i)$ is the mean intra-cluster distance and $b(i)$ is the mean distance to the nearest other cluster. Mean silhouette values range from −1 (misclassification) through 0 (indifferent) to 1 (perfect separation). Values between 0.25 and 0.50 indicate a reasonable structure (Rousseeuw, 1987). The Calinski-Harabasz index (Caliński and Harabasz, 1974) rewards compact clusters relative to their dispersion:

$$\text{CH}(k) = \frac{\text{SS}_B / (k-1)}{\text{SS}_W / (n-k)}$$

where $\text{SS}_B$ and $\text{SS}_W$ are the between-cluster and within-cluster sums of squares. Higher CH values indicate better-separated, more compact clusters.

### 2.3 Street View Imagery and Urban Classification

Street view imagery has been used to classify urban scenes across several dimensions. Li et al. (2015) derived a green view index from Google Street View to map urban greenery at street level. Gebru et al. (2017) used deep learning on street view to estimate neighbourhood income levels. Biljecki and Ito (2021) reviewed 641 studies that apply street view imagery in urban analytics and identified vulnerability and infrastructure assessment as an emerging application domain. No prior study has derived an unsupervised typology of flood-relevant streetscape conditions using vision model outputs in a West African city.

---

## 3. Study Area, Data, and Feature Preparation

### 3.1 Study Area

The study covers Greater Accra, Ghana, encompassing the Accra Metropolitan Area and surrounding municipalities including Ga East, Ga West, Ga South, La Dade-Kotopon, Ledzokuku, Krowor, Tema, Ashaiman, and Weija-Gbawe. A stratified sample of 5,000 road points was drawn from the OpenStreetMap road network across these districts, stratified by flood exposure and settlement type. The 3,836 road points used in this study are the subset for which Google Street View panoramas were available and EfficientNet-B0 inference was completed.

### 3.2 EfficientNet-B0 Flood-Indicator Predictions

The EfficientNet-B0 model (Tan and Le, 2019) was fine-tuned on a hybrid labelled dataset of 2,000 images (400 human-annotated plus 1,600 pseudo-labelled), producing probability estimates for 15 binary flood-indicator labels per image. The 15 labels are listed in Table 1. For each of the 3,836 covered road points, four directional images (headings 0, 90, 180, and 270 degrees) were inferred. A further 590 Mapillary images from gap-point locations were inferred separately. The predictions used in this study are restricted to the 15,344 Street View images for the 3,836 covered points; the Mapillary images were not included in the typology to preserve comparability across headings.

**Table 1. The 15 EfficientNet-B0 flood-indicator labels used in the typology.**

| Label | Short name | Type |
|---|---|---|
| Visible drain present | drain | Drainage presence |
| Open gutter present | open gutter | Drainage type |
| Blocked drain present | blocked drain | Drainage obstruction |
| Stagnant water visible | stagnant water | Surface water |
| Poor road condition | poor road | Road surface |
| Heavy impervious surface | impervious | Surface type |
| Unpaved shoulder | unpaved shoulder | Surface type |
| Informal structure near drainage | informal structure | Built form |
| Solid waste accumulation | solid waste | Management |
| Visible waterway or stream | waterway | Hydrology |
| Low-lying street form | low-lying | Topography proxy |
| Roadside erosion | erosion | Surface degradation |
| Pedestrian exposure | pedestrian | Exposure |
| Culvert or bridge visible | culvert/bridge | Drainage infrastructure |
| No visible drainage | no drainage | Drainage absence |

### 3.3 Segment-Level Aggregation

For each of the 3,836 road points, probabilities for each of the 15 labels were averaged across the four directional images to produce a segment-level mean probability vector $\mathbf{x}_i \in [0, 1]^{15}$. Additionally, the EfficientNet composite vulnerability probability ($p_{\text{vuln}}$) was averaged across headings to provide a scalar vulnerability score per segment.

### 3.4 Feature Standardisation

Before clustering, the 15-dimensional feature vectors were standardised to zero mean and unit variance using the z-score transformation:

$$z_{ij} = \frac{x_{ij} - \bar{x}_j}{\hat{\sigma}_j}$$

where $x_{ij}$ is the raw mean probability of label $j$ for segment $i$, $\bar{x}_j$ is the sample mean of label $j$, and $\hat{\sigma}_j$ is the sample standard deviation. Standardisation is necessary because labels with higher mean prevalence (for example, no-visible-drainage) would otherwise dominate the distance metric relative to rare labels (for example, stagnant water).

### 3.5 Dimensionality Assessment

Principal component analysis (PCA) was applied to the standardised 15-dimensional feature matrix to assess whether the label space was reducible. The scree analysis indicated that five principal components were sufficient to explain 95% of total variance.

---

## 4. Methods

### 4.1 K Selection

K-means clustering was run for $k$ from 2 to 8 using 20 random initialisations (n\_init = 20) and a maximum of 500 iterations per run. For each $k$, the mean silhouette coefficient and the Calinski-Harabasz index were computed. Both metrics were evaluated on the standardised feature matrix $\mathbf{Z}$.

### 4.2 Final Clustering

The final typology was derived by running k-means with $k = 6$, using 50 random initialisations (n\_init = 50) and a maximum of 1,000 iterations to ensure convergence stability. Clusters were indexed 1 through 6. For each cluster, the mean archetype profile was computed as:

$$\boldsymbol{\hat{\mu}}_k = \frac{1}{n_k} \sum_{i \in C_k} \mathbf{x}_i$$

where $\mathbf{x}_i$ are the raw (unstandardised) mean probability vectors. The mean composite vulnerability probability was also computed per cluster:

$$\bar{p}_k = \frac{1}{n_k} \sum_{i \in C_k} p_{\text{vuln},i}$$

Per-segment silhouette values were computed from scikit-learn's silhouette\_samples function to assess the internal cohesion of each cluster (Pedregosa et al., 2011).

### 4.3 Archetype Naming and Interpretation

Each cluster was assigned an archetype name based on inspection of the three to four most discriminating label probabilities relative to the grand mean across all segments. The grand mean for each label was computed to identify which labels were above or below average for a given archetype.

### 4.4 Spatial Analysis

Cluster assignments were joined to the road-segment point coordinates and visualised as a spatial layer. The frequency of each archetype was tabulated by municipality. Cross-tabulation with the SLFVI classification (very low to very high) was used to assess whether archetypes were associated with distinct SLFVI ranges.

---

## 5. Results

### 5.1 Cluster Number Selection

Table 2 presents silhouette and Calinski-Harabasz values for $k$ from 2 to 8; Figure 2 shows the same information as line plots. Both metrics peaked at $k = 2$, with silhouette 0.3081 and CH 1,897.4. The silhouette profile declined smoothly from $k = 2$ to $k = 8$ with no secondary peak, suggesting a continuous rather than a sharply partitioned feature space. The k = 2 solution separated the 3,836 road segments into a high-vulnerability cluster dominated by impervious surfaces and drainage absence (n = 1,147, mean vuln_prob 0.718) and a lower-vulnerability cluster with more distributed indicator values (n = 2,689, mean vuln_prob 0.261).

**Table 2. Cluster validation metrics for k = 2 to 8.**

| k | Silhouette | Calinski-Harabasz |
|---|---|---|
| 2 | 0.3081 | 1,897.4 |
| 3 | 0.2718 | 1,432.6 |
| 4 | 0.2613 | 1,312.1 |
| 5 | 0.2419 | 1,150.8 |
| 6 | 0.2301 | 1,074.3 |
| 7 | 0.2188 | 998.6 |
| 8 | 0.2055 | 938.4 |

A k = 6 solution was selected because it produces six archetypes with distinct and interpretable physical profiles that map directly onto different intervention categories. The silhouette of 0.2301 falls slightly below the 0.25 threshold that Rousseeuw (1987) associated with reasonable structure, though this is not uncommon for complex socio-physical feature spaces that rarely produce the compact spherical clusters that yield high silhouette values. The k = 6 result should be understood as an operational classification rather than a claim of discrete natural groupings in the feature space.

### 5.2 Six-Archetype Typology

Table 3 summarises the six archetypes by size, mean vulnerability probability, and the three most elevated label indicators. Figure 1 shows the workflow used to derive the typology. Figure 2 presents the silhouette and Calinski-Harabasz validation plots for k = 2 to 8. Figure 3 presents the full 15-label mean probability heatmap for each archetype. Figure 4 maps the spatial distribution of all six archetypes across the Greater Accra road network.

**Table 3. Six-archetype typology: size, vulnerability, and defining characteristics.**

| Archetype | n | % | Mean vuln_prob | Silhouette | Primary indicators |
|---|---|---|---|---|---|
| Type I: Dense Impervious Drainage Void | 741 | 19.3 | 0.734 | 0.236 | heavy_impervious 0.784; no_drainage 0.718; informal_structure 0.374 |
| Type II: Compound Drainage Failure | 406 | 10.6 | 0.485 | 0.224 | poor_road 0.692; unpaved_shoulder 0.601; solid_waste 0.593; erosion 0.586; low_lying 0.573 |
| Type III: Watercourse-Adjacent Mixed | 217 | 5.7 | 0.443 | 0.116 | waterway 0.214; culvert 0.211; drain_present 0.432; open_gutter 0.424; blocked 0.343 |
| Type IV: Open Gutter Waste Corridor | 655 | 17.1 | 0.422 | 0.136 | drain_present 0.533; solid_waste 0.525; pedestrian 0.503; unpaved_shoulder 0.527 |
| Type V: Residential Maintenance Deficit | 737 | 19.2 | 0.270 | 0.252 | poor_road 0.499; no_drainage 0.392; solid_waste 0.393; open_gutter 0.332 |
| Type VI: Low-Risk Formal Corridor | 1,080 | 28.2 | 0.150 | 0.288 | lowest across all labels; heavy_impervious 0.249; no_drainage 0.270 |

**Type I: Dense Impervious Drainage Void (n = 741; 19.3% of segments; mean vuln_prob = 0.734)**

This archetype is defined by the near-complete absence of visible drainage infrastructure combined with very high impervious surface coverage. The no-visible-drainage indicator reaches a mean of 0.718, the highest of any archetype for this label. Heavy impervious surface coverage averages 0.784. Road condition is relatively good (poor-road indicator 0.150), and unpaved shoulders are rare (0.102). The paradox of this archetype is that the road surface itself is well-paved, but the drainage infrastructure is invisible or absent. In Greater Accra, this pattern corresponds to recently sealed or resurfaced roads in high-density commercial or peri-commercial corridors where original open gutters have been covered, built over, or incorporated into private foundations without replacement drainage capacity. Stormwater that cannot access drainage fills the sealed road surface itself. The archetype's combination of high imperviousness and absent drainage is the most flood-generating configuration in the dataset.

**Type II: Compound Drainage Failure (n = 406; 10.6%; mean vuln_prob = 0.485)**

This archetype accumulates the most co-occurring failure modes of any cluster. Five indicators exceed 0.570: poor road condition (0.692), unpaved shoulder (0.601), solid waste accumulation (0.593), roadside erosion (0.586), and low-lying street form (0.573). Informal structure near drainage averages 0.517 and pedestrian exposure 0.561. Open gutters are present (0.468) but are filled with waste and sediment. The physical profile corresponds to unpaved or poorly maintained roads in low-lying informal settlements that carry waste, are actively eroding, and have restricted drainage channel capacity due to solid waste and encroachment. These segments represent the most visible and spatially concentrated flood risk areas in the dataset. They are often the locations recorded by NADMO in post-event flood damage reports.

**Type III: Watercourse-Adjacent Mixed Infrastructure (n = 217; 5.7%; mean vuln_prob = 0.443)**

This is the smallest archetype and the least internally cohesive (mean silhouette 0.116). It is distinguished by the highest values of any archetype for visible waterway or stream (0.214) and culvert or bridge visible (0.211). Drainage is present in multiple forms (drain 0.432, open gutter 0.424) but shows signs of stress (blocked drain 0.343, stagnant water 0.305). The physical profile corresponds to roads running alongside or crossing visible channels, streams, and drainage infrastructure, including culverted crossings. The mixed drainage health indicators suggest that waterway-adjacent infrastructure varies considerably between well-maintained bridge approaches and severely blocked culvert outlets. The low silhouette reflects the heterogeneous nature of roads that share proximity to waterways without sharing a single dominant failure mode.

**Type IV: Open Gutter Waste Corridor (n = 655; 17.1%; mean vuln_prob = 0.422)**

This archetype is defined by the co-occurrence of visible open gutters (0.472), drain presence (0.533), and solid waste accumulation (0.525) alongside high pedestrian exposure (0.503) and unpaved shoulders (0.527). Road condition is moderate (poor-road 0.310). The physical profile is the classic Accra informal market street or community access road: open gutters are present and functioning as intended drainage infrastructure, but they are partially obstructed by solid waste deposited by residents and vendors. This is operationally the most tractable archetype for intervention: the drainage infrastructure exists, but the management system is failing. Periodic drain desiltation and improved solid-waste collection are the primary intervention needs.

**Type V: Residential Maintenance Deficit (n = 737; 19.2%; mean vuln_prob = 0.270)**

This archetype represents residential roads in various stages of formal or semi-formal development where maintenance has fallen behind. Road condition is moderate-poor (0.499) and drainage is partially absent (no-drainage 0.392). Solid waste accumulation is moderate (0.393) and open gutters are visible but not dominant (0.332). The overall profile is one of mild and diffuse deterioration rather than acute failure. These segments are typically located in established residential areas that have formal road alignments but where maintenance budgets are insufficient to sustain drainage channel cleaning, road surface repair, and waste collection at adequate frequency. Preventive maintenance and community engagement are the primary policy responses.

**Type VI: Low-Risk Formal Corridor (n = 1,080; 28.2%; mean vuln_prob = 0.150)**

This archetype has the lowest mean vulnerability probability and the lowest indicator values across nearly all labels. It represents the formal road network: motorways, trunk roads, primary roads, and well-maintained secondary roads. Some degree of heavy impervious surface (0.249) and no-visible-drainage (0.270) are present, reflecting sealed road surfaces with subsurface drainage not visible at street level. The low pedestrian exposure (0.185) and low solid waste (0.202) values confirm that these roads have functioning maintenance regimes. This archetype serves as the reference category against which elevated vulnerability in other archetypes should be interpreted.

### 5.3 Spatial Distribution

Figure 4 maps all 3,836 road-segment archetype assignments across the metropolitan area. The six archetypes showed non-random spatial distributions. Type VI (Low-Risk Formal Corridor) was concentrated on primary and trunk roads emanating from the central business district and along the Accra-Tema motorway corridor. Type I (Dense Impervious Drainage Void) was distributed across high-density commercial sub-centres including Madina, Nungua, and Tema new town areas. Type II (Compound Drainage Failure) was concentrated in identified flood hotspots near seasonal watercourses and in low-lying settlement clusters in Ga East, Weija-Gbawe, and coastal Krowor. Type IV (Open Gutter Waste Corridor) was prevalent across informal residential areas throughout the central metropolitan zone. Type III (Watercourse-Adjacent Mixed) formed linear clusters along visible drainage channels and stream corridors.

### 5.4 Cross-Tabulation with SLFVI Classification

Figure 5 presents violin plots of the EfficientNet composite vulnerability probability for each archetype, ranked from highest to lowest mean. The distributions confirm that the archetypes occupy distinct and largely non-overlapping ranges of vulnerability probability. Cross-tabulation between archetype membership and SLFVI class confirmed that the archetypes are associated with distinct vulnerability ranges rather than spanning the full SLFVI distribution uniformly. Type I segments had 88% classified as SLFVI high or very high. Type II segments had 74% in SLFVI high or very high. Type VI segments had 92% in SLFVI very low or low. Types III, IV, and V straddled the moderate to high range, reflecting the intermediate and compound nature of their vulnerability profiles.

---

## 6. Discussion

### 6.1 Interpretability and Policy Utility of the Six-Archetype Solution

The choice of k = 6 over the statistically optimal k = 2 is a deliberate methodological decision with a policy rationale. The k = 2 solution identifies where flood vulnerability is high versus low, which is useful for prioritising attention but not for directing action. The k = 6 solution identifies six distinct causal configurations that map onto five different intervention categories: (1) drainage installation (Type I), (2) multi-faceted infrastructure rehabilitation (Type II), (3) culvert and waterway maintenance (Type III), (4) drain desiltation and waste management (Type IV), and (5) preventive maintenance (Type V). This aligns with the planning reality that flood risk management requires differentiated responses calibrated to the physical condition of each road type, not a single universal intervention.

This approach is consistent with the use of archetypes in social-ecological systems research (Oberlack et al., 2019) and with geodemographic classification methods in urban analytics (Singleton and Spielman, 2014). Both traditions argue that typologies are more useful for guiding differentiated interventions than continuous scores, precisely because they translate multi-dimensional variation into a small number of actionable categories.

### 6.2 The Dense Impervious Drainage Void as a Priority Finding

The most policy-relevant finding is that the highest-vulnerability archetype (Type I) combines excellent road surface quality with absent drainage infrastructure. This pattern is counterintuitive: the road looks maintained, but the drainage system is gone. It arises in Accra when roads are resurfaced or when informal commercial development encloses and seals existing open gutters without replacing them with subsurface drainage capacity. The result is a street that channels rather than drains stormwater: water cannot infiltrate the sealed surface, there is no visible gutter to carry it away, and local depressions collect it until it overflows.

This archetype is likely underidentified by conventional flood risk mapping methods that use DEM-derived topographic wetness index and distance to waterways, because the primary hazard driver is the absence of drainage on an otherwise non-low-lying road. It is only visible from street level. This observation reinforces the argument that street-level imagery detects flood hazard configurations that are invisible to satellite and terrain-based models alone.

### 6.3 Limitations

The typology is based on AI-model-predicted probabilities rather than ground-truth observations. EfficientNet-B0 errors propagate into cluster assignments: a label systematically over-predicted by the model will inflate the probability of archetypes where that label is a key discriminator. The mean inter-annotator Fleiss' κ of 0.51 for the annotation dataset indicates moderate human agreement on the underlying labels, implying that the ground-truth signal itself has noise. The archetype profiles should therefore be interpreted as characterising the modal visible condition of road segments rather than definitive ground-truth classifications.

The global silhouette of 0.2301 is below the 0.25 threshold that Rousseeuw (1987) associated with a reasonable structure. This reflects the continuous nature of the underlying feature space: flood vulnerability indicators co-vary gradually across the urban environment, and sharp boundaries between archetypes do not exist in the data. The k = 6 solution should be understood as a useful discretisation of a continuum rather than a discovery of naturally distinct types.

The typology was derived from covered road segments only (3,836 of 5,000 sampled points). Coverage gaps in Google Street View are systematically concentrated in service and residential road types, which are also the classes most associated with Types II, IV, and V. The distribution of archetypes in the full road network may therefore underestimate the prevalence of the moderate-to-high-vulnerability archetypes in gap areas.

### 6.4 Comparison with Existing Urban Flood Typologies

Prior flood typologies have been derived primarily from geospatial and socioeconomic data rather than from street-level visual evidence. Winsemius et al. (2016) classified countries by their flood risk drivers; Jongman et al. (2012) developed vulnerability curve typologies for European land use classes. Neither study identified infrastructure-condition archetypes at road-segment scale. To the best of the authors' knowledge, this paper provides the first archetype classification of flood-vulnerable streetscape conditions derived from computer vision model outputs in a sub-Saharan African city.

The Types IV and II archetypes are consistent with observations reported in the qualitative literature on Accra flooding. Douglas et al. (2008) described informal settlements in West African cities as characterised by open drainage channels accumulating solid waste, unpaved surfaces generating sediment, and low-lying topography limiting drainage gradients. The compound failure profile of Type II and the waste-obstructed gutter profile of Type IV operationalise these observations as quantitative cluster centroids for the first time.

---

## 7. Intervention Priorities by Archetype

The following intervention priorities are derived from the archetype profiles:

**Type I (Dense Impervious Drainage Void):** Install subsurface drainage or reinstate open gutter capacity before the next resealing cycle. Target engineering inspections at roads where resurfacing has coincided with gutter infilling. Fiscal priority: high. Lead agency: metropolitan assembly roads and drainage department.

**Type II (Compound Drainage Failure):** Multi-agency response required. Road resurfacing and unpaved shoulder stabilisation (public works), drainage channel clearance and waste collection upgrade (sanitation authority), and waterway corridor protection (urban water management authority). Fiscal priority: high. Target: known flood hotspot neighbourhoods in Ga East, Krowor, and Weija-Gbawe.

**Type III (Watercourse-Adjacent Mixed Infrastructure):** Culvert inspection and maintenance audit. Clear blockages at known culvert outlets and bridge abutments. Prevent informal development within drainage corridor buffers. Fiscal priority: medium-high. Lead: municipal drainage engineer with urban planning enforcement.

**Type IV (Open Gutter Waste Corridor):** Increase frequency of drain desiltation, particularly in market and commercial zones. Expand waste collection coverage to Type IV corridors. Community awareness and enforcement of waste disposal by-laws. Fiscal priority: medium. High impact per unit cost as infrastructure is present; management is failing.

**Type V (Residential Maintenance Deficit):** Preventive maintenance: road pothole repair, scheduled drain inspection, bi-annual waste collection drive. Community-level engagement through ward drainage committees. Fiscal priority: medium-low. Consistent maintenance prevents transition to Type II over time.

**Type VI (Low-Risk Formal Corridor):** Maintain current maintenance regimes. Periodic monitoring to detect transition toward Type I if road resealing covers existing drainage.

---

## 8. Conclusion

This paper has presented a six-archetype typology of streetscape flood vulnerability for Greater Accra, Ghana, derived from unsupervised k-means clustering of EfficientNet-B0 flood-indicator predictions at road-segment level. The six archetypes range in size from 5.7% (Type III: Watercourse-Adjacent Mixed) to 28.2% (Type VI: Low-Risk Formal Corridor) of the covered road network and range in mean composite vulnerability probability from 0.150 to 0.734. Each archetype represents a distinct combination of drainage infrastructure condition, road surface quality, solid waste management, and topographic context that maps onto a specific set of intervention priorities.

The most important substantive finding is the identification of the Dense Impervious Drainage Void archetype (Type I), which combines high impervious surface coverage with absent visible drainage on otherwise well-maintained roads. This archetype, comprising 19.3% of covered road segments, is flood-generating in a way that is invisible to conventional satellite and terrain-based mapping but clearly visible in street-level imagery. Its identification from AI model outputs demonstrates the practical value of street-level computer vision for urban flood management beyond the production of aggregate vulnerability scores.

The archetype framework transforms the SLFVI output from a ranking tool into an operational planning guide. By knowing not only how vulnerable a road segment is but also why it is vulnerable, municipal planners and drainage engineers can direct the right intervention to the right location with greater precision and cost-effectiveness than is possible from a composite score alone.

---

## Acknowledgements

[To be completed on submission.]

---

## References

Biljecki, F., and Ito, K. (2021). Street view imagery in urban analytics and GIS: A review. Landscape and Urban Planning, 215, 104217.

Caliński, T., and Harabasz, J. (1974). A dendrite method for cluster analysis. Communications in Statistics, 3(1), 1–27.

Douglas, I., Alam, K., Maghenda, M., Mcdonnell, Y., McLean, L., and Campbell, J. (2008). Unjust waters: climate change, flooding and the urban poor in Africa. Environment and Urbanization, 20(1), 187–205.

Gebru, T., Krause, J., Wang, Y., Chen, D., Deng, J., Aiden, E. L., and Fei-Fei, L. (2017). Using deep learning and Google Street View to estimate the demographic makeup of neighborhoods across the United States. Proceedings of the National Academy of Sciences, 114(50), 13108–13113.

Hallegatte, S., Green, C., Nicholls, R. J., and Corfee-Morlot, J. (2013). Future flood losses in major coastal cities. Nature Climate Change, 3(9), 802–806.

Jongman, B., Kreibich, H., Apel, H., Barredo, J. I., Bates, P. D., Feyen, L., Gericke, A., Neal, J., Aerts, J. C. J. H., and Ward, P. J. (2012). Comparative flood damage model assessment: Towards a European approach. Natural Hazards and Earth System Sciences, 12(12), 3733–3752.

Li, X., Zhang, C., Li, W., Ricard, R., Meng, Q., and Zhang, W. (2015). Assessing street-level urban greenery using Google Street View and a modified green view index. Urban Forestry and Urban Greening, 14(3), 675–685.

MacQueen, J. B. (1967). Some methods for classification and analysis of multivariate observations. Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability, 1, 281–297.

NADMO (2021). Ghana National Disaster Management Organisation Annual Flood Situation Report 2021. Government of Ghana, Accra.

Oberlack, C., Sietz, D., Bürgi Bonanomi, E., De Bremond, A., Dell'Angelo, J., Estel, S., Giger, M., Jacobi, J., Kasper, L. B., Keys, P., Lambinon, P., Meyfroidt, P., Müller, D., Prishchepov, A. V., Messina, J., and Verburg, P. H. (2019). Archetype analysis in sustainability research: Meanings, motivations, and evidence-based policy making. Ecology and Society, 24(2), 26.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.

Rentschler, J., and Salhab, M. (2020). People in Harm's Way: Flood Exposure and Poverty in 189 Countries. World Bank Policy Research Working Paper No. 9447. World Bank, Washington, DC.

Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20, 53–65.

Singleton, A., and Spielman, S. (2014). The past, present, and future of geodemographic research in the United States and United Kingdom. The Professional Geographer, 66(4), 558–567.

Tan, M., and Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In Proceedings of the International Conference on Machine Learning (ICML), 6105–6114.

Winsemius, H. C., Aerts, J. C. J. H., Van Beek, L. P. H., Bierkens, M. F. P., Bouwman, A., Jongman, B., Kwadijk, J. C. J., Ligtvoet, W., Lucas, P. L., Van Vuuren, D. P., and Ward, P. J. (2016). Global drivers of future river flood risk. Nature Climate Change, 6(4), 381–385.

---

---

## Figure Captions

![Figure 1](outputs/figures/fig1_paper3_workflow.png)

**Figure 1.** Methodological workflow for the streetscape typology derivation. Rounded boxes represent processing steps; rectangular boxes represent data or intermediate outputs; the purple terminal box represents final policy outputs. Arrows show data flow direction. Stage labels (S1–S6) correspond to the stages described in the Methods section.

![Figure 2](outputs/figures/typology_cluster_validation.png)

**Figure 2.** Cluster number selection for k = 2 to 8 using k-means on standardised 15-label EfficientNet feature vectors (n = 3,836 road segments). Panel (a) shows the mean silhouette coefficient; panel (b) shows the Calinski-Harabasz index. Both metrics peak at k = 2 (silhouette 0.3081; CH 1,897.4). The dashed red line marks k = 2. A k = 6 solution was chosen for its policy utility in distinguishing intervention categories.

![Figure 3](outputs/figures/typology_cluster_heatmap_k6.png)

**Figure 3.** Mean label probability heatmap for the six streetscape archetypes (rows) across 15 flood-indicator labels (columns). Cell values show mean probability across all segments in that archetype. Archetypes are ordered from highest (Type I, top) to lowest (Type VI, bottom) mean composite vulnerability probability. Red shading indicates high probability; blue shading indicates low probability.

![Figure 4](outputs/figures/fig4_paper3_spatial_archetypes.png)

**Figure 4.** Spatial distribution of six streetscape archetypes across the Greater Accra road-segment sample (n = 3,836 covered road points). Each point represents one road segment, coloured by archetype membership. Archetypes are plotted in vulnerability order (low-vulnerability types plotted first so high-vulnerability types appear on top in dense areas). Point size is 5 pixels with 65% opacity.

![Figure 5](outputs/figures/fig5_paper3_vuln_by_archetype.png)

**Figure 5.** Distribution of EfficientNet composite vulnerability probability by archetype, presented as violin plots with embedded miniature boxplot elements. Archetypes are ordered left to right from highest to lowest mean vulnerability probability. Within each violin, a narrow white rectangle marks the interquartile range (Q1–Q3); a short thick black horizontal bar indicates the median; and a wide dashed line in the archetype's colour indicates the mean, extending across approximately 60 percent of the violin width to distinguish it visually from the median. Whisker lines extend to the 1.5×IQR fence. The horizontal dotted line marks the 0.50 probability threshold. Sample size (n), mean (μ), and median (ḿ) values are reported in the x-axis tick labels for each archetype. The clear separation of violin bodies across the six archetypes confirms that the k = 6 typology captures meaningfully distinct vulnerability levels rather than arbitrary partitions of a uniform distribution.

---

*Code and data availability: All analysis code is openly available at https://github.com/dzimike/ghana-streetscape-flood (Zenodo DOI: to be assigned on publication). The streetscape typology clustering scripts and archetype label outputs are reproducible from the repository. Raw Street View imagery is not redistributed in accordance with Google Maps Platform terms of service; EfficientNet label probabilities used to derive the typology are included in the release dataset.*

*Manuscript prepared 2026-07-18. Target journals: Computers, Environment and Urban Systems; Landscape and Urban Planning; Urban Climate.*



