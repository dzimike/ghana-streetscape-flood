# Annotation Codebook — Accra Flood Vulnerability Streetscape Dataset

## Overview

Each Street View image is annotated at three levels:
1. **Image-level labels** — multi-label checkboxes for visible flood indicators
2. **Object-level bounding boxes** — for detecting specific drainage and waste features
3. **Segment vulnerability class** — overall flood-vulnerability rating for the scene

Annotators should use the Street View heading metadata to understand viewing direction.
Focus on what is visible in *this specific image*, not what you know about the area.

---

## Image-Level Labels (multi-label — check all that apply)

| Label | When to check |
|---|---|
| `visible_drain_present` | Any drain, gutter or channel is clearly visible in the scene. |
| `open_gutter_present` | An open-top roadside gutter is visible (may be concrete, earth, or stone-lined). |
| `blocked_drain_present` | A drain or gutter appears blocked, silted, or filled with debris/waste. |
| `stagnant_water_visible` | Standing or ponded water is visible on the road surface or in a drain. |
| `poor_road_condition` | Potholes, cracking, rutting, erosion, or surface failure is visible on the road. |
| `heavy_impervious_surface` | The scene is dominated by concrete, asphalt, or compacted surfaces with little bare soil or vegetation. |
| `unpaved_shoulder` | The road shoulder or verge is unpaved (earth, gravel, or bare soil). |
| `informal_structure_near_drainage` | Kiosks, containers, sheds, or market structures are sited directly adjacent to a drain or waterway. |
| `solid_waste_accumulation` | Visible heap or scatter of solid waste (bags, plastics, refuse) in the scene. |
| `visible_waterway_or_stream` | A natural or semi-natural stream, river, or channel is visible. |
| `low_lying_street_form` | The road appears to sit in a depression, bowl, or low-lying corridor relative to surrounding land. |
| `roadside_erosion` | Erosion gullies, scour marks, or undercutting are visible at the road edge or verge. |
| `pedestrian_exposure` | Pedestrians are present near flood-risk features (edge of drain, flooded road, etc.). |
| `culvert_or_bridge_visible` | A culvert pipe or bridge structure crossing a drain or waterway is visible. |
| `no_visible_drainage` | No drain, gutter, or waterway infrastructure is visible in the scene. |

---

## Segment Vulnerability Class (single choice)

Choose the **one best class** that describes the overall flood vulnerability visible in the scene:

| Class | Description |
|---|---|
| `low_flood_vulnerability` | No drainage problems, road in good condition, no waste, no water |
| `moderate_flood_vulnerability` | Minor issues visible — partial blockage, some waste, slight surface damage |
| `high_flood_vulnerability` | Clear problems — blocked drain, standing water, significant waste, poor road |
| `uncertain_requires_field_check` | Image is unclear, obstructed, or the situation requires physical verification |

---

## Object Bounding Box Categories

Draw tight bounding boxes around visible objects. One box per object instance.

| Category | What to box |
|---|---|
| `drain` | Engineered drainage channel — concrete or masonry lined. |
| `gutter` | Roadside open gutter, any material. |
| `culvert` | Pipe or box culvert where road crosses a channel. |
| `water` | Ponded, flowing, or stagnant water body. |
| `solid_waste` | Heap or scatter of solid waste or refuse. |
| `road` | Paved or unpaved road surface. |
| `sidewalk` | Pedestrian walkway. |
| `vegetation` | Trees, grass, shrubs, or other vegetation. |
| `building` | Any building structure. |
| `kiosk_container` | Roadside kiosk, shipping container, or informal market stall. |
| `bridge` | Bridge structure over a waterway. |
| `stream_channel` | Natural or semi-natural stream, river, or open channel. |

---

## Annotation Protocol

1. Open the task in Label Studio.
2. View the image at full resolution.
3. Check all applicable **image-level labels**.
4. Draw **bounding boxes** around all visible drain, water, waste, culvert, and stream objects.
5. Select the **segment vulnerability class**.
6. Add a note in the comments field for any unusual or ambiguous scene.
7. Mark as complete.

### Quality rules

- If fewer than 20% of the image shows meaningful content (e.g. camera pointing at sky or wall), mark `uncertain_requires_field_check`.
- Do not guess — if a feature is not clearly visible, do not label it.
- For `blocked_drain_present`, the obstruction must be clearly visible — not just inferred from context.
- At least 15% of images will be double-coded for inter-annotator agreement checking.

---

## Ghana-Specific Notes

- **Open gutters** are common along Accra roads — these are the rectangular concrete channels alongside roads.
- **Drains filled with plastic bags and organic waste** are a major flood driver — always label `blocked_drain_present` AND `solid_waste_accumulation`.
- **Informal kiosks built over drains** should trigger `informal_structure_near_drainage`.
- **Earth shoulders** on paved roads should trigger `unpaved_shoulder`.
- **Low road geometry** — if the road surface appears to be below the surrounding land level, mark `low_lying_street_form`.

---
*Codebook version 1.0 — Streetscape Flood Vulnerability Project, Accra, Ghana*