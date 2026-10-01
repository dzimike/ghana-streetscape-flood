"""Set up Label Studio project for human review of 400 flood vulnerability images.

Creates project, sets label config, enables local file serving, imports tasks.

Usage
-----
  python src/annotation/setup_label_studio.py --token <your_token>
"""
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import requests
from loguru import logger

LS_URL      = "http://localhost:8080"
TASKS_JSON  = ROOT / "data/interim/human_review_tasks.json"
IMAGE_ROOT  = str(ROOT / "data/raw/images")

LABEL_CONFIG = """<View>
  <View style="display:grid;grid-template-columns:1.4fr 1fr;gap:0 1.5em">
    <View>
      <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>
      <View style="background:#f0f4f8;padding:8px 12px;margin-top:8px;border-radius:6px;font-size:13px">
        <Text name="meta" value="AI label: $ai_class | Confidence: $ai_confidence | Road: $highway | Stratum: $flood_stratum"/>
      </View>
    </View>
    <View>
      <Header value="Step 1 — Overall Vulnerability Class"/>
      <Choices name="vulnerability_class" toName="image" choice="single" required="true"
               showInline="false">
        <Choice value="low_flood_vulnerability"        style="background:#d4edda;padding:4px 8px"/>
        <Choice value="moderate_flood_vulnerability"   style="background:#fff3cd;padding:4px 8px"/>
        <Choice value="high_flood_vulnerability"       style="background:#f8d7da;padding:4px 8px"/>
        <Choice value="uncertain_requires_field_check" style="background:#e2e3e5;padding:4px 8px"/>
      </Choices>

      <Header value="Step 2 — Flood Indicators (check all clearly visible)"/>
      <Choices name="image_labels" toName="image" choice="multiple" showInline="false">
        <Choice value="visible_drain_present"/>
        <Choice value="open_gutter_present"/>
        <Choice value="blocked_drain_present"/>
        <Choice value="stagnant_water_visible"/>
        <Choice value="poor_road_condition"/>
        <Choice value="heavy_impervious_surface"/>
        <Choice value="unpaved_shoulder"/>
        <Choice value="informal_structure_near_drainage"/>
        <Choice value="solid_waste_accumulation"/>
        <Choice value="visible_waterway_or_stream"/>
        <Choice value="low_lying_street_form"/>
        <Choice value="roadside_erosion"/>
        <Choice value="pedestrian_exposure"/>
        <Choice value="culvert_or_bridge_visible"/>
        <Choice value="no_visible_drainage"/>
      </Choices>

      <Header value="Step 3 — Notes (optional)"/>
      <TextArea name="notes" toName="image" placeholder="Any observations about this image..."
                rows="2" editable="true"/>
    </View>
  </View>
</View>"""


def get_access_token(refresh_token: str) -> str:
    """Exchange a JWT refresh token for a short-lived access token."""
    resp = requests.post(
        f"{LS_URL}/api/token/refresh/",
        json={"refresh": refresh_token},
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"Could not exchange refresh token ({resp.status_code}). "
            "Get a fresh token from Label Studio → Account & Settings → Access Token."
        )
    return resp.json()["access"]


def get_headers(token: str) -> dict:
    # token may be a refresh JWT — exchange it for an access token first
    if token.startswith("eyJ") and token.count(".") == 2:
        import base64, json as _json
        payload = token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        decoded = _json.loads(base64.urlsafe_b64decode(payload))
        if decoded.get("token_type") == "refresh":
            token = get_access_token(token)
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def create_project(token: str) -> int:
    headers = get_headers(token)
    payload = {
        "title": "Accra Flood Vulnerability — Human Review",
        "description": (
            "Review 400 street-level images from Accra, Ghana. "
            "AI pre-labels (GPT-4o-mini) are shown — confirm or correct each image. "
            "Focus on visible flood infrastructure conditions."
        ),
        "label_config":            LABEL_CONFIG,
        "show_instruction":        True,
        "show_skip_button":        True,
        "enable_empty_annotation": False,
        "show_annotation_history": True,
        "maximum_annotations":     2,
        "instruction": (
            "<b>Instructions for annotators:</b><br>"
            "1. Look carefully at the street-level image from Accra, Ghana.<br>"
            "2. The AI pre-label is shown above the image — correct it if needed.<br>"
            "3. Select the overall vulnerability class (Step 1).<br>"
            "4. Check all flood indicators clearly visible in the image (Step 2).<br>"
            "5. Only label what you can see — do not guess from location or context.<br>"
            "6. Add notes if the image is unclear or unusual (Step 3).<br><br>"
            "<b>Key definitions:</b><br>"
            "- <b>Open gutter</b>: roadside concrete or earthen channel without a cover<br>"
            "- <b>Blocked drain</b>: drain filled with waste, silt, or debris<br>"
            "- <b>Stagnant water</b>: standing water on road or in drain<br>"
            "- <b>Informal structure near drainage</b>: kiosk or stall beside a drain"
        ),
    }
    resp = requests.post(f"{LS_URL}/api/projects/", headers=headers, json=payload)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Failed to create project: {resp.status_code} {resp.text[:300]}")
    project_id = resp.json()["id"]
    logger.info(f"Project created — ID: {project_id}")
    return project_id


def enable_local_files(token: str, project_id: int) -> None:
    headers = get_headers(token)
    payload = {
        "project":       project_id,
        "title":         "Local images",
        "storage_type":  "localfiles",
        "path":          IMAGE_ROOT,
        "regex_filter":  ".*\\.jpg",
        "use_blob_urls": True,
    }
    resp = requests.post(f"{LS_URL}/api/storages/localfiles/",
                         headers=headers, json=payload)
    if resp.status_code in (200, 201):
        logger.info(f"Local file storage linked → {IMAGE_ROOT}")
    else:
        logger.warning(
            f"Local storage not added ({resp.status_code}) — "
            "images will still load if LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true"
        )


def import_tasks(token: str, project_id: int) -> int:
    headers = get_headers(token)
    tasks = json.loads(TASKS_JSON.read_text())
    total = 0
    for i in range(0, len(tasks), 100):
        batch = tasks[i:i + 100]
        resp = requests.post(
            f"{LS_URL}/api/projects/{project_id}/import",
            headers=headers,
            json=batch,
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Import failed at batch {i}: {resp.status_code} {resp.text[:300]}")
        total += len(batch)
        logger.info(f"  Imported {total}/{len(tasks)} tasks …")
    return total


def main(token: str) -> None:
    resp = requests.get(f"{LS_URL}/health")
    if resp.status_code != 200:
        raise RuntimeError(f"Label Studio not reachable at {LS_URL}")
    logger.info(f"Label Studio running at {LS_URL}")

    resp = requests.get(f"{LS_URL}/api/current-user/whoami",
                        headers=get_headers(token))
    if resp.status_code != 200:
        raise RuntimeError(
            f"Token rejected ({resp.status_code}). "
            "Go to Label Studio → Account & Settings → Access Token to get your token."
        )
    user = resp.json()
    logger.info(f"Authenticated as: {user.get('email', user.get('username', 'unknown'))}")

    project_id = create_project(token)
    enable_local_files(token, project_id)
    n = import_tasks(token, project_id)

    print("\n── Label Studio Project Ready ───────────────────────────────")
    print(f"  Project ID:       {project_id}")
    print(f"  Tasks imported:   {n}")
    print(f"  Open in browser:  {LS_URL}/projects/{project_id}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--token", required=True, help="Label Studio user access token")
    args = p.parse_args()
    main(args.token)
