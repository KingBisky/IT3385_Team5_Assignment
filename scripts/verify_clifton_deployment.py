#!/usr/bin/env python3
"""Smoke-test Clifton Task 3 against a deployed Team 5 base URL.

Usage:
    python scripts/verify_clifton_deployment.py https://example.run.app
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid


def request(url, method="GET", data=None, headers=None, timeout=45):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, response.headers, response.read()


def multipart_csv(field_name, filename, content):
    boundary = "----Task3Boundary" + uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        "Content-Type: text/csv\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: verify_clifton_deployment.py <team-base-url>")
    base = sys.argv[1].rstrip("/")
    clifton = base + "/clifton"

    checks = []

    status, _, body = request(clifton + "/")
    checks.append(("Clifton page", status == 200 and b"Mental-health risk classification" in body))

    status, _, body = request(clifton + "/health")
    health = json.loads(body)
    checks.append(("Deep model health", status == 200 and health.get("model_loadable") is True))

    payload = {
        "panic_attack_history": 0,
        "family_history_mental_illness": 0,
        "anxiety_score": 5,
        "depression_score": 5,
        "work_stress_level": 5,
        "financial_stress_level": 5,
        "sleep_hours": 7.0,
        "social_support_score": 6,
        "physical_activity_hours_per_week": 6.0,
    }
    status, _, body = request(
        clifton + "/predict",
        method="POST",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    single = json.loads(body)
    checks.append(("Single real-time prediction", status == 200 and single.get("label") in {"Low", "Moderate", "High"}))

    status, _, template = request(clifton + "/batch/template")
    checks.append(("Batch template", status == 200 and b"anxiety_score" in template))

    batch_body, content_type = multipart_csv("file", "task3_smoke.csv", template)
    status, _, body = request(
        clifton + "/batch/predict",
        method="POST",
        data=batch_body,
        headers={"Content-Type": content_type},
        timeout=90,
    )
    batch = json.loads(body)
    checks.append(("Batch prediction", status == 200 and int(batch.get("row_count", 0)) >= 3))

    download_url = batch.get("download_url")
    if download_url:
        download = urllib.parse.urljoin(clifton + "/", download_url)
        status, _, result = request(download)
        checks.append(("Batch result download", status == 200 and b"predicted_risk_label" in result))
    else:
        checks.append(("Batch result download", False))

    print("Clifton Task 3 deployment verification")
    print("=" * 44)
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL':4}  {name}")
    if not all(passed for _, passed in checks):
        raise SystemExit(1)
    print("\nAll Task 3 deployment smoke checks passed.")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}", file=sys.stderr)
        raise SystemExit(1)
    except urllib.error.URLError as exc:
        print(f"Connection failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
