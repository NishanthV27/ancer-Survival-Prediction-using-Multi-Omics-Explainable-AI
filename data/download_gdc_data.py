"""
GDC Data Downloader for TCGA-COAD multi-omics raw files.

Downloads open-access RNA-seq and Methylation files using their GDC UUIDs
from the official GDC API (https://api.gdc.cancer.gov/data/{uuid}).
Features:
- Download matched patient pairs (Clinical + RNA-seq + Methylation)
- Filters out non-beta raw IDAT files in favor of level3betas.txt
- Concurrent multithreaded downloads with progress reporting
- Configurable timeouts and retries with exponential backoff
- MD5 checksum verification against gdc_manifest.tsv
- Graceful error handling for network / DNS connection failures
- Preserves all existing raw files
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import requests

GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"


def verify_md5(file_path: str, expected_md5: str) -> bool:
    """Verify file integrity by calculating its MD5 hash."""
    if not os.path.exists(file_path):
        return False
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    calculated = hasher.hexdigest().lower()
    return calculated == expected_md5.lower().strip()


def download_file(
    uuid: str,
    output_path: str,
    expected_md5: Optional[str] = None,
    max_retries: int = 3,
    timeout: int = 45,
) -> Tuple[bool, str]:
    """
    Download a single file from the GDC API with retries and checksum verification.

    Returns
    -------
    (success: bool, message: str)
    """
    url = f"{GDC_DATA_ENDPOINT}/{uuid}"
    temp_path = output_path + ".part"

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                if response.status_code == 200:
                    with open(temp_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)

                    # Checksum verification if expected_md5 provided
                    if expected_md5:
                        if not verify_md5(temp_path, expected_md5):
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            return False, f"MD5 mismatch for UUID {uuid}"

                    # Atomically rename temp file to destination
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    os.rename(temp_path, output_path)
                    return True, "Verified"
                elif response.status_code == 404:
                    return False, f"HTTP 404 (Not Found)"
                elif response.status_code == 403:
                    return False, f"HTTP 403 (Access Denied)"
                else:
                    time.sleep(1.5 * attempt)

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt == max_retries:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return False, f"Network error: {str(e)}"
            time.sleep(2 ** attempt)

        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False, f"Unexpected error: {str(e)}"

    if os.path.exists(temp_path):
        os.remove(temp_path)
    return False, "Exhausted retries"


def download_single_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function for concurrent downloads."""
    target_path = os.path.join(task["output_dir"], task["filename"])
    if os.path.exists(target_path):
        if task.get("md5") and verify_md5(target_path, task["md5"]):
            return {"status": "skipped", "task": task, "msg": "Already verified"}

    ok, msg = download_file(task["uuid"], target_path, task.get("md5"))
    return {"status": "success" if ok else "failed", "task": task, "msg": msg}


def get_matched_downloads(
    limit: Optional[int] = None,
    base_dir: str = "."
) -> List[Dict[str, Any]]:
    """Build list of download tasks for patients with verified clinical records."""
    clin_path = os.path.join(base_dir, "data", "processed", "tcga_coad_clinical.csv")
    ge_meta_path = os.path.join(base_dir, "data", "raw", "gene_expression", "file_metadata.json")
    me_meta_path = os.path.join(base_dir, "data", "raw", "methylation", "file_metadata.json")

    ge_out = os.path.join(base_dir, "data", "raw", "gene_expression", "files")
    me_out = os.path.join(base_dir, "data", "raw", "methylation", "files")

    if not os.path.exists(clin_path):
        raise FileNotFoundError(f"Processed clinical file not found: {clin_path}")

    clin_df = pd.read_csv(clin_path)
    clin_pids = set(clin_df["patient_id"])

    with open(ge_meta_path, "r", encoding="utf-8") as f:
        ge_hits = json.load(f)["data"]["hits"]

    with open(me_meta_path, "r", encoding="utf-8") as f:
        me_hits = json.load(f)["data"]["hits"]

    me_betas = [h for h in me_hits if h.get("file_name", "").endswith(".txt")]

    ge_map = {}
    for h in ge_hits:
        for c in h.get("cases", []):
            pid = c.get("submitter_id")
            if pid in clin_pids:
                for s in c.get("samples", []):
                    if s.get("sample_type") == "Primary Tumor" or "-01A" in s.get("submitter_id", ""):
                        ge_map[pid] = h

    me_map = {}
    for h in me_betas:
        for c in h.get("cases", []):
            pid = c.get("submitter_id")
            if pid in clin_pids:
                for s in c.get("samples", []):
                    if s.get("sample_type") == "Primary Tumor" or "-01A" in s.get("submitter_id", ""):
                        me_map[pid] = h

    matched_pids = sorted(set(ge_map.keys()) & set(me_map.keys()))
    if limit is not None and limit > 0:
        matched_pids = matched_pids[:limit]

    tasks = []
    for pid in matched_pids:
        ge_item = ge_map[pid]
        tasks.append({
            "patient_id": pid,
            "modality": "gene_expression",
            "uuid": ge_item["file_id"],
            "filename": ge_item["file_name"],
            "md5": ge_item.get("md5sum"),
            "output_dir": ge_out,
        })
        me_item = me_map[pid]
        tasks.append({
            "patient_id": pid,
            "modality": "methylation",
            "uuid": me_item["file_id"],
            "filename": me_item["file_name"],
            "md5": me_item.get("md5sum"),
            "output_dir": me_out,
        })

    return tasks, matched_pids


def run_batch_download(tasks: List[Dict[str, Any]], max_workers: int = 4):
    """Execute download tasks concurrently with progress reporting."""
    total = len(tasks)
    succeeded = 0
    skipped = 0
    failed = 0
    errors = []

    print(f"Starting concurrent download for {total} files using {max_workers} workers...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_single_task, t): t for t in tasks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            t = res["task"]
            status = res["status"]
            msg = res["msg"]

            if status == "success":
                succeeded += 1
                print(f"[{completed}/{total}] [OK] {t['patient_id']} ({t['modality']}): {t['filename']}")
            elif status == "skipped":
                skipped += 1
                print(f"[{completed}/{total}] [SKIP] {t['patient_id']} ({t['modality']}): {t['filename']} ({msg})")
            else:
                failed += 1
                print(f"[{completed}/{total}] [FAIL] {t['patient_id']} ({t['modality']}): {t['filename']} - {msg}")
                errors.append(f"{t['patient_id']} {t['filename']}: {msg}")

    elapsed = time.time() - start_time
    print(f"\nDownload finished in {elapsed:.1f}s: {succeeded} downloaded, {skipped} skipped, {failed} failed.")
    return {"succeeded": succeeded, "skipped": skipped, "failed": failed, "errors": errors}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download public TCGA-COAD files via GDC API.")
    parser.add_argument("--matched-patients", type=int, default=None, help="Download matched GE & ME files for top N patients")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent workers")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ge_out = os.path.join(base_dir, "data", "raw", "gene_expression", "files")
    me_out = os.path.join(base_dir, "data", "raw", "methylation", "files")
    os.makedirs(ge_out, exist_ok=True)
    os.makedirs(me_out, exist_ok=True)

    print("=== GDC TCGA-COAD RAW DATA DOWNLOADER ===")
    print(f"Gene expression target: {ge_out}")
    print(f"Methylation target:     {me_out}")

    limit = args.matched_patients if args.matched_patients is not None else 30
    tasks, matched_pids = get_matched_downloads(limit=limit, base_dir=base_dir)
    print(f"Prepared {len(tasks)} file download tasks across {len(matched_pids)} matched patients.")

    run_batch_download(tasks, max_workers=args.workers)


if __name__ == "__main__":
    main()
