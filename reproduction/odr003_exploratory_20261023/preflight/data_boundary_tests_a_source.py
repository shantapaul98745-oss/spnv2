"""Frozen image-only streams and developer-only labels for ODR003.

No model imports or target-error computation. Preparation streams CSV rows and
parses pose values only after their ID has entered the frozen developer split.
Runtime ImageManifestDataset never opens any source label file.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys

DOMAINS = {"lightbox": 6740, "sunlamp": 2791}
ITEM_KEYS = {"sample_id", "domain", "filename", "image_sha256"}
MANIFEST_KEYS = {"schema", "purpose", "image_root", "image_folder", "contains_gt", "items", "seed"}
IMAGE_FOLDER = "images_768x512_RGB"
SPLIT_SEED = 20261022
STREAM_SEEDS = (2021, 2022, 2023)
HASH_DISTANCE = 4
THUMB_RMSE_MAX = 0.03


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
    payload = (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as f:
        f.write(payload)
    return {"path": str(path), "sha256": digest(path), "bytes": len(payload)}


def safe_filename(name):
    if not isinstance(name, str) or re.fullmatch(r"img[0-9]{6}\.jpg", name) is None:
        raise ValueError(f"Unexpected image filename: {name!r}")
    return name


def read_ids_only(csv_path):
    names = []
    with Path(csv_path).open(encoding="utf-8") as f:
        for line in f:
            names.append(safe_filename(line.partition(",")[0].strip()))
    if len(names) != len(set(names)):
        raise ValueError("Duplicate sample IDs")
    return names


def read_developer_labels(csv_path, domain, selected_ids):
    """Only parse numeric columns of selected developer IDs; skip reserve rows."""
    import numpy as np
    selected = set(selected_ids)
    records = {}
    with Path(csv_path).open(encoding="utf-8") as f:
        for line in f:
            filename, _, remainder = line.partition(",")
            filename = safe_filename(filename.strip())
            sample_id = domain + "/" + filename
            if sample_id not in selected:
                continue
            values = np.asarray([float(x) for x in remainder.strip().split(",")], dtype=np.float32)
            if values.shape != (35,) or not np.isfinite(values).all():
                raise ValueError(f"Bad developer label schema or non-finite value for {sample_id}")
            bbox = values[0:4].clip(0, 1)
            records[sample_id] = {"sample_id": sample_id, "domain": domain,
                "filename": filename, "boundingbox_normalized": bbox.tolist(),
                "quaternion_wxyz": values[4:8].tolist(), "translation_m": values[8:11].tolist(),
                "origin_normalized": values[11:13].tolist(),
                "keypoints_normalized": values[13:].reshape(11, 2).tolist()}
    if set(records) != selected:
        raise ValueError("Developer label IDs do not match the frozen selection")
    return [records[x] for x in selected_ids]


def fingerprint(image):
    """64-bit low-frequency pHash plus brightness-sensitive thumbnail gate."""
    import cv2
    import numpy as np
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    low = cv2.dct(thumb)[:8, :8].reshape(-1)
    bits = low > np.median(low[1:])
    bits[0] = False
    code = sum(int(v) << i for i, v in enumerate(bits))
    return code, thumb


class HashTree:
    def __init__(self):
        self.nodes = []

    def add(self, value, index):
        if not self.nodes:
            self.nodes.append([value, [index], {}])
            return
        cursor = 0
        while True:
            node = self.nodes[cursor]
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = len(self.nodes)
                self.nodes.append([value, [index], {}])
                return
            cursor = node[2][distance]

    def query(self, value, radius):
        found, todo = [], [0] if self.nodes else []
        while todo:
            node = self.nodes[todo.pop()]
            d = (value ^ node[0]).bit_count()
            if d <= radius:
                found.extend(node[1])
            todo.extend(child for edge, child in node[2].items() if d-radius <= edge <= d+radius)
        return found


def group_images(items, fingerprints, thumbnails):
    import numpy as np
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    tree, exact, edges = HashTree(), {}, []
    for i, (item, code, thumb) in enumerate(zip(items, fingerprints, thumbnails)):
        sha = item["image_sha256"]
        if sha in exact:
            j = exact[sha]
            union(i, j)
            edges.append({"a": items[j]["sample_id"], "b": item["sample_id"], "kind": "exact_bytes", "hamming": 0, "thumbnail_rmse": 0.0})
        else:
            exact[sha] = i
        for j in tree.query(code, HASH_DISTANCE):
            rmse = float(np.sqrt(np.mean((thumb-thumbnails[j])**2)))
            if rmse <= THUMB_RMSE_MAX:
                union(i, j)
                edges.append({"a": items[j]["sample_id"], "b": item["sample_id"], "kind": "perceptual_candidate", "hamming": (code ^ fingerprints[j]).bit_count(), "thumbnail_rmse": rmse})
        tree.add(code, i)
    groups = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(items[i]["sample_id"])
    groups = sorted([sorted(g) for g in groups.values()], key=lambda g: g[0])
    return groups, edges


def choose_developer(groups, per_domain=512, seed=SPLIT_SEED):
    """Deterministic randomized group order; never cut a group to fill a quota."""
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    remaining = dict.fromkeys(DOMAINS, per_domain)
    selected, selected_groups = set(), []
    for i in order:
        counts = Counter(x.split("/", 1)[0] for x in groups[i])
        if all(counts.get(d, 0) <= remaining[d] for d in DOMAINS):
            selected.update(groups[i])
            selected_groups.append(i)
            for domain, n in counts.items():
                remaining[domain] -= n
    if any(remaining.values()):
        raise ValueError(f"Whole groups cannot fill requested developer boundary: remaining={remaining}; scientific review required")
    return selected, selected_groups


def make_manifest(root, purpose, items, seed=None):
    return {"schema": "odr003.images.v1", "purpose": purpose, "image_root": str(Path(root).resolve()),
            "image_folder": IMAGE_FOLDER, "contains_gt": False, "seed": seed, "items": items}


def load_image_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(manifest) != MANIFEST_KEYS or manifest.get("schema") != "odr003.images.v1" or manifest.get("contains_gt") is not False:
        raise ValueError("Not a strict image-only manifest")
    if manifest["image_folder"] != IMAGE_FOLDER:
        raise ValueError("Image folder changed from original A")
    ids = []
    for item in manifest["items"]:
        if set(item) != ITEM_KEYS or item["domain"] not in DOMAINS:
            raise ValueError("Unexpected item keys/domain: annotations are forbidden")
        safe_filename(item["filename"])
        if item["sample_id"] != item["domain"] + "/" + item["filename"]:
            raise ValueError("Sample ID mismatch")
        if re.fullmatch(r"[0-9a-f]{64}", item["image_sha256"]) is None:
            raise ValueError("Invalid image hash")
        ids.append(item["sample_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated IDs are forbidden in these manifests")
    return manifest


class ImageManifestDataset:
    """Image-only map dataset. Pass frozen original-A test transforms explicitly.

    With no transform returns raw uint8 BGR exactly as cv2.IMREAD_COLOR.
    With original build_transforms(cfg,is_train=False,load_labels=False), returns
    the original ImageNet-normalized CHW tensor. No implicit resizing or RGB swap.
    Default per-read SHA verification rejects mutated image bytes.
    """
    def __init__(self, manifest_path, transforms=None, verify_image_sha=True):
        self.manifest = load_image_manifest(manifest_path)
        self.transforms = transforms
        self.verify_image_sha = bool(verify_image_sha)

    def __len__(self):
        return len(self.manifest["items"])

    def __getitem__(self, index):
        import cv2
        import numpy as np
        item = self.manifest["items"][index]
        path = Path(self.manifest["image_root"])/item["domain"]/IMAGE_FOLDER/item["filename"]
        payload = path.read_bytes()
        if self.verify_image_sha and hashlib.sha256(payload).hexdigest() != item["image_sha256"]:
            raise ValueError(f"Image content changed: {item['sample_id']}")
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (512, 768, 3):
            raise ValueError(f"Unexpected original-A image shape: {item['sample_id']}")
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        return {"image": image, "sample_id": item["sample_id"], "domain": item["domain"], "filename": item["filename"]}


def prepare(root, output, proof_path):
    import cv2
    import numpy as np
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Data directory already exists; reuse only after independent verification")
    output.mkdir(parents=True)
    items, fingerprints, thumbnails, sources = [], [], [], {}
    for domain, count in DOMAINS.items():
        csv_path = root/domain/"labels/test.csv"
        names = read_ids_only(csv_path)
        if len(names) != count:
            raise ValueError(f"Wrong source pool size: {domain}")
        sources[domain] = {"csv_path": str(csv_path), "sha256": digest(csv_path), "rows": count,
            "metadata_keys": ["filename"], "acquisition_group_available": False}
        for name in names:
            path = root/domain/IMAGE_FOLDER/name
            payload = path.read_bytes()
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape != (512, 768, 3):
                raise ValueError(f"Bad source image: {path}")
            items.append({"sample_id": domain+"/"+name, "domain": domain, "filename": name,
                          "image_sha256": hashlib.sha256(payload).hexdigest()})
            code, thumb = fingerprint(image)
            fingerprints.append(code); thumbnails.append(thumb)
        print(json.dumps({"event": "image_pool_fingerprinted", "domain": domain, "count": count}), flush=True)
    groups, edges = group_images(items, fingerprints, thumbnails)
    grouping = {"schema": "odr003.groups.v1", "rule": {"phash": "cv2 DCT of 32x32 grayscale; top8x8, median nonDC; DC bit zero", "hamming_max": HASH_DISTANCE,
        "thumbnail_rmse_max": THUMB_RMSE_MAX, "thumbnail_range": [0, 1], "scope": "both_domains_together",
        "exact_bytes_always_grouped": True, "transitive_connected_components": True},
        "limitations": ["No acquisition-group metadata in source CSV/JSON; numeric filenames are not inferred acquisition groups.",
                       "Perceptual candidates are a conservative deterministic leakage screen, not proof of independent acquisitions or absence of all near duplicates."],
        "groups": groups, "edges": edges}
    artifacts = {"groups": save_json(output/"groups.json", grouping)}
    selected, group_indices = choose_developer(groups)
    split = {"schema": "odr003.split.v1", "seed": SPLIT_SEED,
        "selection_rule": "Python random.Random(seed).shuffle(sorted group indices), greedily accept whole groups if per-domain quota not exceeded; fail unless exactly512 each",
        "developer_group_indices": group_indices, "groups_sha256": artifacts["groups"]["sha256"],
        "developer_ids": sorted(selected), "reserve_ids": sorted(x["sample_id"] for x in items if x["sample_id"] not in selected),
        "contains_gt": False, "reserve_images_allowed_in_unlabeled_adaptation": True}
    artifacts["split"] = save_json(output/"split.json", split)
    counts = {}
    for domain, count in DOMAINS.items():
        pool = [x for x in items if x["domain"] == domain]
        dev = [x for x in pool if x["sample_id"] in selected]
        reserve = [x for x in pool if x["sample_id"] not in selected]
        for purpose, rows in [("pool", pool), ("developer", dev), ("reserve", reserve)]:
            key = domain+"/"+purpose
            artifacts[key] = save_json(output/domain/(purpose+"_images.json"), make_manifest(root, purpose, rows, SPLIT_SEED if purpose != "pool" else None))
        labels = read_developer_labels(root/domain/"labels/test.csv", domain, [x["sample_id"] for x in dev])
        artifacts[domain+"/developer_labels"] = save_json(output/domain/"dev_labels.json", {"schema": "odr003.developer_labels.v1", "domain": domain,
            "purpose": "developer_evaluation_only_never_adaptation_or_gating", "image_manifest_sha256": artifacts[domain+"/developer"]["sha256"], "items": labels})
        stream_counts = {}
        for seed in STREAM_SEEDS:
            positions = list(range(count))
            random.Random(seed).shuffle(positions)
            rows = [pool[i] for i in positions[:1024]]
            key = domain+"/stream_"+str(seed)
            artifacts[key] = save_json(output/domain/("stream_"+str(seed)+".json"), make_manifest(root, "paired_adaptation_stream", rows, seed))
            stream_counts[str(seed)] = {"observed": len(rows), "unique": len({x['sample_id'] for x in rows}),
                "developer_images": sum(x['sample_id'] in selected for x in rows),
                "reserve_images_unlabeled": sum(x['sample_id'] not in selected for x in rows)}
        counts[domain] = {"pool": len(pool), "developer": len(dev), "reserve": len(reserve), "streams": stream_counts}
    proof = {"status": "prepared", "created_at": datetime.now(timezone.utc).isoformat(), "script_sha256": digest(__file__),
        "python": sys.version, "opencv": cv2.__version__, "numpy": np.__version__, "sources": sources,
        "artifacts": artifacts, "counts": counts, "group_count": len(groups), "non_singleton_groups": sum(len(g)>1 for g in groups),
        "maximum_group_size": max(map(len, groups)), "perceptual_edge_count": len(edges),
        "cross_domain_groups": sum(len({x.split('/')[0] for x in g})>1 for g in groups),
        "developer_gt_rows_parsed": 1024, "reserve_numeric_gt_rows_parsed": 0,
        "model_loaded": False, "scores_computed": False, "next_gate": "independent_verify_required"}
    save_json(proof_path, proof)
    print(json.dumps({k: proof[k] for k in ['status','counts','group_count','non_singleton_groups','maximum_group_size','cross_domain_groups']}, sort_keys=True), flush=True)


def verify(output, proof_path, result_path):
    """Check frozen documents and group/stream/GT boundaries without source GT."""
    output = Path(output)
    prepared = json.loads(Path(proof_path).read_text())
    checks = []
    for name, asset in prepared["artifacts"].items():
        if digest(asset["path"]) != asset["sha256"]:
            raise ValueError(f"Artifact hash changed: {name}")
    checks.append("all_prepared_artifact_hashes")
    split = json.loads((output/"split.json").read_text())
    groups = json.loads((output/"groups.json").read_text())["groups"]
    dev, reserve = set(split["developer_ids"]), set(split["reserve_ids"])
    if dev & reserve or len(dev) != 1024 or len(reserve) != 8507:
        raise ValueError("Split size or overlap")
    for group in groups:
        if set(group) & dev and set(group) & reserve:
            raise ValueError("Near-duplicate group crosses developer/reserve")
    repeated, _ = choose_developer(groups)
    if repeated != dev:
        raise ValueError("Split is not reproducible")
    checks.extend(["whole_groups_no_cross_split", "seed_selection_reproducible"])
    for domain, n in DOMAINS.items():
        pool = load_image_manifest(output/domain/"pool_images.json")["items"]
        d = load_image_manifest(output/domain/"developer_images.json")["items"]
        r = load_image_manifest(output/domain/"reserve_images.json")["items"]
        if len(pool)!=n or len(d)!=512 or len(r)!=n-512:
            raise ValueError("Wrong domain denominators")
        if {x['sample_id'] for x in d} != {x for x in dev if x.startswith(domain+'/')}:
            raise ValueError("Developer image manifest mismatch")
        labels = json.loads((output/domain/"dev_labels.json").read_text())
        if [x['sample_id'] for x in labels['items']] != [x['sample_id'] for x in d]:
            raise ValueError("Developer labels leak or omit IDs")
        if labels['image_manifest_sha256'] != digest(output/domain/'developer_images.json'):
            raise ValueError("Developer image/label binding")
        for seed in STREAM_SEEDS:
            rows = load_image_manifest(output/domain/("stream_"+str(seed)+".json"))["items"]
            positions = list(range(n)); random.Random(seed).shuffle(positions)
            if rows != [pool[i] for i in positions[:1024]]:
                raise ValueError("Paired stream changed or not reproducible")
            if len(rows)!=1024 or len({x['sample_id'] for x in rows})!=1024:
                raise ValueError("Incorrect unique observations")
        checks.append(domain+"_GT_and_paired_stream_boundaries")
    result = {"status": "passed", "created_at": datetime.now(timezone.utc).isoformat(), "prepared_proof_sha256": digest(proof_path),
        "verifier_sha256": digest(__file__), "checks": checks, "counts": prepared['counts'],
        "full_source_labels_opened": False, "reserve_numeric_gt_rows_parsed": 0, "model_loaded": False, "scores_computed": False,
        "limitations": ["Grouping screens deterministic image similarity; no supplied acquisition groups and no guarantee of acquisition independence."]}
    save_json(result_path, result)
    print(json.dumps(result, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--root", required=True); p.add_argument("--output", required=True); p.add_argument("--proof", required=True)
    p = sub.add_parser("verify"); p.add_argument("--output", required=True); p.add_argument("--proof", required=True); p.add_argument("--result", required=True)
    args = parser.parse_args()
    if args.action == "prepare": prepare(args.root, args.output, args.proof)
    else: verify(args.output, args.proof, args.result)


if __name__ == "__main__":
    main()
