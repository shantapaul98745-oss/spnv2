"""Deterministic compact archive and fully reversible readable task overview."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import time

DEST = Path(__file__).resolve().parent
PLAN = DEST.parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compressed(raw):
    target = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=target, mode="wb", compresslevel=9, mtime=0) as stream:
        stream.write(raw)
    return target.getvalue()


def main():
    raw_path = DEST / "full_main_v3_proposal.json"
    raw = raw_path.read_bytes()
    proposal = json.loads(raw)
    target = DEST / "proposal.json.gz"
    overview = DEST / "task_overview.md"
    proof = DEST / "archive_verification.json"
    if any(path.exists() for path in (target, overview, proof)):
        raise FileExistsError("Do not overwrite frozen archive artifacts")
    data = compressed(raw)
    assert gzip.decompress(data) == raw and compressed(raw) == data
    assert data[4:8] == bytes(4) and not (data[3] & 8), "mtime/FNAME must not introduce nondeterminism"
    target.write_bytes(data)
    fixed = [t["id"] for t in proposal["tasks"] if t.get("metadata", {}).get("odr_group") == "fixed"]
    groups = {
        "S3": ["eval_phi3_synthetic", "eval_phi3_lightbox", "eval_phi3_sunlamp"],
        "S6": ["eval_phi6_synthetic", "eval_phi6_lightbox", "eval_phi6_sunlamp"],
        "F40": fixed,
    }
    lines = [
        "# 完整主流程 v3b 任务索引（静态提案，未激活）", "",
        "共444项；10 source配置190轮、30域评价、4 pilot、400 ODR请求（370独立执行、30别名）。",
        "顺序：现行9项 → 固定40 ODR → Table2/3/4的35项source任务 → B扫描120 → N扫描240。",
        "资源/GPU测量、最终分析和材料受限核验仍在backlog，本表不代表整体目标完成。", "",
        "前置栏使用可完全展开的组名：",
        "- S3：" + "、".join(groups["S3"]),
        "- S6：" + "、".join(groups["S6"]),
        "- F40：下方列出的全部40项固定ODR验收。",
        "每行前置均已按组名展开后与真实requires集合逐项核对；不会省略串行依赖。", "",
        "| 序号 | 任务ID | 前置（组名可展开） | 用途 | 是否alias |",
        "|---:|---|---|---|---|",
    ]
    aliases = 0
    for number, task in enumerate(proposal["tasks"], 1):
        remaining = list(task.get("requires", []))
        tokens = []
        for group in ("S3", "S6", "F40"):
            values = groups[group]
            if set(values) <= set(remaining):
                tokens.append(group)
                remaining = [value for value in remaining if value not in values]
        tokens = remaining + tokens
        expanded = []
        for value in tokens:
            expanded += groups[value] if value in groups else [value]
        assert set(expanded) == set(task.get("requires", []))
        metadata = task.get("metadata", {})
        reuse = metadata.get("reuse_of")
        if task["kind"].startswith("odr_"):
            purpose = (metadata["odr_group"] + " φ" + str(metadata["phi"]) + "/" + metadata["line"]
                       + " " + metadata["domain"] + " s" + str(metadata["observation_seed"])
                       + " N" + str(metadata["N"]) + "/B" + str(metadata["B"]))
        elif task["kind"].startswith("source_training_"):
            purpose = "独立source训练 " + str(metadata.get("epochs")) + "轮"
        elif task["kind"] == "full_source_evaluation":
            purpose = "source全域评价 " + str(metadata.get("domain", Path(metadata["source"]).stem.split("_")[-1]))
        else:
            purpose = "工程pilot：32 attempts / checkpoint16；独立CPU验收"
        alias = "是 → " + reuse if reuse else "否"
        aliases += bool(reuse)
        lines.append("| " + str(number) + " | " + task["id"] + " | "
                     + ("；".join(tokens) or "无") + " | " + purpose + " | " + alias + " |")
    assert len(proposal["tasks"]) == 444 and aliases == 30 and len(fixed) == 40
    lines += ["", "## F40完整展开", ""]
    lines += [str(i) + ". " + task for i, task in enumerate(fixed, 1)]
    overview.write_text("\n".join(lines) + "\n", encoding="utf-8")
    index_path = DEST / "proposal_index.json"
    index = json.loads(index_path.read_text())
    archive = {
        "gzip": str(target), "gzip_sha256": sha(target), "gzip_size_bytes": target.stat().st_size,
        "raw": str(raw_path), "raw_sha256": sha(raw_path), "raw_size_bytes": len(raw),
        "mtime": 0, "filename_header": "", "compression_level": 9,
        "decompressed_raw_sha256": hashlib.sha256(gzip.decompress(target.read_bytes())).hexdigest(),
        "decompressed_bytes_equal_raw": True, "repeated_compression_bytes_equal": True,
        "overview": str(overview), "overview_sha256": sha(overview),
        "overview_rows": 444, "overview_alias_rows": aliases, "overview_dependencies_fully_reversible": True,
    }
    assert archive["decompressed_raw_sha256"] == archive["raw_sha256"] == index["proposal_file_sha256"]
    index.update(
        summary_zh="静态未激活提案：444任务；10配置190轮、30域评价、4 pilot、400ODR请求=370独立+30别名；现9→固定40→source35→扫描360；资源与最终分析仍待验收",
        runner_code_git_commit="aa8e26fd717135f86b37b1dc2ce69febf144acf8",
        runner_file_sha256=sha(PLAN / "pipeline/runner.py"), archive=archive,
    )
    index["artifact_hashes"].update({"proposal.json.gz": sha(target), "task_overview.md": sha(overview)})
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    proof.write_text(json.dumps({"status": "passed", "scope": "deterministic archive/readable index only",
                                "archive": archive, "package_code_sha256": sha(__file__)}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"archive_status": "passed", "gzip_sha256": archive["gzip_sha256"],
                      "raw_sha256": archive["raw_sha256"], "gzip_bytes": len(data),
                      "overview_rows": 444, "alias_rows": aliases}))


if __name__ == "__main__":
    main()
