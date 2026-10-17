"""Atomic SPNv2 state and human-readable progress reporting (standard library only)."""
import copy
import datetime as dt
import json
import os
import tempfile
import time
from pathlib import Path

BEIJING = dt.timezone(dt.timedelta(hours=8))


def now():
    return dt.datetime.now(BEIJING).isoformat(timespec="seconds")


def atomic_write(path, text):
    """Same-directory replace, with data and directory durability."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError:
        return copy.deepcopy(default)


def age_seconds(timestamp):
    if not timestamp:
        return None
    try:
        parsed = dt.datetime.fromisoformat(timestamp)
        return max(0, int((dt.datetime.now(BEIJING) - parsed).total_seconds()))
    except (ValueError, TypeError):
        return None


def cell(value):
    if value is None:
        return "未提供"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return str(value).replace("|", "\\|").replace("\n", " ")


def path_line(label, value):
    return "- " + label + "：" + ("\u0060" + str(value) + "\u0060" if value else "未提供")


def render(state, plan, report_path):
    """Pure rendering: running is asserted only from recorded PID evidence."""
    current = state.get("current") or {}
    rows = [
        "# SPNv2 复现实时进度", "",
        "- 最后更新（北京时间）：" + cell(state.get("updated_at")),
        "- 队列管理器最后心跳（北京时间）：" + cell(state.get("heartbeat_at")),
        "- 管理器状态：" + cell(state.get("status")),
        "- 当前阶段：" + cell(state.get("active")),
        "- 自动刷新：每60秒；阶段、checkpoint、评价与异常事件立即刷新。",
        "- 状态代次：" + cell(state.get("revision", 0)),
        path_line("本轮目录", plan),
        path_line("状态文件", str(Path(plan) / "status.json")),
        path_line("本进度文件", report_path), "",
        "## 当前任务与实际进程", "",
    ]
    for label, key in [
        ("任务", "task_id"), ("阶段", "phase"), ("执行/验收阶段", "execution_phase"), ("进程核验", "process_state"),
        ("实际任务PID", "pid"), ("/proc启动标识", "starttime"), ("screen标识", "screen_id"),
        ("任务开始（北京时间）", "started_at"), ("最后任务事件（北京时间）", "last_event_at"),
        ("来源模型/配置", "source"), ("恢复位置", "resume"), ("最新checkpoint", "checkpoint"),
        ("epoch / 总epoch", "epoch_progress"), ("batch / 总batch", "batch_progress"),
        ("学习率", "lr"), ("loss", "loss"), ("更新尝试数", "update_attempts"),
        ("成功优化器更新数", "optimizer_updates"), ("AMP跳步数", "amp_skips"),
        ("已见样本数", "samples_seen"), ("最新事件", "event"), ("事件信息", "message"),
    ]:
        value = current.get(key)
        if key == "epoch_progress":
            value = str(current.get("epoch", "未提供")) + " / " + str(current.get("epochs", "未提供"))
        if key == "batch_progress":
            value = str(current.get("batch", "未提供")) + " / " + str(current.get("batches", "未提供"))
        rows.append("- " + label + "：" + cell(value))
    elapsed = age_seconds(current.get("started_at"))
    event_age = age_seconds(current.get("last_event_at"))
    rows.append("- 已用时间：" + (str(elapsed) + " 秒" if elapsed is not None else "未提供"))
    rows.append("- 距最后任务事件：" + (str(event_age) + " 秒" if event_age is not None else "尚无任务事件"))
    if event_age is not None and event_age >= 180:
        rows.append("- 事件新鲜度：超过180秒没有新任务事件；保留最后心跳，不推断训练正在推进。")
    for label, key in [("stdout/stderr日志", "log"), ("输出目录/文件", "outputs"),
                       ("实际命令", "command"), ("退出码", "exit_code")]:
        rows.append(path_line(label, current.get(key)))
    if state.get("resource_wait"):
        rows.append("- 资源等待原因：" + cell(state["resource_wait"]))
    if state.get("reporting_warning"):
        rows.extend(["", "## 进度写入告警（后续任务已锁止）", "", cell(state["reporting_warning"])])
    if state.get("error"):
        rows.extend(["", "## 异常", "", cell(state["error"])])
    rows.extend(["", "## 完整任务队列", "",
                 "| 任务 | 类型 | 状态 | 前置任务 | 结果/原因 |",
                 "|---|---|---|---|---|"])
    labels = {
        "pending": "queued（待执行）", "queued": "queued（待执行）",
        "launching": "启动中（尚未证明运行）", "running": "running（进程已核验）",
        "awaiting_verification": "awaiting verification（待验收）",
        "accepted": "accepted（已验收）", "failed": "failed（失败）",
        "limited": "limited（受限）", "waiting_resources": "queued（等待资源）",
    }
    task_details = []
    for task in state.get("tasks", []):
        status = task.get("status", "queued")
        rows.append("| " + " | ".join(cell(x) for x in [
            task.get("id"), task.get("kind"), labels.get(status, status),
            ", ".join(task.get("requires", [])) or "无",
            task.get("error") or task.get("reason") or task.get("accepted_at") or "",
        ]) + " |")
        runtime = task.get("runtime") or {}
        if runtime:
            task_details.append("- " + cell(task["id"]) + "：" + cell({
                "session": runtime.get("session"), "log": runtime.get("log"),
                "record": runtime.get("record"), "result": runtime.get("result"),
                "outputs": task.get("expected_outputs", []),
                "resume": (task.get("metadata") or {}).get("resume"),
            }))
    if state.get("backlog"):
        rows.extend(["", "## 后续复现任务（待执行；不代表已进入当前自动队列）", ""])
        rows.extend("- " + cell(task) for task in state["backlog"])
    evaluation = state.get("last_evaluation")
    if evaluation:
        summary = evaluation.get("summary") or evaluation
        rows.extend(["", "## 最近评价", "", "- 域/阶段：" + cell(evaluation.get("domain", evaluation.get("phase"))),
                     "- 样本数：" + cell(summary.get("n_samples", summary.get("validation_samples"))),
                     "- 热图拒绝数：" + cell(summary.get("heatmap_rejections")),
                     "", "| 指标 | 均值（无有效值记缺失） | 单位 | 有效分母 |", "|---|---|---|---|"])
        for name, metric in (summary.get("metrics") or {}).items():
            rows.append("| " + " | ".join(cell(v) for v in (name, metric.get("mean"), metric.get("unit"), metric.get("count"))) + " |")
    rows.extend(["", "## 任务证据路径", ""] + task_details)
    rows.extend(["", "## 说明与事实边界", "",
                 "screen存在与退出码0均不单独代表任务运行或验收通过。运行状态依据记录PID及/proc启动标识；验收依据验证命令与实际产物。",
                 "如自动心跳停止，上面的最后心跳保持原值；请结合实际进程和日志核验。失败后保留现场并停止后续任务；不会自动重试、换参或换种子。"])
    for note in state.get("notes", []):
        rows.append("- " + cell(note))
    return "\n".join(rows) + "\n"


class Reporter:
    """Single caller owns state writes; a failed publication blocks advancement."""
    def __init__(self, state_path, report_path, state, interval=60, warning_path=None):
        self.state_path = Path(state_path)
        self.report_path = Path(report_path)
        self.state = state
        self.interval = interval
        self.last_success = 0
        self.write_failed = False
        self.warning_path = Path(warning_path or self.state_path.with_suffix(".warnings.log"))

    def warn(self, error):
        message = now() + " REPORT_WRITE_WARNING " + repr(error) + "\n"
        try:
            self.warning_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.warning_path, "a", encoding="utf-8") as stream:
                stream.write(message)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            pass
        print(message, end="", flush=True)

    def publish(self, force=False):
        if not force and time.monotonic() - self.last_success < self.interval:
            return not self.write_failed
        snapshot = copy.deepcopy(self.state)
        snapshot["updated_at"] = now()
        snapshot["heartbeat_at"] = snapshot["updated_at"]
        snapshot["revision"] = int(snapshot.get("revision", 0)) + 1
        try:
            # Both files carry the same revision; a failed pair never authorizes advance.
            atomic_write(self.report_path, render(snapshot, self.state_path.parent, self.report_path))
            atomic_json(self.state_path, snapshot)
        except (OSError, TypeError, ValueError) as error:
            self.write_failed = True
            self.warn(error)
            return False
        for key in ("updated_at", "heartbeat_at", "revision"):
            self.state[key] = snapshot[key]
        self.write_failed = False
        self.last_success = time.monotonic()
        return True
