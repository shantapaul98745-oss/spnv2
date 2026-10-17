"""Read-only compact progress display; never signals or starts experiments."""
import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys
import time

BASE = pathlib.Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai')
PLAN = BASE / 'spnv2_runs/plans/rebaseline_20261017'
ZONE = dt.timezone(dt.timedelta(hours=8))

def load(path):
    return json.loads(path.read_text(encoding='utf-8'))

def live(pid, expected):
    try:
        fields = pathlib.Path('/proc', str(int(pid)), 'stat').read_text().rsplit(')', 1)[1].split()
        return str(fields[19]) == str(expected) and fields[0] not in ('Z', 'X')
    except (OSError, ValueError, TypeError, IndexError):
        return False

def last_event(path):
    if not path:
        return {}
    try:
        with pathlib.Path(path).open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 65536))
            lines = stream.read().decode('utf-8', 'replace').splitlines()
        for line in reversed(lines):
            if line.startswith('SPNV2_EVENT '):
                try:
                    return json.loads(line[len('SPNV2_EVENT '):])
                except ValueError:
                    continue
    except OSError:
        pass
    return {}


def event_time(value):
    """Training uses Unix seconds; ODR may emit an ISO timestamp."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            parsed = dt.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
            return parsed.replace(tzinfo=ZONE).timestamp() if parsed.tzinfo is None else parsed.timestamp()
        except (TypeError, ValueError):
            return None


def duration(seconds):
    if seconds is None:
        return '未提供'
    seconds = max(0, int(seconds))
    return f'{seconds // 3600}小时{seconds % 3600 // 60:02d}分{seconds % 60:02d}秒'

def number(value, spec='.4f'):
    if value is None:
        return '未提供'
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)

def render():
    now = dt.datetime.now(ZONE)
    state = load(PLAN / 'status.json')
    current = dict(state.get('current') or {})
    event = last_event(current.get('log'))
    if (event_time(event.get('timestamp')) or 0) >= (event_time(current.get('timestamp')) or 0):
        current.update(event)
    tasks = state.get('tasks') or []
    task = next((item for item in tasks if item['id'] == state.get('active')), {})
    metadata = task.get('metadata') or {}
    source = pathlib.Path(str(current.get('source') or metadata.get('source') or '未提供')).stem
    model = {'phi3_bn_ehs': 'φ3-BN / E+H+S 完整模型',
             'phi6_gn_ehs': 'φ6-GN / E+H+S 完整模型'}.get(source, source)
    statuses = {'running': '运行中', 'waiting_resources': '等待显卡资源',
                'failed': '失败，后续任务已停止', 'phase_complete': '本阶段完成，完整计划仍有待办',
                'complete': '当前队列完成', 'awaiting_verification': '待验收',
                'queued': '等待执行', 'launching': '启动中'}
    status = statuses.get(state.get('status'), str(state.get('status')))
    alive = live(current.get('pid'), current.get('starttime'))
    manager = state.get('manager') or {}
    manager_alive = live(manager.get('pid'), manager.get('starttime'))
    stamp = event_time(current.get('timestamp'))
    age = max(0, int(time.time() - stamp)) if stamp is not None else None
    phase = {'train': '训练', 'validation': '评价', 'evaluate': '评价',
             'evaluation': '评价', 'odr': 'ODR适应'}.get(current.get('phase'), current.get('phase', '未提供'))
    rows = ['SPNv2 复现进度（简洁版）', '更新时间：' + now.strftime('%Y-%m-%d %H:%M:%S 北京时间'),
            '', f'状态：{status}  |  阶段：{phase}', f'模型：{model}']
    if metadata.get('physical_batch'):
        rows.append(f"训练方式：每批{metadata['physical_batch']}张 × 累积{metadata.get('accum_steps', 1)}次后更新")
    epoch, epochs = current.get('epoch'), current.get('epochs')
    batch, batches = current.get('batch'), current.get('batches')
    if epoch is not None and epochs:
        rows.append(f'训练轮次：第 {epoch} / {epochs} 轮')
    if batch is not None and batches:
        rows.append(f'本轮进度：{batch:,} / {batches:,} 批（{100 * batch / batches:.1f}%）')
    rows.append(f"最新损失：{number(current.get('loss'))}  |  学习率：{number(current.get('lr'), '.6g')}")
    rows.append(f"成功更新：{current.get('optimizer_updates', '未提供')} 次  |  AMP累计跳步：{current.get('amp_skips', '未提供')} 次")
    started = current.get('started_at')
    try:
        elapsed = (now - dt.datetime.fromisoformat(started)).total_seconds()
    except (ValueError, TypeError):
        elapsed = None
    rows.append('本任务已用：' + duration(elapsed))
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                              '--format=csv,noheader,nounits', '--id=0'],
                             capture_output=True, text=True, timeout=3, check=True)
        util, used, total = [int(x.strip()) for x in gpu.stdout.strip().split(',')]
        rows.append(f'显卡整卡：利用率 {util}%  |  显存 {used / 1024:.1f} / {total / 1024:.1f} GiB')
    except (OSError, ValueError, subprocess.SubprocessError):
        rows.append('显卡状态：本次查询失败，不能据此判断空闲')
    rows.append('任务进程：' + ('存活（已核对启动标识）' if alive else '未核实存活，请结合阶段状态判断'))
    rows.append('队列管理器：' + ('存活（已核对启动标识）' if manager_alive else '未核实存活'))
    rows.append('训练/评价日志：' + (f'{age}秒前更新' if age is not None else '尚无事件时间'))
    warnings = []
    for key in ('error', 'reporting_warning', 'resource_wait'):
        if state.get(key):
            warnings.append(str(state[key]))
    if age is not None and age >= 180 and task.get('status') == 'running':
        warnings.append('超过180秒没有新进度事件；不能仅凭进程存在认定正在推进')
    if state.get('status') == 'running' and not manager_alive:
        warnings.append('队列管理器未核实存活，请检查管理器记录')
    rows.append('异常提示：' + ('；'.join(warnings) if warnings else '当前状态文件未报告异常'))
    rows.extend(['', '说明：loss是训练损失；AMP跳步是混合精度保护计数，不等于任务失败。',
                 '简洁页每5秒刷新；Ctrl+C仅关闭此查看器。完整记录保留在原日志。',
                 '详细进度：' + str(BASE / 'paper/SPNv2_复现实时进度.md')])
    return '\n'.join(rows) + '\n'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    follow = not args.once and sys.stdout.isatty()
    try:
        while True:
            try:
                output = render()
            except (OSError, ValueError, KeyError, TypeError) as error:
                output = '进度暂时读取失败（没有操作实验）：' + str(error) + '\n'
            if follow:
                print('\033[2J\033[H', end='')
            print(output, end='', flush=True)
            if not follow:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print('\n已关闭进度查看器；实验不受影响。')

if __name__ == '__main__':
    main()
