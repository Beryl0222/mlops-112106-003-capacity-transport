"""按时间序列执行交换资料，输出各版本分配与反向结果。

时间序列条目表示调用方送达或发起读取的顺序：
  ["arrive", 块]     送达一个候选块（资料里没有该块时只登记声明 → 缺片）
  ["pin", 版本]      把当前已到达的块钉成版本并求解（本轮实际分配）
  ["replace", 版本]  用同名更正替换一份逻辑块，产生新候选版并求解
  ["reverse", 版本]  沿明确绑定的旧版分配反传（dy 与 d_objective 取自资料）
"""

import argparse
import json
from pathlib import Path

import transport_op as tp


def _version_json(report: tp.SolveReport, event: str) -> dict:
    out = {"revision": report.revision, "事件": event, "状态": report.status.value}
    if report.message:
        out["消息"] = report.message
    if report.ok:
        out.update({
            "目标": report.objective,
            "下界": report.lower_bound,
            "间隙": report.gap,
            "行误差": report.row_error,
            "列误差": report.column_error,
            "迭代": report.iterations,
            "弃置": report.discard,
            "列状态": [
                {"列": c.column, "载荷": c.load, "容量": c.capacity,
                 "乘子": c.multiplier, "状态": c.state}
                for c in report.columns
            ],
            "分配": [[r, c, float(v)]
                     for r, c, v in zip(report.edge_rows, report.edge_cols, report.values)],
        })
    else:
        for key, val in (("行误差", report.row_error), ("列误差", report.column_error),
                         ("间隙", report.gap), ("迭代", report.iterations)):
            if val is not None:
                out[key] = val
    out["工作区字节"] = report.workspace_bytes
    return out


def _reverse_json(report: tp.BackwardReport) -> dict:
    out = {"revision": report.revision, "状态": report.status.value}
    if report.message:
        out["消息"] = report.message
    if report.critical_columns:
        out["临界列"] = report.critical_columns
    if report.ok:
        out["梯度"] = [[r, c, float(v)]
                       for r, c, v in zip(report.edge_rows, report.edge_cols, report.gradient)]
    if report.columns:
        out["列状态"] = [
            {"列": c.column, "载荷": c.load, "容量": c.capacity,
             "乘子": c.multiplier, "状态": c.state}
            for c in report.columns
        ]
    out["工作区字节"] = report.workspace_bytes
    return out


def run_scene(data: dict) -> dict:
    session = tp.ExchangeSession(
        epsilon=data["epsilon"],
        tolerance=data["tolerance"],
        budget_bytes=data["budget_bytes"],
        row_mass=data["rows"],
        capacity=data["capacity"],
        discard_rows=data.get("discard_rows", []),
    )
    blocks = data.get("blocks", {})
    replacements = {r["revision"]: r for r in data.get("replacements", [])}
    dy = data.get("dy", [])
    d_objective = data.get("d_objective", 0.0)

    versions, reverses = [], []
    for event in data.get("timeline", []):
        kind, name = event[0], event[1]
        if kind == "arrive":
            if name in blocks:
                session.arrive_block(name, blocks[name])
            else:
                session.declare_block(name)  # 已声明完成却未收到 → 缺片
        elif kind == "pin":
            versions.append(_version_json(session.pin(name), "pin"))
        elif kind == "replace":
            repl = replacements.get(name)
            if repl is None:
                versions.append({"revision": name, "事件": "replace",
                                 "状态": tp.Status.INVALID_INPUT.value,
                                 "消息": "未找到对应更正"})
            else:
                versions.append(_version_json(
                    session.replace(name, repl["replace"], repl["edges"]), "replace"))
        elif kind == "reverse":
            reverses.append(_reverse_json(session.reverse(name, dy, d_objective)))
        else:
            raise ValueError(f"未知时间序列事件: {kind}")

    current = session.current_revision
    for entry in versions:
        # 旧版软标签不能继续算作当前版已确认结果
        entry["当前"] = entry["revision"] == current
    return {"当前版本": current, "版本": versions, "反向": reverses}


def main() -> None:
    parser = argparse.ArgumentParser(description="执行交换资料的时间序列并输出分配与反向结果")
    parser.add_argument("scene", type=Path)
    args = parser.parse_args()
    data = json.loads(args.scene.read_text(encoding="utf-8"))
    report = run_scene(data)
    report["资料"] = args.scene.name
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
