"""可重用的熵正则容量传输算子（前向软分配 + 反向代价梯度）。

约定（见 exchange.md）：
  目标  min Σ c_ij p_ij + ε p_ij (log p_ij − 1)，0 log 0 = 0；
  约束  行质量全部分出，正常列质量和不超过容量，弃置列无容量且仅 discard_rows 可用；
  精度  行误差、列误差（超出容量部分）、可行目标与对偶下界之差，均以 tolerance 为准；
  状态  ok / invalid_input / missing_blocks / infeasible / resource_exceeded /
        not_converged / critical_capacity（仅反向：容量临界，导数不可保证）。

前向是对偶的 Dykstra 块坐标上升：p = u·exp(−c/ε)·w，行缩放 u 使行和精确等于
质量，列缩放 w = min(1, cap/s) 使列和不超过容量；λ = −ε log w ≥ 0，对偶下界
g = −εΣp + εΣ m·log u + εΣ cap·log w 在每次迭代都有效（弱对偶）。

反向对活跃集隐式微分：dy（对有效边质量的导数）与 d_objective（对标量最优目标
的导数）经接地 Laplacian L = [[diag(m), −P_A], [−P_Aᵀ, diag(cap_A)]] 的一次
求解回到参与求解的原始代价。全活跃连通分量有一个平移零空间，钉住其中一个行
节点即可，梯度对该平移不变。松弛与乘子在容差内同时不可区分的列是临界的：
存在临界列时不交付梯度（critical_capacity），而不是悄悄忽略容量影响。

临时空间核算（budget_bytes）：只计跨块暂存——节点数组（对偶 u/w、边际 t/s、
反向 ν 与共轭梯度工作区）与按块流出的边缓冲；不可变块输入（边表与代价）和
最终输出（分配、梯度）不计入。结构可行性预检（Dinic 最大流）仅在迭代停滞或
达到迭代限时触发，单独核算并与求解阶段取峰值。
"""

from __future__ import annotations

import enum
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

DISCARD = "discard"

# 列约束状态
SLACK = "slack"        # 松弛：乘子为零、容量未用满
ACTIVE = "active"      # 受限：乘子为正、容量用满
CRITICAL = "critical"  # 临界：容差内无法区分正乘子与零松弛

# 临时空间核算常数
_EDGE_CHUNK_FLOATS = 3        # 每个边块最多 3 个 float64 暂存（k、p、加权量）
_FORWARD_NODE_FLOATS = 2      # 前向每侧节点数组数（u/t、w/s）
_BACKWARD_NODE_FLOATS = 8     # 反向节点数组数（ν、h、diag、var、r、z、p、Ap）
_FLOW_BYTES_PER_NODE = 64     # 结构预检每节点估算（含 Python 对象）
_FLOW_BYTES_PER_ARC = 64      # 结构预检每条弧估算
_STAGNATION_WINDOW = 256      # 停滞检测窗口（迭代数）
_DENSE_LIMIT = 512            # 反向稠密直解的默认变量上限


class Status(str, enum.Enum):
    OK = "ok"
    INVALID_INPUT = "invalid_input"
    MISSING_BLOCKS = "missing_blocks"
    INFEASIBLE = "infeasible"
    RESOURCE_EXCEEDED = "resource_exceeded"
    NOT_CONVERGED = "not_converged"
    CRITICAL_CAPACITY = "critical_capacity"


@dataclass(frozen=True)
class Problem:
    """一轮候选：正质量行、正常列容量、稀疏候选边、弃置许可。"""

    epsilon: float
    tolerance: float
    budget_bytes: int
    row_mass: dict
    capacity: dict
    edges: tuple
    discard_rows: frozenset = frozenset()
    max_iterations: int = 4000


@dataclass
class ColumnState:
    """一个正常原型的容量状态（供训练端确认、标注端查看）。"""

    column: str
    load: float
    capacity: float
    multiplier: float
    state: str


@dataclass
class SolveReport:
    """前向结果。status 非 ok 时分配相关字段为 None，残差字段尽量保留。"""

    revision: str | None
    status: Status
    message: str = ""
    objective: float | None = None      # 可行目标 f(P)
    lower_bound: float | None = None    # 对偶下界 g
    gap: float | None = None            # f(P) − g
    row_error: float | None = None      # 行最大误差
    column_error: float | None = None   # 列最大超出量
    iterations: int = 0
    edge_rows: list | None = None
    edge_cols: list | None = None
    values: object = None               # 与边对齐的分配质量
    columns: list = field(default_factory=list)
    discard: dict = field(default_factory=dict)
    workspace_bytes: int = 0

    @property
    def ok(self) -> bool:
        return self.status is Status.OK

    def assignment(self) -> dict:
        return {(r, c): float(v) for r, c, v in zip(self.edge_rows, self.edge_cols, self.values)}


@dataclass
class BackwardReport:
    """反向结果。临界时不交付梯度（gradient 为 None）。"""

    revision: str | None
    status: Status
    message: str = ""
    edge_rows: list | None = None
    edge_cols: list | None = None
    gradient: object = None             # 与边对齐的代价导数
    critical_columns: list = field(default_factory=list)
    columns: list = field(default_factory=list)
    workspace_bytes: int = 0

    @property
    def ok(self) -> bool:
        return self.status is Status.OK

    def gradient_dict(self) -> dict:
        return {(r, c): float(v) for r, c, v in zip(self.edge_rows, self.edge_cols, self.gradient)}


@dataclass
class _Compiled:
    """校验并规范化后的候选：索引化边表（正常边按 (行,列) 排序，弃置边按行排序）。"""

    row_ids: list
    col_ids: list
    mass: np.ndarray
    cap: np.ndarray
    n_ri: np.ndarray
    n_cj: np.ndarray
    n_cost: np.ndarray
    d_ri: np.ndarray
    d_cost: np.ndarray
    ridx: dict
    cidx: dict


@dataclass
class _Forward:
    """前向收敛后的内部量（供反向复用，属已交付输出，不计入临时空间）。"""

    u: np.ndarray
    w: np.ndarray
    p_n: np.ndarray
    p_d: np.ndarray
    loads: np.ndarray
    multipliers: np.ndarray
    states: list
    columns: list
    roots: list          # 行+正常列的连通分量根
    comp_discard: dict   # 分量根 → 是否含弃置边
    workspace_bytes: int


def _compile(problem: Problem):
    """校验公开格式并索引化。返回 (_Compiled, None) 或 (None, 非法原因)。"""
    eps, tol = problem.epsilon, problem.tolerance
    if not (isinstance(eps, (int, float)) and math.isfinite(eps) and eps > 0):
        return None, "epsilon 必须为正有限数"
    if not (isinstance(tol, (int, float)) and math.isfinite(tol) and tol > 0):
        return None, "tolerance 必须为正有限数"
    if not isinstance(problem.budget_bytes, int) or problem.budget_bytes < 0:
        return None, "budget_bytes 必须为非负整数"
    if problem.max_iterations < 1:
        return None, "max_iterations 至少为 1"
    if DISCARD in problem.row_mass:
        return None, f"行名与弃置位保留名 {DISCARD!r} 冲突"
    if DISCARD in problem.capacity:
        return None, f"列名与弃置位保留名 {DISCARD!r} 冲突"

    row_ids = list(problem.row_mass)
    if not row_ids:
        return None, "缺少行"
    mass = np.array([problem.row_mass[r] for r in row_ids], dtype=float)
    if not np.all(np.isfinite(mass)) or not np.all(mass > 0):
        return None, "行质量必须为正有限数"

    col_ids = list(problem.capacity)
    cap = np.array([problem.capacity[c] for c in col_ids], dtype=float)
    if not np.all(np.isfinite(cap)) or np.any(cap < 0):
        return None, "列容量必须为非负有限数"

    discard_rows = frozenset(problem.discard_rows)
    unknown_discard = discard_rows - set(row_ids)
    if unknown_discard:
        return None, f"弃置许可引用了未知行: {sorted(unknown_discard)}"

    ridx = {r: i for i, r in enumerate(row_ids)}
    cidx = {c: j for j, c in enumerate(col_ids)}
    seen = set()
    n_ri, n_cj, n_cost, d_ri, d_cost = [], [], [], [], []
    for e in problem.edges:
        try:
            r, c, cost = e
        except (TypeError, ValueError):
            return None, f"边格式应为 (行, 列, 代价): {e!r}"
        if not isinstance(r, str) or not isinstance(c, str):
            return None, f"边的行列必须是字符串: {e!r}"
        try:
            cost = float(cost)
        except (TypeError, ValueError):
            return None, f"边代价不是数值: {e!r}"
        if not math.isfinite(cost):
            return None, f"边代价必须有限: {e!r}"
        if r not in ridx:
            return None, f"边引用了未知行: {r!r}"
        if (r, c) in seen:
            return None, f"重复边（同一原型不会分别计容量）: {(r, c)!r}"
        seen.add((r, c))
        if c == DISCARD:
            if r not in discard_rows:
                return None, f"行 {r!r} 未在公开允许范围内使用弃置位"
            d_ri.append(ridx[r])
            d_cost.append(cost)
        elif c in cidx:
            n_ri.append(ridx[r])
            n_cj.append(cidx[c])
            n_cost.append(cost)
        else:
            return None, f"边引用了未知列（未列出的边为硬禁止，不能显式给出）: {c!r}"

    n_ri = np.array(n_ri, dtype=np.int64)
    n_cj = np.array(n_cj, dtype=np.int64)
    n_cost = np.array(n_cost, dtype=float)
    d_ri = np.array(d_ri, dtype=np.int64)
    d_cost = np.array(d_cost, dtype=float)

    degree = np.bincount(n_cj, minlength=len(col_ids)) if len(col_ids) else np.zeros(0)
    for j, c in enumerate(col_ids):
        if degree[j] > 0 and cap[j] == 0:
            return None, f"容量为 0 的列不能带候选边: {c!r}"

    # 规范化顺序：分块顺序不改变约束含义
    order = np.lexsort((n_cj, n_ri)) if len(n_ri) else n_ri
    n_ri, n_cj, n_cost = n_ri[order], n_cj[order], n_cost[order]
    dorder = np.argsort(d_ri, kind="stable") if len(d_ri) else d_ri
    d_ri, d_cost = d_ri[dorder], d_cost[dorder]

    return _Compiled(row_ids, col_ids, mass, cap, n_ri, n_cj, n_cost, d_ri, d_cost, ridx, cidx), None


def _components(comp: _Compiled):
    """行+正常列经候选边的连通分量。返回 (每节点根, 根 → 分量统计)。"""
    R, C = len(comp.mass), len(comp.cap)
    parent = list(range(R + C))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in range(len(comp.n_ri)):
        ra, rb = find(comp.n_ri[e]), find(R + comp.n_cj[e])
        if ra != rb:
            parent[ra] = rb
    roots = [find(x) for x in range(R + C)]
    info = {}
    for i in range(R):
        st = info.setdefault(roots[i], {"mass": 0.0, "cap": 0.0, "cols": 0, "discard": False})
        st["mass"] += comp.mass[i]
    for j in range(C):
        st = info.setdefault(roots[R + j], {"mass": 0.0, "cap": 0.0, "cols": 0, "discard": False})
        st["cap"] += comp.cap[j]
        st["cols"] += 1
    for e in range(len(comp.d_ri)):
        info[roots[comp.d_ri[e]]]["discard"] = True
    return roots, info


def _flow_workspace(comp: _Compiled) -> int:
    """结构预检（最大流）的估算占用。"""
    n_nodes = len(comp.mass) + len(comp.cap) + 2
    n_arcs = 2 * (len(comp.n_ri) + len(comp.d_ri) + len(comp.mass) + len(comp.cap))
    return _FLOW_BYTES_PER_NODE * n_nodes + _FLOW_BYTES_PER_ARC * n_arcs


def _maxflow_deficit(comp: _Compiled):
    """Dinic 最大流。返回 (是否可行, 缺口)。源→行(质量)，行→列(边)，列→汇(容量)，
    允许弃置的行→汇(质量)。最大流等于总质量当且仅当容量约束结构可行——
    即使每行单看能配，一组行也可能只争同一批不足的列。"""
    mass, cap = comp.mass, comp.cap
    R, C = len(mass), len(cap)
    total = float(mass.sum())
    n = R + C + 2
    src, sink = 0, n - 1
    graph = [[] for _ in range(n)]

    def add(u, v, c):
        graph[u].append([v, len(graph[v]), c])
        graph[v].append([u, len(graph[u]) - 1, 0.0])

    inf = total
    for i in range(R):
        add(src, 1 + i, mass[i])
    for e in range(len(comp.n_ri)):
        add(1 + comp.n_ri[e], 1 + R + comp.n_cj[e], inf)
    for j in range(C):
        add(1 + R + j, sink, cap[j])
    for e in range(len(comp.d_ri)):
        add(1 + comp.d_ri[e], sink, inf)

    eps = 1e-12 * max(1.0, total)
    tol = 1e-9 * max(1.0, total)
    flow = 0.0
    while flow < total - tol:
        level = [-1] * n
        level[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for e in graph[u]:
                if e[2] > eps and level[e[0]] < 0:
                    level[e[0]] = level[u] + 1
                    queue.append(e[0])
        if level[sink] < 0:
            break
        ptr = [0] * n
        while True:
            stack, path, minc = [src], [], [inf]
            found = False
            while stack:
                u = stack[-1]
                if u == sink:
                    found = True
                    break
                advanced = False
                while ptr[u] < len(graph[u]):
                    ei = ptr[u]
                    e = graph[u][ei]
                    if e[2] > eps and level[e[0]] == level[u] + 1:
                        path.append((u, ei))
                        stack.append(e[0])
                        minc.append(minc[-1] if minc[-1] < e[2] else e[2])
                        advanced = True
                        break
                    ptr[u] += 1
                if not advanced:
                    level[u] = -1
                    stack.pop()
                    if path:
                        path.pop()
                        minc.pop()
            if not found:
                break
            b = minc[-1]
            for u, ei in path:
                e = graph[u][ei]
                e[2] -= b
                graph[e[0]][e[1]][2] += b
            flow += b
    return flow >= total - tol, total - flow


def _classify_by_flow(comp: _Compiled, budget: int, revision, residual):
    """用最大流区分结构不可行与数值未收敛。"""
    flow_bytes = _flow_workspace(comp)
    if flow_bytes > budget:
        return SolveReport(
            revision, Status.RESOURCE_EXCEEDED,
            message=f"结构预检需要 {flow_bytes} 字节，超出预算 {budget}",
            **residual)
    feasible, deficit = _maxflow_deficit(comp)
    if feasible:
        return SolveReport(
            revision, Status.NOT_CONVERGED,
            message="容量约束结构可行，但迭代未在限内收敛（可重求当前精度）",
            **residual)
    return SolveReport(
        revision, Status.INFEASIBLE,
        message=f"结构不可行：最大流缺口 {deficit:.6g}（一组行只能争同一批不足的列）",
        **residual)


def _forward(comp: _Compiled, prob: Problem, revision):
    """Dykstra 块坐标上升。返回 (SolveReport, _Forward|None)。"""
    eps, tol, budget, max_iter = prob.epsilon, prob.tolerance, prob.budget_bytes, prob.max_iterations
    mass, cap = comp.mass, comp.cap
    R, C = len(mass), len(cap)
    n_ri, n_cj, n_cost = comp.n_ri, comp.n_cj, comp.n_cost
    d_ri, d_cost = comp.d_ri, comp.d_cost
    Nn, Nd = len(n_ri), len(d_ri)

    node_bytes = 8 * _FORWARD_NODE_FLOATS * (R + C)
    if node_bytes > budget:
        return SolveReport(revision, Status.RESOURCE_EXCEEDED,
                           message=f"对偶节点数组需要 {node_bytes} 字节，超出预算 {budget}"), None
    chunk = min(max(1, Nn + Nd), (budget - node_bytes) // (8 * _EDGE_CHUNK_FLOATS))
    if chunk < 1:
        return SolveReport(revision, Status.RESOURCE_EXCEEDED,
                           message="预算不足以暂存一个边块"), None
    workspace = node_bytes + 8 * _EDGE_CHUNK_FLOATS * chunk

    u = np.ones(R)
    w = np.ones(C)
    t = np.zeros(R)
    s_load = np.zeros(C)
    row_err = col_err = gap = math.inf
    f_obj = lower = math.nan
    window_err = None
    it = 0
    converged = False

    while it < max_iter:
        it += 1
        # 行更新：t_i = Σ_j k_ij w_j（弃置边 w ≡ 1），随后 u = m / t 使行和精确
        t[:] = 0.0
        for s0 in range(0, Nn, chunk):
            sl = slice(s0, min(s0 + chunk, Nn))
            k = np.exp(-n_cost[sl] / eps)
            t += np.bincount(n_ri[sl], weights=k * w[n_cj[sl]], minlength=R)
        for s0 in range(0, Nd, chunk):
            sl = slice(s0, min(s0 + chunk, Nd))
            k = np.exp(-d_cost[sl] / eps)
            t += np.bincount(d_ri[sl], weights=k, minlength=R)
        if not np.all(t > 0.0):
            residual = dict(row_error=math.inf, column_error=math.inf,
                            gap=math.inf, iterations=it, workspace_bytes=workspace)
            report = _classify_by_flow(comp, budget, revision, residual)
            report.message += "；存在数值精度内消失的有效边" if report.status is Status.NOT_CONVERGED else ""
            return report, None
        u = mass / t

        # 列和与证书：p = u·k·w（此时行和精确等于质量）
        s_load[:] = 0.0
        f_obj = 0.0
        sum_p = 0.0
        for s0 in range(0, Nn, chunk):
            sl = slice(s0, min(s0 + chunk, Nn))
            k = np.exp(-n_cost[sl] / eps)
            p = u[n_ri[sl]] * k * w[n_cj[sl]]
            s_load += np.bincount(n_cj[sl], weights=p, minlength=C)
            with np.errstate(divide="ignore", invalid="ignore"):
                plogp = np.where(p > 0, p * (np.log(p) - 1.0), 0.0)
            f_obj += float(n_cost[sl] @ p) + eps * float(plogp.sum())
            sum_p += float(p.sum())
        for s0 in range(0, Nd, chunk):
            sl = slice(s0, min(s0 + chunk, Nd))
            k = np.exp(-d_cost[sl] / eps)
            p = u[d_ri[sl]] * k
            with np.errstate(divide="ignore", invalid="ignore"):
                plogp = np.where(p > 0, p * (np.log(p) - 1.0), 0.0)
            f_obj += float(d_cost[sl] @ p) + eps * float(plogp.sum())
            sum_p += float(p.sum())
        lower = -eps * sum_p + eps * float(mass @ np.log(u)) + eps * float(cap @ np.log(w))
        gap = f_obj - lower
        row_err = float(np.max(np.abs(u * t - mass))) if R else 0.0
        col_err = float(np.max(np.maximum(0.0, s_load - cap))) if C else 0.0

        if row_err <= tol and col_err <= tol and abs(gap) <= tol:
            converged = True
            break

        # 列更新：w = min(1, cap / s̃)，s̃ = Σ u k = s_load / w
        s_tilde = np.divide(s_load, w, out=np.ones(C), where=w > 0)
        w = np.minimum(1.0, np.divide(cap, s_tilde, out=np.ones(C), where=s_tilde > 0))

        if it % _STAGNATION_WINDOW == 0:
            if window_err is not None and col_err > 0.9 * window_err and col_err > tol:
                residual = dict(row_error=row_err, column_error=col_err, gap=gap,
                                iterations=it, workspace_bytes=max(workspace, _flow_workspace(comp)))
                return _classify_by_flow(comp, budget, revision, residual), None
            window_err = col_err

    if not converged:
        residual = dict(row_error=row_err, column_error=col_err, gap=gap,
                        iterations=it, workspace_bytes=max(workspace, _flow_workspace(comp)))
        return _classify_by_flow(comp, budget, revision, residual), None

    # 收敛：输出分配、弃置量与容量状态
    p_n = np.empty(Nn)
    for s0 in range(0, Nn, chunk):
        sl = slice(s0, min(s0 + chunk, Nn))
        p_n[sl] = u[n_ri[sl]] * np.exp(-n_cost[sl] / eps) * w[n_cj[sl]]
    p_d = np.empty(Nd)
    for s0 in range(0, Nd, chunk):
        sl = slice(s0, min(s0 + chunk, Nd))
        p_d[sl] = u[d_ri[sl]] * np.exp(-d_cost[sl] / eps)

    slack_tol = 10.0 * tol
    mult_tol = eps * math.sqrt(tol)
    degree = np.bincount(n_cj, minlength=C) if C else np.zeros(0, dtype=np.int64)
    multipliers = -eps * np.log(w)
    slacks = cap - s_load
    roots, comp_info = _components(comp)
    # 容量紧实分量（无弃置、各列均满、Σcap≈Σmass）内，质量守恒强制各列恒满，
    # 乘子的零点是规范平移（(α,λ) 同加常数不改变对偶），不是松弛/受限的拐点
    max_slack = {}
    for j in range(C):
        root = roots[R + j]
        max_slack[root] = max(max_slack.get(root, 0.0), float(slacks[j]))
    tight = {root: (not st["discard"])
             and max_slack.get(root, 0.0) <= slack_tol
             and abs(st["cap"] - st["mass"]) <= slack_tol * (st["cols"] + 1)
             for root, st in comp_info.items()}
    states, columns = [], []
    for j in range(C):
        lam = float(multipliers[j])
        slack = float(slacks[j])
        if degree[j] == 0 or (slack > slack_tol and lam <= mult_tol):
            state = SLACK
        elif slack <= slack_tol and lam > mult_tol:
            state = ACTIVE
        elif slack <= slack_tol and lam <= mult_tol and tight.get(roots[R + j], False):
            state = ACTIVE  # 紧实分量内的规范零点，导数仍由钉住的 Laplacian 唯一确定
        else:
            state = CRITICAL
        states.append(state)
        columns.append(ColumnState(comp.col_ids[j], float(s_load[j]), float(cap[j]), lam, state))

    discard_mass = np.bincount(d_ri, weights=p_d, minlength=R) if Nd else np.zeros(R)
    discard = {comp.row_ids[i]: float(v) for i, v in enumerate(discard_mass)
               if comp.row_ids[i] in prob.discard_rows or v > 0}

    edge_rows = [comp.row_ids[i] for i in n_ri] + [comp.row_ids[i] for i in d_ri]
    edge_cols = [comp.col_ids[j] for j in n_cj] + [DISCARD] * Nd
    values = np.concatenate([p_n, p_d])

    report = SolveReport(
        revision, Status.OK,
        objective=f_obj, lower_bound=lower, gap=gap,
        row_error=row_err, column_error=col_err, iterations=it,
        edge_rows=edge_rows, edge_cols=edge_cols, values=values,
        columns=columns, discard=discard, workspace_bytes=workspace)
    fwd = _Forward(u, w, p_n, p_d, s_load.copy(), multipliers, states, columns,
                   roots, {r: st["discard"] for r, st in comp_info.items()}, workspace)
    return report, fwd


def solve_problem(problem: Problem, revision=None) -> SolveReport:
    """功能式前向：校验 → 求解 → 证书。"""
    comp, err = _compile(problem)
    if err:
        return SolveReport(revision, Status.INVALID_INPUT, message=err)
    report, _ = _forward(comp, problem, revision)
    return report


def _backward(comp: _Compiled, fwd: _Forward, prob: Problem, dy, d_objective,
              revision, dense_limit=_DENSE_LIMIT) -> BackwardReport:
    """隐式微分反向。dy: [(行, 列, 值)]，d_objective: 对标量最优目标的导数。"""
    eps, tol, budget = prob.epsilon, prob.tolerance, prob.budget_bytes
    mass, cap = comp.mass, comp.cap
    R, C = len(mass), len(cap)
    n_ri, n_cj = comp.n_ri, comp.n_cj
    d_ri = comp.d_ri
    Nn, Nd = len(n_ri), len(d_ri)

    critical = [comp.col_ids[j] for j in range(C) if fwd.states[j] == CRITICAL]
    if critical:
        return BackwardReport(
            revision, Status.CRITICAL_CAPACITY,
            message="存在临界列：正乘子与零松弛在约定误差内不可区分，无法保证导数",
            critical_columns=critical, columns=fwd.columns,
            workspace_bytes=fwd.workspace_bytes)
    if not (isinstance(d_objective, (int, float)) and math.isfinite(d_objective)):
        return BackwardReport(revision, Status.INVALID_INPUT,
                              message="d_objective 必须为有限数", columns=fwd.columns)

    # dy 对齐到边（身份由行和列确定）
    dy_n = np.zeros(Nn)
    dy_d = np.zeros(Nd)
    if dy:
        n_keys = n_ri * max(C, 1) + n_cj  # 正常边已按 (行,列) 排序
        seen = set()
        for entry in dy:
            try:
                r, c, v = entry
            except (TypeError, ValueError):
                return BackwardReport(revision, Status.INVALID_INPUT,
                                      message=f"dy 格式应为 (行, 列, 值): {entry!r}",
                                      columns=fwd.columns)
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                return BackwardReport(revision, Status.INVALID_INPUT,
                                      message=f"dy 值必须为有限数: {entry!r}",
                                      columns=fwd.columns)
            if (r, c) in seen:
                return BackwardReport(revision, Status.INVALID_INPUT,
                                      message=f"dy 重复引用了同一条边: {(r, c)!r}",
                                      columns=fwd.columns)
            seen.add((r, c))
            ri = comp.ridx.get(r)
            if ri is None:
                return BackwardReport(revision, Status.INVALID_INPUT,
                                      message=f"dy 引用了未知行: {r!r}", columns=fwd.columns)
            if c == DISCARD:
                pos = int(np.searchsorted(d_ri, ri))
                if pos >= Nd or d_ri[pos] != ri:
                    return BackwardReport(revision, Status.INVALID_INPUT,
                                          message=f"dy 引用了版本中不存在的边: {(r, c)!r}",
                                          columns=fwd.columns)
                dy_d[pos] = float(v)
            else:
                cj = comp.cidx.get(c)
                if cj is None:
                    return BackwardReport(revision, Status.INVALID_INPUT,
                                          message=f"dy 引用了未知列: {c!r}", columns=fwd.columns)
                pos = int(np.searchsorted(n_keys, ri * max(C, 1) + cj))
                if pos >= Nn or n_ri[pos] != ri or n_cj[pos] != cj:
                    return BackwardReport(revision, Status.INVALID_INPUT,
                                          message=f"dy 引用了版本中不存在的边: {(r, c)!r}",
                                          columns=fwd.columns)
                dy_n[pos] = float(v)

    # 反向工作区核算
    node_bytes = 8 * _BACKWARD_NODE_FLOATS * (R + C)
    lookup_bytes = 8 * Nn
    base = node_bytes + lookup_bytes
    if base > budget:
        return BackwardReport(revision, Status.RESOURCE_EXCEEDED,
                              message=f"反向节点工作区需要 {base} 字节，超出预算 {budget}",
                              columns=fwd.columns)
    dense_side = min(dense_limit, int(math.sqrt((budget - base) // 8)))
    chunk = min(max(1, Nn + Nd), (budget - base) // (8 * _EDGE_CHUNK_FLOATS))
    if chunk < 1:
        return BackwardReport(revision, Status.RESOURCE_EXCEEDED,
                              message="预算不足以暂存一个边块", columns=fwd.columns)

    # 接地：含松弛列或弃置边的分量为接地；全活跃（紧实）分量有平移零空间，
    # 钉住其中一个行节点，梯度对该平移不变
    roots = fwd.roots
    grounded = np.zeros(R + C, dtype=bool)
    for j in range(C):
        if fwd.states[j] == SLACK:
            grounded[roots[R + j]] = True
    for root, has_discard in fwd.comp_discard.items():
        if has_discard:
            grounded[root] = True
    pinned = np.zeros(R + C, dtype=bool)
    for i in range(R):
        root = roots[i]
        if not grounded[root]:
            pinned[i] = True       # 钉住该分量一个行节点
            grounded[root] = True  # 同一分量只钉一次

    active = np.array([s == ACTIVE for s in fwd.states])
    var = np.full(R + C, -1, dtype=np.int64)
    nvar = 0
    for i in range(R):
        if not pinned[i]:
            var[i] = nvar
            nvar += 1
    for j in range(C):
        if active[j]:
            var[R + j] = nvar
            nvar += 1

    diag = np.zeros(R + C)
    diag[:R] = mass
    diag[R:] = np.where(active, cap, 0.0)

    # h：行侧 Σ p·dy，活跃列侧 −Σ p·dy
    h = np.zeros(R + C)
    for s0 in range(0, Nn, chunk):
        sl = slice(s0, min(s0 + chunk, Nn))
        pdy = fwd.p_n[sl] * dy_n[sl]
        h[:R] += np.bincount(n_ri[sl], weights=pdy, minlength=R)
        am = active[n_cj[sl]]
        if np.any(am):
            h[R:] += np.bincount(n_cj[sl][am], weights=-pdy[am], minlength=C)
    for s0 in range(0, Nd, chunk):
        sl = slice(s0, min(s0 + chunk, Nd))
        h[:R] += np.bincount(d_ri[sl], weights=fwd.p_d[sl] * dy_d[sl], minlength=R)

    workspace = base + 8 * _EDGE_CHUNK_FLOATS * chunk
    if nvar == 0:
        nu = np.zeros(0)
    else:
        idx = np.where(var >= 0)[0]
        h_var = h[idx]
        diag_var = diag[idx]
        if nvar <= dense_side:
            workspace += 8 * nvar * nvar
            lap = np.zeros((nvar, nvar))
            lap[np.arange(nvar), np.arange(nvar)] = diag_var
            for s0 in range(0, Nn, chunk):
                sl = slice(s0, min(s0 + chunk, Nn))
                am = active[n_cj[sl]]
                if not np.any(am):
                    continue
                aa = var[n_ri[sl][am]]
                bb = var[R + n_cj[sl][am]]
                both = (aa >= 0) & (bb >= 0)
                if np.any(both):
                    np.subtract.at(lap, (aa[both], bb[both]), fwd.p_n[sl][am][both])
                    np.subtract.at(lap, (bb[both], aa[both]), fwd.p_n[sl][am][both])
            nu = np.linalg.solve(lap, h_var)
        else:
            # 矩阵自由共轭梯度（Jacobi 预条件），钉点映射到哨兵变量
            def matvec(x):
                y = diag_var * x
                xext = np.concatenate([x, [0.0]])
                for s0 in range(0, Nn, chunk):
                    sl = slice(s0, min(s0 + chunk, Nn))
                    am = active[n_cj[sl]]
                    if not np.any(am):
                        continue
                    aa = np.where(var[n_ri[sl][am]] >= 0, var[n_ri[sl][am]], nvar)
                    bb = np.where(var[R + n_cj[sl][am]] >= 0, var[R + n_cj[sl][am]], nvar)
                    pw = fwd.p_n[sl][am]
                    y += np.bincount(aa, weights=-pw * xext[bb], minlength=nvar + 1)[:nvar]
                    y += np.bincount(bb, weights=-pw * xext[aa], minlength=nvar + 1)[:nvar]
                return y

            tol_cg = min(1e-9, max(1e-13, tol * 0.01))
            x = np.zeros(nvar)
            r = h_var - matvec(x)
            z = r / diag_var
            p = z.copy()
            rz = float(r @ z)
            bnorm = max(1.0, math.sqrt(float(h_var @ h_var)))
            if math.sqrt(float(r @ r)) > tol_cg * bnorm:
                for _ in range(10000):
                    ap = matvec(p)
                    alpha = rz / float(p @ ap)
                    x += alpha * p
                    r -= alpha * ap
                    if math.sqrt(float(r @ r)) <= tol_cg * bnorm:
                        break
                    z = r / diag_var
                    rz_new = float(r @ z)
                    p = z + (rz_new / rz) * p
                    rz = rz_new
            nu = x

    nu_node = np.zeros(R + C)
    nu_node[var >= 0] = nu

    # 梯度：g_e = p_e/ε·(ν_行 − ν_活跃列 − dy_e) + d_objective·p_e
    g_n = np.empty(Nn)
    for s0 in range(0, Nn, chunk):
        sl = slice(s0, min(s0 + chunk, Nn))
        col_nu = np.where(active[n_cj[sl]], nu_node[R + n_cj[sl]], 0.0)
        g_n[sl] = fwd.p_n[sl] / eps * (nu_node[n_ri[sl]] - col_nu - dy_n[sl]) \
            + d_objective * fwd.p_n[sl]
    g_d = np.empty(Nd)
    for s0 in range(0, Nd, chunk):
        sl = slice(s0, min(s0 + chunk, Nd))
        g_d[sl] = fwd.p_d[sl] / eps * (nu_node[d_ri[sl]] - dy_d[sl]) \
            + d_objective * fwd.p_d[sl]

    edge_rows = [comp.row_ids[i] for i in n_ri] + [comp.row_ids[i] for i in d_ri]
    edge_cols = [comp.col_ids[j] for j in n_cj] + [DISCARD] * Nd
    return BackwardReport(
        revision, Status.OK,
        edge_rows=edge_rows, edge_cols=edge_cols,
        gradient=np.concatenate([g_n, g_d]),
        columns=fwd.columns,
        workspace_bytes=max(workspace, fwd.workspace_bytes))


def backward_problem(problem: Problem, dy, d_objective, revision=None) -> BackwardReport:
    """功能式反向：校验 → 前向 → 隐式微分。"""
    comp, err = _compile(problem)
    if err:
        return BackwardReport(revision, Status.INVALID_INPUT, message=err)
    report, fwd = _forward(comp, problem, revision)
    if not report.ok:
        return BackwardReport(revision, report.status, message=report.message,
                              columns=report.columns)
    return _backward(comp, fwd, problem, dy, d_objective, revision)


@dataclass(frozen=True)
class _Version:
    revision: str
    blocks: dict  # 块名 → 边表（不可变快照）


class ExchangeSession:
    """按块交接的版本化会话：声明/到达/钉版/更正/反传。

    已声明完成却未收到的块按缺片处理；只有收到的完整块未列出的边才是禁止。
    旧版软标签不算当前版已确认结果；已经开始的反传可沿其明确绑定的旧版分配计算。
    """

    def __init__(self, *, epsilon, tolerance, budget_bytes, row_mass, capacity,
                 discard_rows=(), max_iterations=4000):
        self._epsilon = epsilon
        self._tolerance = tolerance
        self._budget = budget_bytes
        self._row_mass = dict(row_mass)
        self._capacity = dict(capacity)
        self._discard_rows = frozenset(discard_rows)
        self._max_iterations = max_iterations
        self._declared = []
        self._arrived = {}
        self._versions = {}
        self._current = None
        self._solutions = {}
        self._forwards = {}

    @property
    def current_revision(self):
        return self._current

    def declare_block(self, name):
        if name not in self._declared:
            self._declared.append(name)

    def arrive_block(self, name, edges):
        self.declare_block(name)
        self._arrived[name] = tuple(tuple(e) for e in edges)

    def _missing(self):
        return [b for b in self._declared if b not in self._arrived]

    def _problem(self, version: _Version) -> Problem:
        edges = []
        for name in sorted(version.blocks):  # 分块顺序不改变约束含义
            edges.extend(version.blocks[name])
        return Problem(
            epsilon=self._epsilon, tolerance=self._tolerance, budget_bytes=self._budget,
            row_mass=dict(self._row_mass), capacity=dict(self._capacity),
            edges=tuple(edges), discard_rows=self._discard_rows,
            max_iterations=self._max_iterations)

    def _solve_version(self, version: _Version) -> SolveReport:
        rev = version.revision
        missing = self._missing()
        if missing:
            report = SolveReport(rev, Status.MISSING_BLOCKS,
                                 message="尚缺候选片: " + ", ".join(missing))
        else:
            problem = self._problem(version)
            comp, err = _compile(problem)
            if err:
                report = SolveReport(rev, Status.INVALID_INPUT, message=err)
            else:
                report, fwd = _forward(comp, problem, rev)
                if fwd is not None:
                    self._forwards[rev] = (comp, fwd)
        self._solutions[rev] = report
        return report

    def pin(self, revision) -> SolveReport:
        """把当前已到达的块钉成一个可绑定版本并求解。"""
        if revision in self._versions:
            return SolveReport(revision, Status.INVALID_INPUT,
                               message=f"版本 {revision!r} 已存在")
        version = _Version(revision, {k: v for k, v in self._arrived.items()})
        self._versions[revision] = version
        self._current = revision
        return self._solve_version(version)

    def replace(self, revision, block, edges) -> SolveReport:
        """更正：替换一份逻辑块，产生新候选版并求解。"""
        if revision in self._versions:
            return SolveReport(revision, Status.INVALID_INPUT,
                               message=f"版本 {revision!r} 已存在")
        if self._current is None:
            return SolveReport(revision, Status.INVALID_INPUT,
                               message="无基线版本，无法更正")
        blocks = dict(self._versions[self._current].blocks)
        blocks[block] = tuple(tuple(e) for e in edges)
        version = _Version(revision, blocks)
        self._versions[revision] = version
        self._current = revision
        return self._solve_version(version)

    def solve(self, revision=None) -> SolveReport:
        rev = revision or self._current
        if rev is None or rev not in self._versions:
            return SolveReport(rev, Status.INVALID_INPUT, message="未知版本")
        if rev not in self._solutions:
            return self._solve_version(self._versions[rev])
        return self._solutions[rev]

    def reverse(self, revision, dy, d_objective) -> BackwardReport:
        """沿明确绑定的旧版分配反传（即使它已不再是当前版）。"""
        if revision not in self._versions:
            return BackwardReport(revision, Status.INVALID_INPUT,
                                  message=f"未知版本 {revision!r}")
        report = self.solve(revision)
        if not report.ok:
            return BackwardReport(revision, report.status, message=report.message,
                                  columns=report.columns)
        comp, fwd = self._forwards[revision]
        return _backward(comp, fwd, self._problem(self._versions[revision]),
                         dy, d_objective, revision)
