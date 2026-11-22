"""容量限制软分配算子的校验夹具。

覆盖 exchange.md 的约定：精度证书、状态区分（非法/缺片/不可行/资源/未收敛）、
版本更正语义、临界列不可交付、稳定区 1e-4 中心扰动下 1e-3 的梯度绝对误差、
以及大稀疏图在约定临时空间内完成分配。
"""

import json
import unittest
from pathlib import Path

import numpy as np

import transport_op as tp
import run_scene

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def load(name):
    return json.loads((SAMPLES / name).read_text(encoding="utf-8"))


def scene_problem(data, edges, tol=None, max_iterations=4000):
    return tp.Problem(
        epsilon=data["epsilon"],
        tolerance=tol if tol is not None else data["tolerance"],
        budget_bytes=data["budget_bytes"],
        row_mass=data["rows"],
        capacity=data["capacity"],
        edges=tuple(tuple(e) for e in edges),
        discard_rows=frozenset(data["discard_rows"]),
        max_iterations=max_iterations)


def scene_r1_edges(data):
    return [tuple(e) for block in data["blocks"].values() for e in block]


def downstream_loss(report, dy, d_objective):
    assign = dict(zip(zip(report.edge_rows, report.edge_cols), report.values))
    return sum(v * assign[(r, c)] for r, c, v in dy) + d_objective * report.objective


class ForwardTests(unittest.TestCase):
    def test_scene_r1_forward_ok(self):
        data = load("scene.json")
        report = tp.solve_problem(scene_problem(data, scene_r1_edges(data)))
        self.assertEqual(report.status, tp.Status.OK)
        tol = data["tolerance"]
        self.assertLessEqual(report.row_error, tol)
        self.assertLessEqual(report.column_error, tol)
        self.assertLessEqual(abs(report.gap), tol)
        # 行质量全部分出
        for row, mass in data["rows"].items():
            got = sum(v for (r, _), v in report.assignment().items() if r == row)
            self.assertAlmostEqual(got, mass, places=9)
        # 正常列不超过容量，弃置只发生在 只允许 d
        for c in report.columns:
            self.assertLessEqual(c.load, c.capacity + tol)
            self.assertEqual(c.state, tp.ACTIVE)
        self.assertEqual(set(report.discard), {"d"})
        # 每列可超出容量至多 tol，弃置下界相应放宽
        self.assertGreaterEqual(report.discard["d"], 0.3 - 10 * tol)
        self.assertLessEqual(report.workspace_bytes, data["budget_bytes"])

    def test_hall_infeasible(self):
        # 每行单看都能配，但 x、y 只能争同一列不足的容量
        prob = tp.Problem(
            epsilon=0.25, tolerance=1e-6, budget_bytes=1 << 20,
            row_mass={"x": 1.0, "y": 1.0, "z": 1.0},
            capacity={"P": 1.5, "Q": 1.5},
            edges=(("x", "P", 0.1), ("y", "P", 0.1), ("z", "P", 0.1), ("z", "Q", 0.1)),
            discard_rows=frozenset())
        report = tp.solve_problem(prob)
        self.assertEqual(report.status, tp.Status.INFEASIBLE)
        self.assertIn("缺口", report.message)

    def test_duplicate_edges_rejected(self):
        data = load("scene.json")
        edges = scene_r1_edges(data) + [("a", "north", 0.2)]
        report = tp.solve_problem(scene_problem(data, edges))
        self.assertEqual(report.status, tp.Status.INVALID_INPUT)
        self.assertIn("重复", report.message)

    def test_forbidden_edge_cannot_be_explicit(self):
        data = load("scene.json")
        edges = scene_r1_edges(data) + [("a", "west", 0.3)]  # 未登记的列
        report = tp.solve_problem(scene_problem(data, edges))
        self.assertEqual(report.status, tp.Status.INVALID_INPUT)

    def test_discard_outside_allowance_rejected(self):
        data = load("scene.json")
        edges = scene_r1_edges(data) + [("a", "discard", 0.3)]
        report = tp.solve_problem(scene_problem(data, edges))
        self.assertEqual(report.status, tp.Status.INVALID_INPUT)

    def test_resource_exceeded(self):
        data = load("scene.json")
        prob = scene_problem(data, scene_r1_edges(data))
        prob = tp.Problem(prob.epsilon, prob.tolerance, 16, prob.row_mass,
                          prob.capacity, prob.edges, prob.discard_rows)
        report = tp.solve_problem(prob)
        self.assertEqual(report.status, tp.Status.RESOURCE_EXCEEDED)

    def test_not_converged(self):
        data = load("scene.json")
        prob = scene_problem(data, scene_r1_edges(data), max_iterations=1)
        report = tp.solve_problem(prob)
        self.assertEqual(report.status, tp.Status.NOT_CONVERGED)
        self.assertIsNotNone(report.row_error)
        self.assertIn("重求", report.message)

    def test_block_order_invariant(self):
        data = load("scene.json")
        edges = scene_r1_edges(data)
        base = tp.solve_problem(scene_problem(data, edges))
        # 同一批边按不同分块与到达顺序交接
        shuffled = [edges[i] for i in [4, 0, 8, 2, 6, 1, 3, 5, 7]]
        session = tp.ExchangeSession(
            epsilon=data["epsilon"], tolerance=data["tolerance"],
            budget_bytes=data["budget_bytes"], row_mass=data["rows"],
            capacity=data["capacity"], discard_rows=data["discard_rows"])
        session.arrive_block("z2", shuffled[5:])
        session.arrive_block("z1", shuffled[:5])
        alt = session.pin("v1")
        self.assertEqual(alt.status, tp.Status.OK)
        self.assertAlmostEqual(base.objective, alt.objective, places=12)
        amap, bmap = base.assignment(), alt.assignment()
        self.assertEqual(set(amap), set(bmap))
        for e in amap:
            self.assertAlmostEqual(amap[e], bmap[e], places=12)


class BackwardTests(unittest.TestCase):
    def test_envelope_gradient(self):
        # dy 为零、d_objective=1 时梯度就是分配本身（包络定理）
        data = load("scene.json")
        prob = scene_problem(data, scene_r1_edges(data))
        report = tp.solve_problem(prob)
        back = tp.backward_problem(prob, [], 1.0)
        self.assertEqual(back.status, tp.Status.OK)
        np.testing.assert_allclose(back.gradient, report.values, atol=1e-10)

    def test_gradient_perturbation_stable_region(self):
        # 校验夹具：稳定区域 1e-4 中心扰动，绝对误差目标 1e-3
        data = load("scene.json")
        edges = scene_r1_edges(data)
        dy = [tuple(e) for e in data["dy"]]
        d_obj = data["d_objective"]
        tol = 1e-9
        prob = scene_problem(data, edges, tol=tol, max_iterations=40000)
        back = tp.backward_problem(prob, dy, d_obj)
        self.assertEqual(back.status, tp.Status.OK)
        grad = dict(zip(zip(back.edge_rows, back.edge_cols), back.gradient))
        costs = {(r, c): co for r, c, co in prob.edges}
        h = 1e-4
        for edge, analytic in grad.items():
            vals = []
            for sign in (+h, -h):
                perturbed = tuple((r, c, co + (sign if (r, c) == edge else 0.0))
                                  for (r, c), co in costs.items())
                rep = tp.solve_problem(scene_problem(data, perturbed, tol=tol,
                                                     max_iterations=40000))
                self.assertEqual(rep.status, tp.Status.OK)
                vals.append(downstream_loss(rep, dy, d_obj))
            numeric = (vals[0] - vals[1]) / (2 * h)
            self.assertAlmostEqual(analytic, numeric, delta=1e-3,
                                   msg=f"边 {edge} 梯度超差")

    def test_capacity_edge_critical(self):
        # r1：north 恰在容量边界（零松弛、零乘子）→ 临界，不可交付
        data = load("capacity-edge.json")
        prob = scene_problem(data, scene_r1_edges(data))
        report = tp.solve_problem(prob)
        self.assertEqual(report.status, tp.Status.OK)
        states = {c.column: c.state for c in report.columns}
        self.assertEqual(states["north"], tp.CRITICAL)
        self.assertEqual(states["east"], tp.SLACK)
        back = tp.backward_problem(prob, [tuple(e) for e in data["dy"]],
                                   data["d_objective"])
        self.assertEqual(back.status, tp.Status.CRITICAL_CAPACITY)
        self.assertIsNone(back.gradient)
        self.assertEqual(back.critical_columns, ["north"])

    def test_capacity_edge_repriced_deliverable(self):
        # r2：b 类候选改价后 north 明确受限，梯度可交付并通过扰动核对
        data = load("capacity-edge.json")
        edges = ([tuple(e) for e in data["blocks"]["a-capture"]]
                 + [tuple(e) for e in data["replacements"][0]["edges"]])
        dy = [tuple(e) for e in data["dy"]]
        prob = scene_problem(data, edges, tol=1e-9, max_iterations=40000)
        back = tp.backward_problem(prob, dy, data["d_objective"])
        self.assertEqual(back.status, tp.Status.OK)
        grad = dict(zip(zip(back.edge_rows, back.edge_cols), back.gradient))
        costs = {(r, c): co for r, c, co in edges}
        h = 1e-4
        for edge, analytic in grad.items():
            vals = []
            for sign in (+h, -h):
                perturbed = tuple((r, c, co + (sign if (r, c) == edge else 0.0))
                                  for (r, c), co in costs.items())
                rep = tp.solve_problem(scene_problem(data, perturbed, tol=1e-9,
                                                     max_iterations=40000))
                vals.append(downstream_loss(rep, dy, data["d_objective"]))
            self.assertAlmostEqual(analytic, (vals[0] - vals[1]) / (2 * h),
                                   delta=1e-3, msg=f"边 {edge} 梯度超差")

    def test_dy_on_forbidden_edge_rejected(self):
        data = load("scene.json")
        prob = scene_problem(data, scene_r1_edges(data))
        back = tp.backward_problem(prob, [("a", "south", 1.0)], 0.0)
        self.assertEqual(back.status, tp.Status.INVALID_INPUT)

    def test_cg_matches_dense(self):
        # 全活跃紧实分量（Σcap=Σmass，钉住一个行节点）：共轭梯度与稠密直解一致，
        # 且规范零乘子不误报临界；导数通过 1e-4 中心扰动核对
        rng = np.random.default_rng(11)
        R, C, deg = 600, 30, 5
        rows = {f"r{i}": 1.0 for i in range(R)}
        cols = {f"c{j}": 600.0 / C for j in range(C)}  # 总容量=总质量，全活跃
        edges = []
        for i in range(R):
            for j in rng.choice(C, size=deg, replace=False):
                edges.append((f"r{i}", f"c{j}", float(rng.uniform(0.0, 1.0))))
        prob = tp.Problem(0.5, 1e-6, 64 << 20, rows, cols, tuple(edges), frozenset())
        report, fwd = tp._forward(tp._compile(prob)[0], prob, None)
        self.assertEqual(report.status, tp.Status.OK)
        self.assertTrue(all(s == tp.ACTIVE for s in fwd.states))
        dy = [(edges[0][0], edges[0][1], 0.7), (edges[123][0], edges[123][1], -0.4)]
        comp = tp._compile(prob)[0]
        dense = tp._backward(comp, fwd, prob, dy, 0.5, None, dense_limit=10000)
        cg = tp._backward(comp, fwd, prob, dy, 0.5, None, dense_limit=0)
        self.assertEqual(dense.status, tp.Status.OK)
        self.assertEqual(cg.status, tp.Status.OK)
        np.testing.assert_allclose(cg.gradient, dense.gradient, atol=1e-6)

        # 零右端（仅 d_objective）：CG 路径不退化，梯度即 d_obj·p（包络）
        env = tp._backward(comp, fwd, prob, [], 1.0, None, dense_limit=0)
        self.assertEqual(env.status, tp.Status.OK)
        np.testing.assert_allclose(env.gradient, report.values, atol=1e-8)

        # 紧实分量的导数对可扰动代价有唯一解释：抽查两条边做中心扰动
        grad = dict(zip(zip(dense.edge_rows, dense.edge_cols), dense.gradient))
        costs = {(r, c): co for r, c, co in edges}
        h = 1e-4
        for edge in (edges[0][:2], edges[123][:2]):
            vals = []
            for sign in (+h, -h):
                perturbed = tuple((r, c, co + (sign if (r, c) == edge else 0.0))
                                  for (r, c), co in costs.items())
                rep = tp.solve_problem(tp.Problem(0.5, 1e-9, 64 << 20, rows, cols,
                                                  perturbed, frozenset(),
                                                  max_iterations=20000))
                self.assertEqual(rep.status, tp.Status.OK)
                vals.append(downstream_loss(rep, dy, 0.5))
            self.assertAlmostEqual(grad[edge], (vals[0] - vals[1]) / (2 * h),
                                   delta=1e-3, msg=f"紧实分量边 {edge} 梯度超差")


class VersionTests(unittest.TestCase):
    def test_scene_timeline(self):
        data = load("scene.json")
        out = run_scene.run_scene(data)
        versions = {v["revision"]: v for v in out["版本"]}
        self.assertEqual(versions["r1"]["状态"], "ok")
        self.assertEqual(versions["r2"]["状态"], "infeasible")  # a、b 只剩 north，容量不足
        self.assertEqual(versions["r3"]["状态"], "ok")
        self.assertEqual(out["当前版本"], "r3")
        self.assertFalse(versions["r1"]["当前"])  # 旧版软标签不算当前版已确认结果
        self.assertTrue(versions["r3"]["当前"])
        self.assertIn("d", versions["r1"]["弃置"])
        # 反传沿绑定的 r1 计算：即使 a-east 在 r2 已被硬禁止
        reverse = out["反向"][0]
        self.assertEqual(reverse["revision"], "r1")
        self.assertEqual(reverse["状态"], "ok")
        graded = {(r, c) for r, c, _ in reverse["梯度"]}
        self.assertIn(("a", "east"), graded)
        self.assertEqual(len(graded), 9)

    def test_capacity_edge_timeline(self):
        data = load("capacity-edge.json")
        out = run_scene.run_scene(data)
        versions = {v["revision"]: v for v in out["版本"]}
        self.assertEqual(versions["r1"]["状态"], "ok")
        self.assertEqual(versions["r2"]["状态"], "ok")
        reverse = out["反向"][0]
        self.assertEqual(reverse["状态"], "critical_capacity")
        self.assertEqual(reverse["临界列"], ["north"])
        self.assertNotIn("梯度", reverse)

    def test_missing_block(self):
        data = load("scene.json")
        session = tp.ExchangeSession(
            epsilon=data["epsilon"], tolerance=data["tolerance"],
            budget_bytes=data["budget_bytes"], row_mass=data["rows"],
            capacity=data["capacity"], discard_rows=data["discard_rows"])
        session.declare_block("ab")  # 已声明完成却未收到 → 缺片
        session.arrive_block("cd", data["blocks"]["cd"])
        report = session.pin("r1")
        self.assertEqual(report.status, tp.Status.MISSING_BLOCKS)
        self.assertIn("ab", report.message)

    def test_reverse_unknown_revision(self):
        data = load("scene.json")
        session = tp.ExchangeSession(
            epsilon=data["epsilon"], tolerance=data["tolerance"],
            budget_bytes=data["budget_bytes"], row_mass=data["rows"],
            capacity=data["capacity"], discard_rows=data["discard_rows"])
        back = session.reverse("r9", [], 0.0)
        self.assertEqual(back.status, tp.Status.INVALID_INPUT)


class LargeSparseTests(unittest.TestCase):
    def test_large_sparse_within_budget(self):
        # 正常载荷上限：2 万行、5 千列、稀疏候选，容量只够平均载荷的 1.2 倍，
        # 部分原型真正受限，在约定临时空间内完成分配
        rng = np.random.default_rng(7)
        R, C, deg = 20000, 5000, 6
        rows = {f"r{i}": 1.0 for i in range(R)}
        cols = {f"c{j}": 1.2 * R / C for j in range(C)}
        edges = []
        for i in range(R):
            for j in rng.choice(C, size=deg, replace=False):
                edges.append((f"r{i}", f"c{j}", float(rng.uniform(0.0, 1.0))))
            if i % 5 == 0:
                edges.append((f"r{i}", "discard", 2.0))
        discard_rows = frozenset(f"r{i}" for i in range(0, R, 5))
        budget = 4 << 20
        prob = tp.Problem(0.5, 1e-6, budget, rows, cols, tuple(edges), discard_rows)
        report = tp.solve_problem(prob)
        self.assertEqual(report.status, tp.Status.OK)
        self.assertLessEqual(report.row_error, 1e-6)
        self.assertLessEqual(report.column_error, 1e-6)
        self.assertLessEqual(abs(report.gap), 1e-6)
        self.assertLessEqual(report.workspace_bytes, budget)
        # 容量竞争真实发生
        active = sum(c.state == tp.ACTIVE for c in report.columns)
        self.assertGreater(active, 1000)
        # 抽查行的质量全部分出
        assign_rows = np.array(report.edge_rows)
        for row in ("r0", "r9999", "r19999"):
            got = float(report.values[assign_rows == row].sum())
            self.assertAlmostEqual(got, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
