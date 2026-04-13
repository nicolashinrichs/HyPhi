#!/usr/bin/env python3
"""
Per-window Kuramoto PLV -> Forman curvature -> Forman Ricci-flow analysis.

Workflow:
1. Load run-wise Kuramoto phase signals from disk.
2. If missing, simulate them with software_module/KuramotoSimulations.py.
3. For each time window, merge all run nodes, compute PLV, and build one graph.
4. Compute Forman curvature, then Forman Ricci flow.
5. Save per-window computational outputs.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import networkx as nx
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

HYPHI_MODULE_DIR = SRC_DIR / "hyphi"
if str(HYPHI_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(HYPHI_MODULE_DIR))

def load_config_file(config_path: Path):
    try:
        from io_utils import load_config

        return load_config(config_path)
    except ModuleNotFoundError:
        try:
            import tomllib
        except ModuleNotFoundError:
            try:
                import tomli as tomllib
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "TOML parser not available. Install tomli for Python < 3.11."
                ) from exc

        with open(config_path, "rb") as fp:
            return tomllib.load(fp)


def load_connectome_matrix(connectivity_pkl: Path):
    from simulations import load_connectome

    W, _, _ = load_connectome(str(connectivity_pkl))
    return W


def compute_plv_matrix_window(phase_window: np.ndarray) -> np.ndarray:
    from windowing import compute_plv_matrix

    return compute_plv_matrix(phase_window)


def compute_frc_graph(graph: nx.Graph, method: str) -> nx.Graph:
    from curvatures import compute_frc

    return compute_frc(graph, method=method).copy()


def save_pickle(obj, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph_series(pkl_path: Path) -> List[nx.Graph]:
    with pkl_path.open("rb") as f:
        series = pickle.load(f)
    if not isinstance(series, list) or not series:
        raise ValueError(f"Invalid graph series in {pkl_path}")
    return series


def parse_run_ids(raw_runs: Iterable[object]) -> List[int]:
    run_ids: List[int] = []
    for item in raw_runs:
        if isinstance(item, int):
            run_ids.append(item)
        elif isinstance(item, str) and item.isdigit():
            run_ids.append(int(item))
    return run_ids


def phase_windows_from_array(arr: np.ndarray, win_size: int, win_stride: int) -> List[np.ndarray]:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        # Supports (W, N, T) and (W, T, N).
        if arr.shape[1] > arr.shape[2]:
            arr = np.transpose(arr, (0, 2, 1))
        return [np.asarray(arr[w], dtype=float) for w in range(arr.shape[0])]

    if arr.ndim != 2:
        raise ValueError(f"Unsupported phase array shape: {arr.shape}")

    phases = np.asarray(arr, dtype=float)
    if phases.shape[0] > phases.shape[1]:
        phases = phases.T

    windows: List[np.ndarray] = []
    _, n_time = phases.shape
    for start in range(0, n_time - win_size + 1, win_stride):
        windows.append(phases[:, start:start + win_size])
    return windows


def load_phase_windows(phase_path: Path, win_size: int, win_stride: int) -> List[np.ndarray]:
    suffix = phase_path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(phase_path, allow_pickle=True)
    elif suffix == ".npz":
        z = np.load(phase_path, allow_pickle=True)
        key = "phases" if "phases" in z.files else z.files[0]
        arr = z[key]
    elif suffix == ".pkl":
        with phase_path.open("rb") as f:
            arr = pickle.load(f)
        arr = np.asarray(arr)
    else:
        raise ValueError(f"Unsupported phase file extension: {phase_path}")
    return phase_windows_from_array(arr, win_size=win_size, win_stride=win_stride)


def simulate_missing_phase_files(
    missing_runs: List[int],
    phase_dir: Path,
    phase_pattern: str,
    data_dir: Path,
    win_size: int,
    win_stride: int,
    target_windows: int,
    n_osc: int,
    dt: float,
    k: float,
    omega_std: float,
    seed_base: int,
) -> None:
    from jax import random
    import jax.numpy as jnp
    from software_module.KuramotoSimulations import (
        getPLVGraphs,
        kuramotoVectorField,
        rk4,
        simulateKuramoto,
    )

    n_steps = win_size + (target_windows - 1) * win_stride
    phase_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    for rid in missing_runs:
        key_omega = random.PRNGKey(seed_base + rid * 2)
        key_init = random.PRNGKey(seed_base + rid * 2 + 1)
        omegas = omega_std * random.normal(key_omega, (n_osc,))
        init_thetas = random.uniform(key_init, (n_osc,), maxval=2 * jnp.pi)

        theta_hist = simulateKuramoto(
            lambda th, _omegas=omegas: kuramotoVectorField(th, k, _omegas),
            rk4,
            init_thetas,
            dt,
            n_steps,
        )
        theta_hist_np = np.asarray(theta_hist)
        phases_nt = theta_hist_np.T

        phase_path = phase_dir / phase_pattern.format(run=rid)
        np.save(phase_path, phases_nt)

        graph_path = data_dir / f"{rid}_connectome_kuramoto.pkl"
        if not graph_path.exists():
            run_graphs = getPLVGraphs(n_steps, win_size, win_stride, theta_hist_np)
            save_pickle(run_graphs, graph_path)


def resolve_connectome_weights(connectivity_pkl: Path, nodes_per_run: int) -> np.ndarray:
    # Reuse consolidated connectome loader from src/hyphi/simulations.py.
    W = load_connectome_matrix(connectivity_pkl)
    W = np.asarray(W, dtype=float)
    np.fill_diagonal(W, 0.0)

    if W.shape[0] == nodes_per_run:
        return W

    if nodes_per_run == 2 * W.shape[0]:
        W_pair = np.zeros((nodes_per_run, nodes_per_run), dtype=float)
        n = W.shape[0]
        W_pair[:n, :n] = W
        W_pair[n:, n:] = W
        return W_pair

    # Two-brain reduced model: take first nodes_per_brain regions from each brain.
    if nodes_per_run % 2 == 0 and (nodes_per_run // 2) <= W.shape[0]:
        n = nodes_per_run // 2
        W_small = W[:n, :n]
        W_pair = np.zeros((nodes_per_run, nodes_per_run), dtype=float)
        W_pair[:n, :n] = W_small
        W_pair[n:, n:] = W_small
        return W_pair

    return np.zeros((nodes_per_run, nodes_per_run), dtype=float)


def attach_connectome_weights(
    graph: nx.Graph,
    connectome_w: np.ndarray,
    nodes_per_run: int,
    run_ids: List[int],
) -> None:
    for u, v, data in graph.edges(data=True):
        run_idx_u = u // nodes_per_run
        run_idx_v = v // nodes_per_run
        local_u = u % nodes_per_run
        local_v = v % nodes_per_run

        plv_w = float(data.get("weight", 1.0))
        data["weight"] = plv_w
        data["plv_weight"] = plv_w

        if run_idx_u == run_idx_v:
            data["connectome_weight"] = float(connectome_w[local_u, local_v])
            data["source_run"] = int(run_ids[run_idx_u])
        else:
            data["connectome_weight"] = 0.0
            data["source_run"] = -1


def aggregate_window_graphs(window_graphs: List[nx.Graph]) -> nx.Graph:
    return nx.disjoint_union_all(window_graphs)


def build_merged_plv_graph(window_phases_by_run: List[np.ndarray], plv_threshold: float | None = None) -> nx.Graph:
    merged_phase_window = np.concatenate(window_phases_by_run, axis=0)
    C = compute_plv_matrix_window(merged_phase_window)

    # Optional sparsification to reduce runtime/memory on large merged graphs.
    if plv_threshold is not None:
        C = np.where(C >= plv_threshold, C, 0.0)

    G = nx.from_numpy_array(C, create_using=nx.Graph)
    # Slightly cheaper downstream computations without self-loops.
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def forman_ricci_flow(
    graph: nx.Graph,
    iterations: int,
    step: float,
    delta: float,
    method: str,
    weight: str = "weight",
) -> nx.Graph:
    G = graph.copy()
    eps = 1e-12
    normalized_weight = float(max(1, G.number_of_edges()))

    if not nx.get_edge_attributes(G, "formanCurvature"):
        G = compute_frc_graph(G, method=method)

    if not nx.get_edge_attributes(G, "original_FRC"):
        for u, v in G.edges():
            G[u][v]["original_FRC"] = float(G[u][v].get("formanCurvature", 0.0))

    for _ in range(iterations):
        for u, v in G.edges():
            curv = float(G[u][v].get("formanCurvature", 0.0))
            w = float(G[u][v].get(weight, 1.0))
            G[u][v][weight] = max(eps, w - step * curv * w)

        weights = nx.get_edge_attributes(G, weight)
        sumw = sum(weights.values())
        if sumw <= eps:
            break

        scale = normalized_weight / sumw
        for edge in weights:
            weights[edge] = max(eps, weights[edge] * scale)
        nx.set_edge_attributes(G, values=weights, name=weight)

        G = compute_frc_graph(G, method=method)
        rc = nx.get_edge_attributes(G, "formanCurvature")
        if not rc or (max(rc.values()) - min(rc.values()) < delta):
            break

    return G


def edge_stats(graph: nx.Graph, key: str) -> Dict[str, float]:
    vals = [float(d.get(key, 0.0)) for _, _, d in graph.edges(data=True)]
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {"min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean())}


def save_partition_visualization(
    graph: nx.Graph,
    partitions: List[List[int]],
    name: str,
    save_path: Path,
    title: str,
) -> None:
    from curvature_visualisation import visualize_graph_partitions_markers

    visualize_graph_partitions_markers(
        graph=graph,
        partitions=partitions,
        name=name,
        save=True,
        save_path=str(save_path),
        show=False,
        title=title,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-window merged PLV Forman Ricci-flow analysis.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/analysis/CCORRconfig_001.toml"),
        help="Config used for default runs/data paths/window count.",
    )
    parser.add_argument("--phase-dir", type=Path, default=None, help="Directory containing *_kuramoto_phases files.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing *_connectome_kuramoto.pkl files.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/ricci_flow_analysis"), help="Output directory.")
    parser.add_argument("--connectivity-pkl", type=Path, default=Path("software_module/connectivity_data.pkl"))
    parser.add_argument("--runs", type=int, nargs="+", default=None, help="Run IDs to process.")
    parser.add_argument(
        "--graph-construction",
        choices=["merge_signals_plv", "merge_graphs"],
        default="merge_signals_plv",
        help=(
            "merge_signals_plv: concatenate all run signals per window, then PLV, then graph. "
            "merge_graphs: load per-run window graphs and disjoint-union them."
        ),
    )
    parser.add_argument("--phase-pattern", type=str, default="{run}_kuramoto_phases.npy")
    parser.add_argument("--target-windows", type=int, default=None, help="Target number of time windows (default from config).")
    parser.add_argument("--win-size", type=int, default=None, help="Window size in samples.")
    parser.add_argument("--win-stride", type=int, default=None, help="Window stride in samples.")
    parser.add_argument("--simulate-missing-phases", dest="simulate_missing_phases", action="store_true")
    parser.add_argument("--no-simulate-missing-phases", dest="simulate_missing_phases", action="store_false")
    parser.set_defaults(simulate_missing_phases=True)
    parser.add_argument(
        "--force-simulate-phases",
        action="store_true",
        help="Regenerate all run phase files with KuramotoSimulations.py even if they already exist.",
    )
    parser.add_argument("--sim-n-osc", type=int, default=8, help="Oscillator count for simulated phases.")
    parser.add_argument("--sim-dt", type=float, default=0.01)
    parser.add_argument("--sim-k", type=float, default=2.0)
    parser.add_argument("--sim-omega-std", type=float, default=1.0)
    parser.add_argument("--sim-seed-base", type=int, default=1000)
    parser.add_argument("--sim-win-len", type=float, default=2.0, help="Simulation window length (sec).")
    parser.add_argument("--sim-win-step", type=float, default=0.5, help="Simulation window step (sec).")
    parser.add_argument("--plv-threshold", type=float, default=None, help="Optional PLV threshold (e.g., 0.2) to sparsify merged graphs.")
    parser.add_argument("--viz-dirname", type=str, default="visualizations")
    parser.add_argument(
        "--enable-visualization",
        action="store_true",
        help="Enable per-window graph visualization (disabled by default for speed).",
    )
    parser.add_argument(
        "--attach-connectome-weights",
        action="store_true",
        help="Attach connectome edge metadata/stats (disabled by default for speed).",
    )
    parser.add_argument(
        "--save-graphs",
        action="store_true",
        help="Save large graph pickle files (disabled by default for speed).",
    )
    parser.add_argument("--flow-iterations", type=int, default=30)
    parser.add_argument("--flow-step", type=float, default=1.0)
    parser.add_argument("--flow-delta", type=float, default=1e-4)
    parser.add_argument("--flow-method", choices=["1d", "augmented"], default="1d")
    parser.add_argument("--max-windows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    need_config = any(v is None for v in (args.runs, args.data_dir, args.phase_dir, args.target_windows))
    cfg = load_config_file(args.config) if need_config else {}

    runs = args.runs if args.runs is not None else parse_run_ids(cfg.get("num_kuramotos", []))
    if not runs:
        raise ValueError("No run IDs provided and none found in config num_kuramotos.")

    data_dir = args.data_dir if args.data_dir is not None else Path(cfg.get("kuramoto_loc", "data"))
    phase_dir = args.phase_dir if args.phase_dir is not None else data_dir
    target_windows = args.target_windows if args.target_windows is not None else int(cfg.get("kuramoto_time", 24))

    if args.win_size is None and args.win_stride is None:
        win_size = int(args.sim_win_len / args.sim_dt)
        win_stride = int(args.sim_win_step / args.sim_dt)
    elif args.win_size is not None and args.win_stride is not None:
        win_size = args.win_size
        win_stride = args.win_stride
    else:
        raise ValueError("Provide both --win-size and --win-stride, or neither.")

    phase_dir = phase_dir.resolve()
    data_dir = data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / args.viz_dirname if args.enable_visualization else None
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[init] "
        f"mode={args.graph_construction} runs={runs} target_windows={target_windows} "
        f"flow_iters={args.flow_iterations} phase_dir={phase_dir}",
        flush=True,
    )

    phase_windows_by_run: Dict[int, List[np.ndarray]] = {}
    series_by_run: Dict[int, List[nx.Graph]] = {}

    if args.graph_construction == "merge_graphs":
        for rid in runs:
            run_graph_path = (data_dir / f"{rid}_connectome_kuramoto.pkl").resolve()
            if not run_graph_path.exists():
                raise FileNotFoundError(f"Missing run graph file: {run_graph_path}")
            series_by_run[rid] = load_graph_series(run_graph_path)
            print(f"[load] run={rid} graph_windows={len(series_by_run[rid])}", flush=True)

        available_windows = min(len(series) for series in series_by_run.values())
        nodes_per_run = series_by_run[runs[0]][0].number_of_nodes()
        for rid in runs:
            if series_by_run[rid][0].number_of_nodes() != nodes_per_run:
                raise ValueError("All runs must have the same node count per window.")
    else:
        # Load or simulate phases in merge_signals_plv mode.
        if args.force_simulate_phases:
            if not args.simulate_missing_phases:
                raise ValueError("--force-simulate-phases requires simulation enabled.")
            print(
                f"[phase] force simulation enabled for runs={runs} with sim_n_osc={args.sim_n_osc}",
                flush=True,
            )
            simulate_missing_phase_files(
                missing_runs=runs,
                phase_dir=phase_dir,
                phase_pattern=args.phase_pattern,
                data_dir=data_dir,
                win_size=win_size,
                win_stride=win_stride,
                target_windows=target_windows,
                n_osc=args.sim_n_osc,
                dt=args.sim_dt,
                k=args.sim_k,
                omega_std=args.sim_omega_std,
                seed_base=args.sim_seed_base,
            )

        missing_phase_runs: List[int] = []
        for rid in runs:
            phase_path = phase_dir / args.phase_pattern.format(run=rid)
            if phase_path.exists():
                phase_windows_by_run[rid] = load_phase_windows(phase_path, win_size=win_size, win_stride=win_stride)
                first_shape = phase_windows_by_run[rid][0].shape
                print(f"[load] run={rid} phase_windows={len(phase_windows_by_run[rid])} first_shape={first_shape}", flush=True)
            else:
                missing_phase_runs.append(rid)

        if missing_phase_runs:
            if not args.simulate_missing_phases:
                missing_names = [str(phase_dir / args.phase_pattern.format(run=rid)) for rid in missing_phase_runs]
                raise FileNotFoundError(f"Missing phase files: {', '.join(missing_names)}")
            print(f"[phase] simulating missing runs={missing_phase_runs}", flush=True)
            simulate_missing_phase_files(
                missing_runs=missing_phase_runs,
                phase_dir=phase_dir,
                phase_pattern=args.phase_pattern,
                data_dir=data_dir,
                win_size=win_size,
                win_stride=win_stride,
                target_windows=target_windows,
                n_osc=args.sim_n_osc,
                dt=args.sim_dt,
                k=args.sim_k,
                omega_std=args.sim_omega_std,
                seed_base=args.sim_seed_base,
            )
            for rid in missing_phase_runs:
                phase_path = phase_dir / args.phase_pattern.format(run=rid)
                phase_windows_by_run[rid] = load_phase_windows(phase_path, win_size=win_size, win_stride=win_stride)
                first_shape = phase_windows_by_run[rid][0].shape
                print(f"[load] run={rid} phase_windows={len(phase_windows_by_run[rid])} first_shape={first_shape}", flush=True)

        available_windows = min(len(wins) for wins in phase_windows_by_run.values())
        nodes_per_run = phase_windows_by_run[runs[0]][0].shape[0]
        for rid in runs:
            if phase_windows_by_run[rid][0].shape[0] != nodes_per_run:
                raise ValueError("All runs must have the same node count per window.")

        if nodes_per_run != args.sim_n_osc and not args.force_simulate_phases:
            print(
                "[warn] Existing phase files were used, so sim_n_osc was not applied. "
                f"loaded_nodes_per_run={nodes_per_run}, sim_n_osc={args.sim_n_osc}. "
                "Use --force-simulate-phases to regenerate.",
                flush=True,
            )

    if available_windows < target_windows:
        raise ValueError(f"Need {target_windows} windows but only {available_windows} are available.")

    n_windows = target_windows
    if args.max_windows is not None:
        n_windows = min(n_windows, args.max_windows)

    connectome_w = None
    if args.attach_connectome_weights:
        connectome_w = resolve_connectome_weights(args.connectivity_pkl.resolve(), nodes_per_run=nodes_per_run)

    est_nodes = len(runs) * nodes_per_run if args.graph_construction == "merge_signals_plv" else len(runs) * nodes_per_run
    print(
        f"[compute] windows={n_windows} nodes_per_run={nodes_per_run} merged_nodes={est_nodes} "
        f"plv_threshold={args.plv_threshold}",
        flush=True,
    )

    all_summaries = []
    for widx in range(n_windows):
        print(f"[window {widx:02d}] start", flush=True)

        if args.graph_construction == "merge_graphs":
            window_graphs = [series_by_run[rid][widx] for rid in runs]
            agg_graph = aggregate_window_graphs(window_graphs)
        else:
            window_phase_list = [phase_windows_by_run[rid][widx] for rid in runs]
            agg_graph = build_merged_plv_graph(window_phase_list, plv_threshold=args.plv_threshold)
            expected_nodes = len(runs) * nodes_per_run
            if agg_graph.number_of_nodes() != expected_nodes:
                raise RuntimeError(
                    f"Merged graph node mismatch: {agg_graph.number_of_nodes()} != {expected_nodes}"
                )

        if connectome_w is not None:
            attach_connectome_weights(agg_graph, connectome_w, nodes_per_run, runs)

        forman_graph = compute_frc_graph(agg_graph.copy(), method=args.flow_method)
        graph_flow = forman_ricci_flow(
            graph=forman_graph,
            iterations=args.flow_iterations,
            step=args.flow_step,
            delta=args.flow_delta,
            method=args.flow_method,
            weight="weight",
        )

        base = output_dir / f"window_{widx:02d}"
        if args.save_graphs:
            save_pickle(agg_graph, base.with_name(base.name + "_aggregated_graph.pkl"))
            save_pickle(forman_graph, base.with_name(base.name + "_forman_graph.pkl"))
            save_pickle(graph_flow, base.with_name(base.name + "_ricci_flow_graph.pkl"))

        frc_vals = np.array(
            [float(d.get("formanCurvature", 0.0)) for _, _, d in forman_graph.edges(data=True)],
            dtype=float,
        )
        np.save(base.with_name(base.name + "_forman_values.npy"), frc_vals)

        # Visualization disabled by default for performance.
        # To re-enable, pass --enable-visualization.
        # partitions = [sorted(list(comp)) for comp in nx.connected_components(graph_flow)]
        # n_clusters = len(partitions)
        # viz_name = f"window_{widx:02d}_clusters_{n_clusters}"
        # save_partition_visualization(
        #     graph=graph_flow,
        #     partitions=partitions,
        #     name=viz_name,
        #     save_path=viz_dir,
        #     title=f"Ricci Flow Graph | window={widx:02d} | clusters={n_clusters}",
        # )
        n_clusters = int(nx.number_connected_components(graph_flow))

        summary = {
            "window": widx,
            "construction_mode": args.graph_construction,
            "nodes": int(agg_graph.number_of_nodes()),
            "edges": int(agg_graph.number_of_edges()),
            "plv_weight_stats": edge_stats(agg_graph, "plv_weight" if connectome_w is not None else "weight"),
            "connectome_weight_stats": (
                edge_stats(agg_graph, "connectome_weight") if connectome_w is not None else None
            ),
            "clusters": n_clusters,
            "visualization_file": None,
        }
        all_summaries.append(summary)
        print(
            f"[window {widx:02d}] done nodes={summary['nodes']} edges={summary['edges']} "
            f"plv_mean={summary['plv_weight_stats']['mean']:.6f} "
            f"clusters={summary['clusters']}",
            flush=True,
        )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Saved per-window results to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
