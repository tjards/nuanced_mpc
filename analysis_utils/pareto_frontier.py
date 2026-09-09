"""
Pareto frontier analysis utilities.
Handles data loading, metric calculations, frontier computation, and visualization.
"""
from pathlib import Path
import csv
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cala

# ==============================================================
# BASIC HELPERS
# ==============================================================

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def recursive_find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            result = recursive_find_key(value, key)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for value in obj:
            result = recursive_find_key(value, key)
            if result is not None:
                return result
    return None

def get_config_folder(data_file, data_root, config_root):
    """Map data/trials/R0_validation/R_x.../trial_.../evaluation.h5 to configs/generated/R0_validation/R_x.../trial_.../"""
    relative_folder = data_file.parent.relative_to(data_root)
    config_folder = config_root / relative_folder
    if not config_folder.exists():
        raise FileNotFoundError(f"Matching config folder not found:\n{config_folder}")
    return config_folder

def get_nominal_R0(config_folder):
    cfg = load_json(config_folder / "config_mpc.json")
    R_diag = np.asarray(cfg["R_diag"], dtype=float)
    return float(np.mean(R_diag))

def get_Ts(config_folder):
    cfg = load_json(config_folder / "config_mpc.json")
    return float(cfg["Ts"])

def get_d_max(config_folder):
    cfg = load_json(config_folder / "config_cala.json")
    d_max = recursive_find_key(cfg, "d_max")
    if d_max is None:
        raise KeyError(f"Could not find d_max in:\n{config_folder / 'config_cala.json'}")
    return float(d_max)

# ==============================================================
# R_k / R_0 DISTRIBUTION RECONSTRUCTION
# ==============================================================

def load_final_policy(training_file):
    """Load the final learned CALA mean policy."""
    with h5py.File(training_file, "r") as f:
        if "learning" not in f:
            raise KeyError(f"No learning group in {training_file}")
        if "mu" not in f["learning"]:
            raise KeyError(f"No mu dataset in {training_file}")
        mu = np.asarray(f["learning"]["mu"][-1], dtype=float)
    return mu

def reconstruct_evaluation_R_ratio(states, steps, config_folder, training_file):
    """
    Reconstruct frozen evaluation-policy R_k/R_0:
        R_k = R_0 * exp(adjustment)
    Multiple input channels are collapsed using the geometric mean of their multiplicative ratios.
    """
    cfg_plant = load_json(config_folder / "config_plant.json")
    Ts = get_Ts(config_folder)
    x0 = np.asarray(cfg_plant["x0"], dtype=float)
    mu = load_final_policy(training_file)
    n_inputs = mu.shape[1]
    feature_map = cala.RTFeatureMap(str(config_folder))
    cala_model = cala.CALA_NLD(str(config_folder), feature_map, n_inputs)
    cala_model.mu = mu.copy()
    n = len(states)
    
    # evaluation stores x_(k+1); R_k was selected from x_k
    states_for_policy = np.vstack([x0.reshape(1, -1), states[:-1]])
    if steps is not None:
        times_for_policy = steps[:n] - Ts
    else:
        times_for_policy = np.arange(n) * Ts
    
    adjustments = np.zeros((n, n_inputs))
    for k in range(n):
        phi = feature_map.build_features(states_for_policy[k], times_for_policy[k])
        adjustments[k] = cala_model.get_correction(phi)
    
    return np.exp(np.mean(adjustments, axis=1))

def reconstruct_training_R_ratio(training_file, config_folder):
    """
    Reconstruct the evolving learned-mean policy during training.
    This is the evolving mean policy, not necessarily the exact exploratory R_k applied at every training step.
    """
    d_max = get_d_max(config_folder)
    
    with h5py.File(training_file, "r") as f:
        if "learning" not in f:
            raise KeyError(f"No learning group in {training_file}")
        g = f["learning"]
        if "mu" not in g:
            raise KeyError(f"No mu dataset in {training_file}")
        if "phi" not in g:
            raise KeyError(f"No phi dataset in {training_file}")
        mu_history = np.asarray(g["mu"][:], dtype=float)
        phi_history = np.asarray(g["phi"][:], dtype=float)
    
    mu_history = np.squeeze(mu_history)
    phi_history = np.squeeze(phi_history)
    
    if mu_history.ndim == 2:
        mu_history = mu_history[:, :, None]
    if phi_history.ndim == 1:
        phi_history = phi_history[None, :]
    
    if mu_history.ndim != 3:
        raise ValueError(f"Unexpected mu shape: {mu_history.shape}")
    if phi_history.ndim != 2:
        raise ValueError(f"Unexpected phi shape: {phi_history.shape}")
    
    n = min(mu_history.shape[0], phi_history.shape[0])
    mu_history = mu_history[:n]
    phi_history = phi_history[:n]
    
    if mu_history.shape[1] != phi_history.shape[1]:
        raise ValueError(f"Feature dimension mismatch:\nmu  = {mu_history.shape}\nphi = {phi_history.shape}")
    
    local_adjustment = 2.0 * d_max * (mu_history - 0.5)
    adjustment = np.einsum("nf,nfi->ni", phi_history, local_adjustment)
    
    return np.exp(np.mean(adjustment, axis=1))

def collect_R_ratio_data(data_root, config_root, evaluation_filename, training_filename):
    """Reconstruct pooled evaluation and training R_k/R_0 samples directly from the raw trial files."""
    evaluation_ratios = []
    training_ratios = []
    evaluation_files = sorted(data_root.rglob(evaluation_filename))
    
    print(f"\nReconstructing R_k/R_0 distributions...")
    
    for i, evaluation_file in enumerate(evaluation_files, start=1):
        try:
            config_folder = get_config_folder(evaluation_file, data_root, config_root)
            training_file = evaluation_file.parent / training_filename
            
            if not training_file.exists():
                raise FileNotFoundError(f"Missing training file: {training_file}")
            
            with h5py.File(evaluation_file, "r") as f:
                if "rl_learned_R" not in f:
                    raise KeyError(f"Missing phase 'rl_learned_R'")
                gn = f["rl_learned_R"]
                states = np.asarray(gn["state"][:], dtype=float)
                if "step" in gn:
                    steps = np.asarray(gn["step"][:], dtype=float).reshape(-1)
                else:
                    steps = None
            
            evaluation_ratios.append(reconstruct_evaluation_R_ratio(states, steps, config_folder, training_file))
            training_ratios.append(reconstruct_training_R_ratio(training_file, config_folder))
        
        except Exception as exc:
            print(f"    R-ratio ERROR [{i}/{len(evaluation_files)}]: {exc}")
    
    if len(evaluation_ratios) == 0:
        raise RuntimeError("No evaluation R_k/R_0 samples reconstructed.")
    if len(training_ratios) == 0:
        raise RuntimeError("No training R_k/R_0 samples reconstructed.")
    
    return np.concatenate(evaluation_ratios), np.concatenate(training_ratios)

def print_ratio_statistics(name, ratios):
    ratios = np.asarray(ratios, dtype=float)
    print(f"\n{name} R_k/R_0")
    print("----------------------------------------")
    print(f"N samples : {len(ratios)}")
    print(f"Minimum   : {np.min(ratios):.4f}")
    print(f"Mean      : {np.mean(ratios):.4f}")
    print(f"Median    : {np.median(ratios):.4f}")
    print(f"Std       : {np.std(ratios, ddof=1):.4f}")
    print(f"Maximum   : {np.max(ratios):.4f}")
    print(f"R_k < R_0 : {100*np.mean(ratios < 1.0):.2f}%")
    print(f"R_k > R_0 : {100*np.mean(ratios > 1.0):.2f}%")

# ==============================================================
# RAW EVALUATION DATA
# ==============================================================

def calculate_metrics(states, targets, inputs):
    """
    Position tracking RMSE: sqrt(mean(||p_k - p_ref,k||^2))
    Mean control effort: mean(||u_k||^2)
    """
    n = min(len(states), len(targets), len(inputs))
    if n == 0:
        raise ValueError("Empty trajectory encountered.")
    
    states = np.asarray(states[:n], dtype=float)
    targets = np.asarray(targets[:n], dtype=float)
    inputs = np.asarray(inputs[:n], dtype=float)
    
    error_squared = np.sum((states[:, :2] - targets[:, :2]) ** 2, axis=1)
    rmse = float(np.sqrt(np.mean(error_squared)))
    effort = float(np.mean(np.sum(inputs ** 2, axis=1)))
    
    return rmse, effort

def process_evaluation_trial(evaluation_file, data_root, config_root):
    """Read one raw evaluation.h5 file and compute Static/Nuanced performance directly from the stored trajectories."""
    config_folder = get_config_folder(evaluation_file, data_root, config_root)
    R0 = get_nominal_R0(config_folder)
    
    with h5py.File(evaluation_file, "r") as f:
        if "benchmark_R" not in f:
            raise KeyError(f"Missing phase 'benchmark_R' in {evaluation_file}")
        if "rl_learned_R" not in f:
            raise KeyError(f"Missing phase 'rl_learned_R' in {evaluation_file}")
        
        gs = f["benchmark_R"]
        gn = f["rl_learned_R"]
        
        for group_name, group in [("benchmark_R", gs), ("rl_learned_R", gn)]:
            for key in ["state", "target", "input"]:
                if key not in group:
                    raise KeyError(f"Missing '{group_name}/{key}' in {evaluation_file}")
        
        static_states = np.asarray(gs["state"][:], dtype=float)
        static_targets = np.asarray(gs["target"][:], dtype=float)
        static_inputs = np.asarray(gs["input"][:], dtype=float)
        nuanced_states = np.asarray(gn["state"][:], dtype=float)
        nuanced_targets = np.asarray(gn["target"][:], dtype=float)
        nuanced_inputs = np.asarray(gn["input"][:], dtype=float)
    
    static_rmse, static_effort = calculate_metrics(static_states, static_targets, static_inputs)
    nuanced_rmse, nuanced_effort = calculate_metrics(nuanced_states, nuanced_targets, nuanced_inputs)
    
    rmse_change_pct = 100.0 * (nuanced_rmse - static_rmse) / static_rmse
    effort_change_pct = 100.0 * (nuanced_effort - static_effort) / static_effort
    
    return {
        "file": str(evaluation_file),
        "R0": R0,
        "static_rmse": static_rmse,
        "nuanced_rmse": nuanced_rmse,
        "rmse_change_pct": rmse_change_pct,
        "static_effort": static_effort,
        "nuanced_effort": nuanced_effort,
        "effort_change_pct": effort_change_pct,
    }

def collect_evaluation_data(data_root, config_root, evaluation_filename):
    files = sorted(data_root.rglob(evaluation_filename))
    print(f"\nFound {len(files)} raw evaluation files.")
    results = []
    
    for i, filename in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {filename}")
        try:
            results.append(process_evaluation_trial(filename, data_root, config_root))
        except Exception as exc:
            print(f"    ERROR: {exc}")
    
    print(f"\nSuccessfully processed {len(results)} evaluation trials.")
    if len(results) == 0:
        raise RuntimeError("No valid evaluation trials were loaded.")
    
    return results

# ==============================================================
# GROUP TRIALS BY R0
# ==============================================================

def mean_operating_points(evaluation_results):
    """One mean Static point and one mean Nuanced point per R0."""
    R0_values = sorted(set(float(r["R0"]) for r in evaluation_results))
    points = []
    
    for R0 in R0_values:
        rows = [r for r in evaluation_results if np.isclose(r["R0"], R0)]
        points.append({
            "R0": R0,
            "n_trials": len(rows),
            "static_effort": float(np.mean([r["static_effort"] for r in rows])),
            "static_rmse": float(np.mean([r["static_rmse"] for r in rows])),
            "nuanced_effort": float(np.mean([r["nuanced_effort"] for r in rows])),
            "nuanced_rmse": float(np.mean([r["nuanced_rmse"] for r in rows])),
        })
    
    return points

# ==============================================================
# GENERIC PARETO HELPERS
# ==============================================================

def nondominated(points, effort_key="effort", rmse_key="rmse"):
    """
    Return the nondominated subset for two minimization objectives.
    Point i is dominated when there exists point j such that:
        effort_j <= effort_i and rmse_j <= rmse_i
    with at least one inequality strict.
    """
    frontier = []
    
    for i, point_i in enumerate(points):
        dominated = False
        
        for j, point_j in enumerate(points):
            if i == j:
                continue
            
            no_worse_effort = point_j[effort_key] <= point_i[effort_key]
            no_worse_rmse = point_j[rmse_key] <= point_i[rmse_key]
            strictly_better = (point_j[effort_key] < point_i[effort_key] or point_j[rmse_key] < point_i[rmse_key])
            
            if no_worse_effort and no_worse_rmse and strictly_better:
                dominated = True
                break
        
        if not dominated:
            frontier.append(point_i.copy())
    
    return sorted(frontier, key=lambda p: p[effort_key])

def build_controller_points(mean_points, controller):
    """Convert mean operating-point rows to a generic Pareto format."""
    if controller not in {"static", "nuanced"}:
        raise ValueError("controller must be 'static' or 'nuanced'")
    
    result = []
    for row in mean_points:
        result.append({
            "controller": controller,
            "R0": row["R0"],
            "n_trials": row["n_trials"],
            "effort": row[f"{controller}_effort"],
            "rmse": row[f"{controller}_rmse"],
        })
    
    return result

def interpolate_frontier_rmse(effort, frontier):
    """Piecewise-linear interpolation of an empirical frontier. No extrapolation is performed."""
    if len(frontier) < 2:
        return np.nan
    
    x = np.asarray([p["effort"] for p in frontier], dtype=float)
    y = np.asarray([p["rmse"] for p in frontier], dtype=float)
    
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    
    if effort < x.min() or effort > x.max():
        return np.nan
    
    return float(np.interp(effort, x, y))

# ==============================================================
# STATIC FRONTIER GAIN
# ==============================================================

def calculate_static_frontier_gain(mean_points):
    static_points = build_controller_points(mean_points, "static")
    nuanced_points = build_controller_points(mean_points, "nuanced")
    static_frontier = nondominated(static_points)
    results = []
    
    for point in nuanced_points:
        static_equivalent_rmse = interpolate_frontier_rmse(point["effort"], static_frontier)
        
        if np.isfinite(static_equivalent_rmse):
            pareto_gain = static_equivalent_rmse - point["rmse"]
            pareto_gain_pct = 100.0 * pareto_gain / static_equivalent_rmse
        else:
            pareto_gain = np.nan
            pareto_gain_pct = np.nan
        
        results.append({
            "R0": point["R0"],
            "nuanced_effort": point["effort"],
            "nuanced_rmse": point["rmse"],
            "static_frontier_rmse": static_equivalent_rmse,
            "pareto_gain": pareto_gain,
            "pareto_gain_pct": pareto_gain_pct,
        })
    
    return results, static_frontier

# ==============================================================
# COMBINED STATIC + NUANCED FRONTIER
# ==============================================================

def calculate_combined_frontier(mean_points):
    static_points = build_controller_points(mean_points, "static")
    nuanced_points = build_controller_points(mean_points, "nuanced")
    static_frontier = nondominated(static_points)
    nuanced_frontier = nondominated(nuanced_points)
    combined_frontier = nondominated(static_points + nuanced_points)
    
    return static_frontier, nuanced_frontier, combined_frontier

# ==============================================================
# CSV OUTPUTS
# ==============================================================

def save_trial_results_csv(evaluation_results, output_folder):
    filename = output_folder / "paired_trial_results_raw.csv"
    fields = ["file", "R0", "static_rmse", "nuanced_rmse", "rmse_change_pct", "static_effort", "nuanced_effort", "effort_change_pct"]
    
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in evaluation_results:
            writer.writerow({key: row[key] for key in fields})
    
    print(f"Saved: {filename}")

def save_mean_points_csv(mean_points, output_folder):
    filename = output_folder / "mean_operating_points_raw.csv"
    fields = ["R0", "n_trials", "static_effort", "static_rmse", "nuanced_effort", "nuanced_rmse"]
    
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(mean_points)
    
    print(f"Saved: {filename}")

def save_frontier_gain_csv(pareto_results, output_folder):
    filename = output_folder / "pareto_frontier_gain_raw.csv"
    fields = ["R0", "nuanced_effort", "nuanced_rmse", "static_frontier_rmse", "pareto_gain", "pareto_gain_pct"]
    
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pareto_results)
    
    print(f"Saved: {filename}")

def save_combined_frontier_csv(combined_frontier, output_folder):
    filename = output_folder / "combined_pareto_frontier_raw.csv"
    fields = ["controller", "R0", "n_trials", "effort", "rmse"]
    
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for point in combined_frontier:
            writer.writerow({key: point[key] for key in fields})
    
    print(f"Saved: {filename}")

# ==============================================================
# PLOT: PAIRED EFFECT
# ==============================================================

def plot_paired_effects(evaluation_results, output_folder):
    effort_change = np.asarray([r["effort_change_pct"] for r in evaluation_results], dtype=float)
    rmse_change = np.asarray([r["rmse_change_pct"] for r in evaluation_results], dtype=float)
    R0 = np.asarray([r["R0"] for r in evaluation_results], dtype=float)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(effort_change, rmse_change, c=R0, s=50, alpha=0.75)
    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.axvline(0.0, linestyle="--", linewidth=1.2)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(r"Nominal $R_0$")
    ax.set_xlabel("Change in mean control effort (%)")
    ax.set_ylabel("Change in tracking RMSE (%)")
    ax.set_title("Paired effect of Nuanced MPC")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    filename = output_folder / "paired_tracking_effort_effect_raw.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")

# ==============================================================
# PLOT: NORMALIZED TRACKING-EFFORT EFFICIENCY
# ==============================================================

def plot_normalized_tracking_efficiency(evaluation_results, output_folder):
    static_rmse = np.asarray([r["static_rmse"] for r in evaluation_results], dtype=float)
    nuanced_rmse = np.asarray([r["nuanced_rmse"] for r in evaluation_results], dtype=float)
    static_effort = np.asarray([r["static_effort"] for r in evaluation_results], dtype=float)
    nuanced_effort = np.asarray([r["nuanced_effort"] for r in evaluation_results], dtype=float)
    R0_all = np.asarray([r["R0"] for r in evaluation_results], dtype=float)
    
    tracking_ratio = static_rmse / nuanced_rmse
    effort_ratio = nuanced_effort / static_effort
    eta = tracking_ratio / effort_ratio
    efficiency_gain = 100.0 * (eta - 1.0)
    
    R0_values = np.array(sorted(set(R0_all)), dtype=float)
    mean_gain = []
    std_gain = []
    n_trials = []
    
    for R0 in R0_values:
        mask = np.isclose(R0_all, R0)
        values = efficiency_gain[mask]
        mean_gain.append(np.mean(values))
        std_gain.append(np.std(values, ddof=1) if len(values) > 1 else 0.0)
        n_trials.append(len(values))
    
    mean_gain = np.asarray(mean_gain, dtype=float)
    std_gain = np.asarray(std_gain, dtype=float)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(R0_values, mean_gain, marker="o", linewidth=2.2, label="Mean normalized efficiency gain")
    ax.fill_between(R0_values, mean_gain - std_gain, mean_gain + std_gain, alpha=0.20, label=r"$\pm 1$ std. dev.")
    ax.axhline(0.0, linestyle="--", linewidth=1.3, color="black", label="No net efficiency change")
    
    for i, R0 in enumerate(R0_values):
        ax.annotate(f"n={n_trials[i]}", (R0_values[i], mean_gain[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    
    ax.set_xlabel(r"Nominal $R_0$")
    ax.set_ylabel("Normalized tracking-efficiency gain (%)")
    ax.set_title("Tracking improvement normalized by control effort")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    
    filename = output_folder / "normalized_tracking_efficiency_raw.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nNormalized tracking-effort efficiency")
    print("----------------------------------------")
    print(f"N trials overall    : {len(efficiency_gain)}")
    print(f"Overall mean gain   : {np.mean(efficiency_gain):+.2f}%")
    print(f"Overall median gain : {np.median(efficiency_gain):+.2f}%")
    print(f"Overall std         : {np.std(efficiency_gain, ddof=1):.2f}%")
    print(f"Trials above zero   : {100*np.mean(efficiency_gain > 0):.1f}%")
    
    for i, R0 in enumerate(R0_values):
        print(f"R0={R0:g}: mean={mean_gain[i]:+.2f}%, std={std_gain[i]:.2f}%, n={n_trials[i]}")
    
    print(f"\nSaved: {filename}")

# ==============================================================
# PLOT: R_RATIO DISTRIBUTION
# ==============================================================

def plot_R_ratio_distribution(evaluation_ratios, training_ratios, output_folder, ratio_bin_width):
    """Plot the pooled Evaluation vs Training contextual-weighting distribution graph."""
    all_ratios = np.concatenate([evaluation_ratios, training_ratios])
    minimum = float(np.min(all_ratios))
    maximum = float(np.max(all_ratios))
    
    lower = np.floor(minimum / ratio_bin_width) * ratio_bin_width
    upper = np.ceil(maximum / ratio_bin_width) * ratio_bin_width
    
    if np.isclose(lower, upper):
        upper = lower + ratio_bin_width
    
    edges = np.arange(lower, upper + ratio_bin_width, ratio_bin_width)
    centres = (edges[:-1] + edges[1:]) / 2.0
    
    fig, ax = plt.subplots(figsize=(10, 7))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    eval_color = color_cycle[0]
    train_color = color_cycle[1] if len(color_cycle) > 1 else "tab:orange"
    
    eval_counts, _ = np.histogram(evaluation_ratios, bins=edges)
    eval_frequency = eval_counts / np.sum(eval_counts)
    train_counts, _ = np.histogram(training_ratios, bins=edges)
    train_frequency = train_counts / np.sum(train_counts)
    
    ax.fill_between(centres, 0.0, eval_frequency, color=eval_color, alpha=0.18)
    ax.plot(centres, eval_frequency, color=eval_color, linestyle="-", linewidth=2.3, marker="o", markersize=5, 
            markerfacecolor=eval_color, markeredgecolor="white", markeredgewidth=0.7, 
            label=f"Evaluation (N={len(evaluation_ratios):,})")
    
    ax.fill_between(centres, 0.0, train_frequency, color=train_color, alpha=0.18)
    ax.plot(centres, train_frequency, color=train_color, linestyle="--", linewidth=2.3, marker="o", markersize=5, 
            markerfacecolor=train_color, markeredgecolor="white", markeredgewidth=0.7, 
            label=f"Training (N={len(training_ratios):,})")
    
    ax.axvline(1.0, linestyle=":", linewidth=2.0, color="black", label=r"Nominal $R_k/R_0=1$")
    ax.set_xlabel(r"Contextual weighting ratio $R_k/R_0$")
    ax.set_ylabel("Fraction of samples")
    ax.set_title("Distribution of learned contextual weighting")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    
    filename = output_folder / "R_ratio_distribution_both_raw.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print_ratio_statistics("Evaluation", evaluation_ratios)
    print_ratio_statistics("Training", training_ratios)
    print(f"\nSaved: {filename}")

# ==============================================================
# PLOT: STATIC FRONTIER + NUANCED POINTS
# ==============================================================

def plot_static_frontier_analysis(mean_points, pareto_results, static_frontier, output_folder):
    static_points = build_controller_points(mean_points, "static")
    nuanced_points = build_controller_points(mean_points, "nuanced")
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    static_scatter = ax.scatter([p["effort"] for p in static_points], [p["rmse"] for p in static_points], 
                                marker="o", s=55, label="Static MPC")
    static_color = static_scatter.get_facecolor()[0]
    
    nuanced_scatter = ax.scatter([p["effort"] for p in nuanced_points], [p["rmse"] for p in nuanced_points], 
                                 marker="o", s=65, label="Nuanced MPC", zorder=4)
    nuanced_color = nuanced_scatter.get_facecolor()[0]
    
    ax.plot([p["effort"] for p in static_frontier], [p["rmse"] for p in static_frontier], 
            color=static_color, linewidth=2.5, linestyle="-", label="Static Pareto frontier")
    
    for p in static_points:
        ax.annotate(f"{p['R0']:g}", (p["effort"], p["rmse"]), xytext=(-5, 6), textcoords="offset points", 
                    ha="right", fontsize=8, color=static_color)
    
    for p in nuanced_points:
        ax.annotate(f"{p['R0']:g}", (p["effort"], p["rmse"]), xytext=(5, 6), textcoords="offset points", 
                    ha="left", fontsize=8, color=nuanced_color)
    
    for result in pareto_results:
        if not np.isfinite(result["static_frontier_rmse"]):
            continue
        ax.plot([result["nuanced_effort"], result["nuanced_effort"]], 
                [result["nuanced_rmse"], result["static_frontier_rmse"]], 
                linestyle=":", linewidth=1.0, color="0.4", alpha=0.7)
    
    ax.set_xlabel(r"Mean control effort $\|u\|^2$")
    ax.set_ylabel("Position tracking RMSE")
    ax.set_title("Nuanced MPC relative to Static-MPC Pareto frontier")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    
    filename = output_folder / "pareto_frontier_analysis_raw.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")

# ==============================================================
# PLOT: BOTH EMPIRICAL FRONTIERS + COMBINED FRONTIER
# ==============================================================

def plot_combined_frontiers(mean_points, static_frontier, nuanced_frontier, combined_frontier, output_folder):
    """
    Plot Static and Nuanced MPC mean operating points and their nondominated frontiers.
    Styling: clean circular markers, black dashed connector between Static/Nuanced points for same R0.
    """
    static_points = build_controller_points(mean_points, "static")
    nuanced_points = build_controller_points(mean_points, "nuanced")
    static_points = sorted(static_points, key=lambda p: p["R0"])
    nuanced_points = sorted(nuanced_points, key=lambda p: p["R0"])
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    static_line, = ax.plot([p["effort"] for p in static_points], [p["rmse"] for p in static_points], 
                           marker="o", markersize=6, linewidth=1.5, alpha=0.35, label="Static MPC means")
    
    nuanced_line, = ax.plot([p["effort"] for p in nuanced_points], [p["rmse"] for p in nuanced_points], 
                            marker="o", markersize=6, linewidth=1.5, alpha=0.35, label="Nuanced MPC means")
    
    static_color = static_line.get_color()
    nuanced_color = nuanced_line.get_color()
    
    ax.plot([p["effort"] for p in static_frontier], [p["rmse"] for p in static_frontier], 
            marker="o", markersize=7, linewidth=2.8, color=static_color)
    
    ax.plot([p["effort"] for p in nuanced_frontier], [p["rmse"] for p in nuanced_frontier], 
            marker="o", markersize=7, linewidth=2.8, color=nuanced_color)
    
    # Dashed black connectors for matching R0 pairs
    for static_point in static_points:
        R0 = static_point["R0"]
        matches = [p for p in nuanced_points if np.isclose(p["R0"], R0)]
        if len(matches) == 0:
            continue
        nuanced_point = matches[0]
        ax.plot([static_point["effort"], nuanced_point["effort"]], 
                [static_point["rmse"], nuanced_point["rmse"]], 
                linestyle="--", linewidth=1.0, color="black", alpha=0.60, zorder=1)
    
    # Label all R0 values
    R0_values = [p["R0"] for p in static_points if any(np.isclose(p["R0"], q["R0"]) for q in nuanced_points)]
    
    if len(R0_values) > 0:
        label_indices = list(range(len(R0_values)))
        all_rmse = np.asarray([p["rmse"] for p in static_points] + [p["rmse"] for p in nuanced_points], dtype=float)
        y_span = max(float(np.max(all_rmse) - np.min(all_rmse)), 1e-6)
        y_offset = 0
        x_offset = 0.1
        
        for idx in label_indices:
            R0 = R0_values[idx]
            static_point = next(p for p in static_points if np.isclose(p["R0"], R0))
            nuanced_point = next(p for p in nuanced_points if np.isclose(p["R0"], R0))
            x_label = (static_point["effort"] + x_offset)
            y_label = (static_point["rmse"] + y_offset)
            ax.text(x_label, y_label, rf"$R_0={R0:.2f}$", color="black", fontsize=10, ha="center", va="bottom", zorder=8)
    
    ax.set_xlabel(r"Mean control effort $\|u\|^2$")
    ax.set_ylabel("Position tracking RMSE")
    ax.set_title("Static and Nuanced MPC Pareto frontiers")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    
    filename = output_folder / "combined_pareto_frontiers_raw.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")

# ==============================================================
# PRINT PARETO SUMMARY
# ==============================================================

def print_pareto_summary(pareto_results, static_frontier, nuanced_frontier, combined_frontier):
    print("\n" + "="*50)
    print("STATIC MPC NONDOMINATED FRONTIER")
    print("="*50)
    
    for point in static_frontier:
        print(f"R0={point['R0']:g}: effort={point['effort']:.4f}, RMSE={point['rmse']:.4f}")
    
    print("\n" + "="*50)
    print("NUANCED MPC RELATIVE TO STATIC FRONTIER")
    print("="*50)
    
    for result in pareto_results:
        if np.isfinite(result["pareto_gain_pct"]):
            print(f"R0={result['R0']:g}: effort={result['nuanced_effort']:.4f}, Nuanced RMSE={result['nuanced_rmse']:.4f}, "
                  f"Static frontier RMSE={result['static_frontier_rmse']:.4f}, gain={result['pareto_gain_pct']:+.2f}%")
        else:
            print(f"R0={result['R0']:g}: outside tested Static-MPC frontier range")
    
    print("\n" + "="*50)
    print("COMBINED NONDOMINATED FRONTIER")
    print("="*50)
    
    static_count = 0
    nuanced_count = 0
    
    for point in combined_frontier:
        if point["controller"] == "static":
            static_count += 1
        else:
            nuanced_count += 1
        print(f"{point['controller']:7s} R0={point['R0']:g}: effort={point['effort']:.4f}, RMSE={point['rmse']:.4f}")
    
    total = len(combined_frontier)
    print(f"\nCombined frontier points : {total}")
    print(f"Static contributions      : {static_count}")
    print(f"Nuanced contributions     : {nuanced_count}")
    
    if total > 0:
        print(f"Nuanced share of combined frontier: {100.0 * nuanced_count / total:.1f}%")
