#!/usr/bin/env python3
"""Worked example: one ventricular mesh in, a pressure-volume loop out.

The only input is `data/076_100_120k/reference_mesh.vtu` -- the reference
(unloaded) tetrahedral mesh of geometry 076, with a region label on every node.
The directory name also carries the two physiological settings of the case,
passive stiffness 100 and Tmax 120k.  Geometry 076 is one of the geometries
`../output/best_model.pt` held out of training, so the network has never seen
this anatomy.

No finite-element results are read anywhere in this script.  Everything below is
predicted.

    1. The mesh goes through the network once per volume on a sampling grid,
       which gives the pressure surface P(V, t) of this ventricle over a
       cardiac cycle, together with the generalized coordinates q(V, t) and the
       spatial modes phi(n).  The modes depend on the mesh alone, so one pass
       fixes them for good.

    2. That surface is the ventricular element of a lumped-parameter
       circulation -- left atrium, mitral valve, aorta, systemic resistance and
       venous return.  Integrating the circuit lets the ventricle find its own
       operating point instead of being told one: filling, isovolumic
       contraction, ejection and relaxation all emerge from the coupling, and
       the last cycle of the integration is the pressure-volume loop.

    3. Along that loop the volume is no longer fixed, so reading q at the loop's
       own (V, t) and recombining it with the spatial modes gives the wall
       motion that belongs to it: u(n, t) = sum_k q_k(V(t), t) phi_k(n).

Outputs

    output/pressure/pv_loop.csv           time, volume and pressure around the loop
    output/pressure/pv_loop.png           the loop
    output/disp/vtk/cycle_*.vtu           deformed mesh, one file per frame
    output/disp/vtk/*.pvd                 ParaView time-series index
    output/disp/displacement_cycle.mp4    the beating ventricle

Run it with no arguments:

    python3 example.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_undirected


HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = HERE.parent / "output" / "best_model.pt"
DEFAULT_CASE_DIR = HERE / "data" / "076_100_120k"
DEFAULT_OUTPUT_DIR = HERE / "output"
TRAIN_SCRIPT = HERE.parent / "train.py"
MESH_FILE_NAME = "reference_mesh.vtu"

# The meshes are in centimetres; every length this script reports is millimetres.
MM_PER_UNIT = 10.0


def load_training_module(path: Path):
    """Import train.py, the single source of truth for the model and its inputs.

    The example never re-implements normalization or feature construction: any
    divergence between training and inference would be silent, so both come
    from the same module.
    """
    spec = importlib.util.spec_from_file_location("cgfenet_train", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


net = load_training_module(TRAIN_SCRIPT)


# --------------------------------------------------------------------------- #
# Model inputs
#
# How a mesh and an operating point become network input.  These are the
# formulas the cached training set was built with, so inference has to apply
# exactly the same ones; the statistics they use travel inside the checkpoint.
# --------------------------------------------------------------------------- #

def safe_coord_std(coord_std: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(coord_std) < 1e-6, 1.0, coord_std).astype(np.float32)


def build_graph_input(base_graph: Data, coord_mean: np.ndarray, coord_std: np.ndarray) -> Data:
    """Node features: standardized coordinates concatenated with the region one-hot."""
    graph = base_graph.clone()
    coords = graph.pos.numpy()
    coords_norm = (coords - coord_mean) / coord_std
    graph.x = torch.from_numpy(
        np.concatenate([coords_norm, graph.region.numpy()], axis=1).astype(np.float32)
    )
    return graph


def build_dynamic_features(
    volume_seq: np.ndarray,
    stiff: float,
    tmax: float,
    stats: NormalizationStats,
) -> np.ndarray:
    """Per-time-step conditioning tensor [T, 3] = standardized volume, stiffness, Tmax.

    Time is represented once, explicitly and periodically, by
    CircularFourierEmbedding inside the model.  The dynamic tensor therefore
    carries only the physical operating parameters.
    """
    volume_norm = (volume_seq - stats.volume_mean) / stats.volume_std
    stiff_norm = np.full_like(volume_norm, (stiff - stats.stiff_mean) / stats.stiff_std)
    tmax_norm = np.full_like(volume_norm, (tmax - stats.tmax_mean) / stats.tmax_std)
    return np.stack([volume_norm, stiff_norm, tmax_norm], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Reading the mesh file
#
# Turning a .vtu into a graph is inference-side work: train.py consumes a cache
# that was built once and never opens a mesh itself.
# --------------------------------------------------------------------------- #

def parse_case_name(case_name: str) -> Tuple[str, float, float]:
    """<geometry>_<passive stiffness>_<Tmax> -> ("076", 100.0, 120.0)."""
    parts = case_name.split("_")
    if len(parts) != 3:
        raise ValueError(f"Unexpected case name: {case_name}")
    return parts[0], float(parts[1]), float(parts[2].rstrip("kK"))


def extract_tetra(mesh: pv.DataSet) -> np.ndarray:
    celltypes = np.asarray(mesh.celltypes)
    if np.any(celltypes != 10):
        raise ValueError(f"Non-tetra cell types found: {np.unique(celltypes).tolist()}")
    return mesh.cells.reshape(-1, 5)[:, 1:].astype(np.int64)


def build_edge_index(tetra: np.ndarray, num_nodes: int) -> torch.Tensor:
    """Undirected graph over the tetrahedral element edges."""
    pairs: List[Tuple[int, int]] = []
    for tet in tetra:
        nodes = [int(v) for v in tet]
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = nodes[i], nodes[j]
                pairs.append((a, b))
                pairs.append((b, a))
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    return to_undirected(edge_index, num_nodes=num_nodes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")

    physiology = parser.add_argument_group("physiology of this ventricle")
    physiology.add_argument("--stiffness", type=float, default=None,
                            help="Passive stiffness C. Defaults to the middle field of the "
                                 "case directory name (076_100_120k -> 100).")
    physiology.add_argument("--tmax", type=float, default=None,
                            help="Peak active tension Tmax, in thousands. Defaults to the last "
                                 "field of the case directory name (076_100_120k -> 120).")

    surface = parser.add_argument_group("pressure surface")
    surface.add_argument("--volume-min", type=float, default=50.0,
                         help="Smallest sampled cavity volume (mL).")
    surface.add_argument("--volume-max", type=float, default=120.0,
                         help="Largest sampled cavity volume (mL); the loop cannot leave this range.")
    surface.add_argument("--volume-samples", type=int, default=30)
    surface.add_argument("--time-samples", type=int, default=800,
                         help="Time points per cycle, one per millisecond.")

    circulation = parser.add_argument_group("circulation")
    circulation.add_argument("--cycles", type=int, default=10,
                             help="Cycles integrated before the loop is read off the last one.")
    circulation.add_argument("--time-step", type=float, default=0.5,
                             help="Integration step (ms).")

    output = parser.add_argument_group("output")
    output.add_argument("--movie-frames", type=int, default=120)
    output.add_argument("--movie-fps", type=int, default=24)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# The ventricle: one mesh, one network, one pressure surface
# --------------------------------------------------------------------------- #

@dataclass
class Mesh:
    case_name: str
    stiff: float
    tmax: float
    graph: object                  # torch_geometric Data, model-ready node features
    points_mm: np.ndarray          # [N, 3] reference coordinates
    tetra: np.ndarray              # [C, 4]
    surface_faces: np.ndarray      # [F, 3] triangles indexing into points_mm


def read_mesh(
    case_dir: Path,
    stats,
    mm_per_unit: float,
    stiffness: Optional[float] = None,
    tmax: Optional[float] = None,
) -> Mesh:
    """Build everything the model and the writers need from the one input file.

    The two physiological settings ride along in the directory name --
    <geometry>_<passive stiffness>_<Tmax>, so 076_100_120k means C = 100 and
    Tmax = 120k -- and either can be overridden to ask the same ventricle how it
    would behave stiffer, or with weaker contraction.
    """
    mesh_path = case_dir / MESH_FILE_NAME
    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing input mesh: {mesh_path}")
    _, default_stiffness, default_tmax = parse_case_name(case_dir.name)
    stiff = float(default_stiffness if stiffness is None else stiffness)
    tmax = float(default_tmax if tmax is None else tmax)

    mesh = pv.read(mesh_path)
    tetra = extract_tetra(mesh)
    points = np.asarray(mesh.points, dtype=np.float32)
    labels = np.clip(np.asarray(mesh.point_data["point_label"], dtype=np.int64), 0, 2)
    region = np.zeros((points.shape[0], 3), dtype=np.float32)
    region[np.arange(points.shape[0]), labels] = 1.0

    graph = Data(
        pos=torch.from_numpy(points),
        region=torch.from_numpy(region),
        edge_index=build_edge_index(tetra, points.shape[0]),
    )
    graph = build_graph_input(graph, stats.coord_mean, stats.coord_std)

    # Outer boundary of the tetrahedral mesh, renumbered back onto the full
    # node list so that predicted displacement can be applied to it directly.
    boundary = mesh.extract_surface(pass_pointid=True).triangulate()
    original_ids = np.asarray(boundary.point_data["vtkOriginalPointIds"], dtype=np.int64)
    surface_faces = original_ids[boundary.faces.reshape(-1, 4)[:, 1:]]

    return Mesh(
        case_name=case_dir.name,
        stiff=stiff,
        tmax=tmax,
        graph=graph,
        points_mm=points * float(mm_per_unit),
        tetra=tetra,
        surface_faces=surface_faces,
    )


@dataclass
class PressureSurface:
    """The predicted response of one ventricle: P and q as functions of V and t."""

    volumes_ml: np.ndarray         # [V]
    times_ms: np.ndarray           # [T]
    pressure_mmHg: np.ndarray      # [V, T]
    coordinates: np.ndarray        # [V, T, K+1]  generalized coordinates q
    modes: np.ndarray              # [N, K+1, 3]  spatial modes phi, mesh only
    relaxation_ms: np.ndarray      # [V]  peak-to-end duration, per volume
    peak_time_ms: float
    cycle_duration_ms: float


@torch.no_grad()
def predict_surface(
    model,
    stats,
    mesh: Mesh,
    volumes_ml: np.ndarray,
    times_ms: np.ndarray,
    pressure_scale: float,
    device: torch.device,
) -> PressureSurface:
    """One forward pass per sampled volume, over a whole cycle each time."""
    graph_batch = Batch.from_data_list([mesh.graph]).to(device)
    time_values = torch.from_numpy(times_ms.astype(np.float32)).unsqueeze(0).to(device)

    pressure_rows: List[np.ndarray] = []
    coordinate_rows: List[np.ndarray] = []
    modes = None
    for volume in volumes_ml:
        dynamic = build_dynamic_features(
            np.full(times_ms.shape, float(volume)), mesh.stiff, mesh.tmax, stats
        )
        outputs = model(graph_batch, torch.from_numpy(dynamic).unsqueeze(0).to(device), time_values)
        pressure_rows.append(outputs["pressure"][0].cpu().numpy().astype(np.float64))
        coordinate_rows.append(outputs["q"][0].cpu().numpy().astype(np.float64))
        if modes is None:
            # phi is a function of the mesh alone, so the first pass settles it.
            modes = outputs["phi"].cpu().numpy().astype(np.float64)

    pressure = regularize_surface(volumes_ml, np.asarray(pressure_rows))
    coordinates = np.asarray(coordinate_rows)
    # q_0 is the pressure coordinate by construction, so it has to follow the
    # regularized surface rather than the raw pass it came from.
    coordinates[:, :, 0] = pressure / float(pressure_scale)

    peak_times = times_ms[np.argmax(pressure, axis=1)]
    return PressureSurface(
        volumes_ml=np.asarray(volumes_ml, dtype=float),
        times_ms=np.asarray(times_ms, dtype=float),
        pressure_mmHg=pressure,
        coordinates=coordinates,
        modes=modes,
        relaxation_ms=np.maximum(times_ms[-1] - peak_times, 1e-6),
        peak_time_ms=float(np.median(peak_times)),
        cycle_duration_ms=float(times_ms[-1]),
    )


def regularize_surface(volumes_ml: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """Make the sampled surface usable as a constitutive law for the circuit.

    The rows are independent forward passes, so nothing in the network forces
    the surface to be smooth along the volume axis or to rise with volume.  The
    circuit integrator needs both: a ventricle whose pressure dips as it fills
    would let the solver run away.  A short Savitzky-Golay pass removes
    row-to-row noise across volume, isotonic regression removes any remaining
    non-physical dip, and a second pass along time removes ripple.
    """
    smoothed = np.maximum(np.asarray(pressure, dtype=float), 0.0)
    n_volumes = smoothed.shape[0]

    window = min(7, n_volumes if n_volumes % 2 else n_volumes - 1)
    polyorder = min(3, window - 1)
    isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip")
    for step in range(smoothed.shape[1]):
        column = smoothed[:, step]
        if window >= 3 and polyorder >= 1:
            column = savgol_filter(column, window_length=window, polyorder=polyorder, mode="interp")
        smoothed[:, step] = isotonic.fit_transform(volumes_ml, np.maximum(column, 0.0))

    smoothed = savgol_filter(smoothed, window_length=15, polyorder=3, axis=1, mode="wrap")
    return np.maximum(smoothed, 0.0)


# --------------------------------------------------------------------------- #
# The circulation: a lumped-parameter circuit closed around the ventricle
#
#            venous return                systemic bed
#      +--------- Rv <--------- Cv <--------- Rs <---------+
#      |                                                   |
#      v                                                   |
#   left atrium (Ca) --- mitral valve (Rm) ---> LEFT VENTRICLE
#                                                    |
#                                          aortic valve (Ra)
#                                                    |
#                                                    v
#                                          arterial compliance (Ca_art)
#
# The ventricle contributes no equation of its own: its pressure is read off
# the predicted surface at the current volume and cardiac phase.  Both valves
# are ideal diodes with a linear resistance and a small opening offset, so flow
# is one-way and the four phases of the cycle appear on their own.
#
# The values below are the operating point used for the CGFENet-v2 haemodynamic
# results: arterial compliance 1.5 mL/mmHg, systemic resistance 1785 (dyn-like
# units consistent with mL/ms and mmHg), everything else at its baseline.
# --------------------------------------------------------------------------- #

@dataclass
class Circulation:
    atrial_compliance: float = 5.0            # mL/mmHg
    arterial_compliance: float = 1.5          # mL/mmHg
    venous_compliance: float = 5.0            # mL/mmHg
    mitral_resistance: float = 18.75159388548
    aortic_resistance: float = 41.253506548057
    systemic_resistance: float = 1785.1517378977214
    venous_resistance: float = 15.001275108384
    valve_opening_offset: float = 0.1         # mmHg a valve needs before it opens

    initial_ventricular_volume: float = 130.0  # mL, clipped into the sampled range
    initial_atrial_pressure: float = 10.5      # mmHg
    initial_arterial_pressure: float = 80.0    # mmHg
    initial_venous_pressure: float = 10.0      # mmHg


class SurfaceInterpolator:
    """Ventricular pressure at an arbitrary volume and cardiac phase.

    Pressure is shape-preserving (PCHIP) along volume, so interpolation cannot
    introduce the overshoot that a cubic spline would, and linear along time.
    Past its own relaxation the ventricle stays at end-relaxation pressure
    rather than being read off the end of the sampled window, which is what
    `relaxation_ms` records per volume.
    """

    def __init__(self, surface: PressureSurface) -> None:
        self.volumes = surface.volumes_ml
        self.times = surface.times_ms
        self.relaxation = surface.relaxation_ms
        self.peak_time_ms = surface.peak_time_ms
        self._by_time = [PchipInterpolator(self.volumes, surface.pressure_mmHg[:, step],
                                           extrapolate=False)
                         for step in range(surface.pressure_mmHg.shape[1])]

    def effective_time(self, volume_ml: float, cycle_time_ms: float) -> float:
        relaxation = float(np.interp(volume_ml, self.volumes, self.relaxation))
        return float(np.clip(min(cycle_time_ms, self.peak_time_ms + relaxation),
                             self.times[0], self.times[-1]))

    def pressure(self, volume_ml: float, effective_time_ms: float) -> float:
        volume = float(np.clip(volume_ml, self.volumes[0], self.volumes[-1]))
        upper = int(np.clip(np.searchsorted(self.times, effective_time_ms, side="right"),
                            1, len(self.times) - 1))
        lower = upper - 1
        beta = ((effective_time_ms - self.times[lower])
                / max(self.times[upper] - self.times[lower], 1e-12))
        blended = (1.0 - beta) * float(self._by_time[lower](volume)) \
            + beta * float(self._by_time[upper](volume))
        return max(blended, 0.0)


def simulate_circulation(
    surface: PressureSurface,
    circulation: Circulation,
    time_step_ms: float,
    cycles: int,
) -> Dict[str, np.ndarray]:
    """Explicit Euler integration of the four compartment volumes."""
    interpolator = SurfaceInterpolator(surface)
    period = surface.cycle_duration_ms
    steps = int(round(period * cycles / time_step_ms)) + 1
    time_ms = np.linspace(0.0, period * cycles, steps)
    cycle_time_ms = np.mod(time_ms, period)

    ventricular = np.zeros(steps)
    atrial = np.zeros(steps)
    arterial = np.zeros(steps)
    venous = np.zeros(steps)
    ventricular[0] = float(np.clip(circulation.initial_ventricular_volume,
                                   surface.volumes_ml[0], surface.volumes_ml[-1]))
    atrial[0] = circulation.atrial_compliance * circulation.initial_atrial_pressure
    arterial[0] = circulation.arterial_compliance * circulation.initial_arterial_pressure
    venous[0] = circulation.venous_compliance * circulation.initial_venous_pressure

    p_ventricular = np.zeros(steps)
    p_atrial = np.zeros(steps)
    p_arterial = np.zeros(steps)
    p_venous = np.zeros(steps)
    mitral_flow = np.zeros(steps)
    aortic_flow = np.zeros(steps)

    for i in range(steps):
        effective = interpolator.effective_time(ventricular[i], cycle_time_ms[i])
        p_ventricular[i] = interpolator.pressure(ventricular[i], effective)
        p_atrial[i] = atrial[i] / circulation.atrial_compliance
        p_arterial[i] = arterial[i] / circulation.arterial_compliance
        p_venous[i] = venous[i] / circulation.venous_compliance

        offset = circulation.valve_opening_offset
        mitral_flow[i] = max((p_atrial[i] - p_ventricular[i] - offset)
                             / circulation.mitral_resistance, 0.0)
        aortic_flow[i] = max((p_ventricular[i] - p_arterial[i] - offset)
                             / circulation.aortic_resistance, 0.0)
        venous_flow = max((p_venous[i] - p_atrial[i]) / circulation.venous_resistance, 0.0)
        systemic_flow = (p_arterial[i] - p_venous[i]) / circulation.systemic_resistance

        if i + 1 == steps:
            break
        dt = time_step_ms
        ventricular[i + 1] = np.clip(ventricular[i] + dt * (mitral_flow[i] - aortic_flow[i]),
                                     surface.volumes_ml[0], surface.volumes_ml[-1])
        atrial[i + 1] = max(atrial[i] + dt * (venous_flow - mitral_flow[i]), 1e-6)
        arterial[i + 1] = max(arterial[i] + dt * (aortic_flow[i] - systemic_flow), 1e-6)
        venous[i + 1] = max(venous[i] + dt * (systemic_flow - venous_flow), 1e-6)

    return {
        "time_ms": time_ms,
        "cycle_time_ms": cycle_time_ms,
        "volume_ml": ventricular,
        "pressure_mmHg": p_ventricular,
        "atrial_pressure_mmHg": p_atrial,
        "arterial_pressure_mmHg": p_arterial,
        "venous_pressure_mmHg": p_venous,
        "mitral_flow": mitral_flow,
        "aortic_flow": aortic_flow,
    }


def last_cycle(history: Dict[str, np.ndarray], period_ms: float, cycles: int) -> Dict[str, np.ndarray]:
    """The converged cycle: the circuit has settled by the time it is reached."""
    mask = history["time_ms"] >= period_ms * (cycles - 1) - 1e-9
    loop = {key: values[mask].copy() for key, values in history.items()}
    loop["time_ms"] = loop["time_ms"] - loop["time_ms"][0]
    return loop


def loop_summary(loop: Dict[str, np.ndarray]) -> Dict[str, float]:
    end_diastolic = float(loop["volume_ml"].max())
    end_systolic = float(loop["volume_ml"].min())
    stroke = end_diastolic - end_systolic
    return {
        "end_diastolic_volume_ml": end_diastolic,
        "end_systolic_volume_ml": end_systolic,
        "stroke_volume_ml": stroke,
        "ejection_fraction_percent": 100.0 * stroke / max(end_diastolic, 1e-9),
        "peak_pressure_mmHg": float(loop["pressure_mmHg"].max()),
        "end_diastolic_pressure_mmHg": float(
            loop["pressure_mmHg"][int(np.argmax(loop["volume_ml"]))]),
    }


# --------------------------------------------------------------------------- #
# The wall motion that belongs to the loop
# --------------------------------------------------------------------------- #

def displacement_along_loop(
    surface: PressureSurface,
    loop: Dict[str, np.ndarray],
    frames: Sequence[int],
    coord_std: np.ndarray,
    mm_per_unit: float,
) -> np.ndarray:
    """u(n, t) = sum_k q_k(V(t), t) phi_k(n), evaluated on the loop's own path.

    Because the modes are fixed by the mesh, following the loop costs one
    interpolation of the seven generalized coordinates per frame -- the spatial
    field never has to be predicted again.
    """
    interpolator = SurfaceInterpolator(surface)
    query = np.asarray([
        [np.clip(loop["volume_ml"][i], surface.volumes_ml[0], surface.volumes_ml[-1]),
         interpolator.effective_time(loop["volume_ml"][i], loop["cycle_time_ms"][i])]
        for i in frames
    ])
    coordinates = RegularGridInterpolator(
        (surface.volumes_ml, surface.times_ms), surface.coordinates,
        bounds_error=False, fill_value=None,
    )(query)                                                     # [F, K+1]
    displacement = np.einsum("fk,nkd->nfd", coordinates, surface.modes)
    return displacement * coord_std * float(mm_per_unit)


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #

def write_loop_table(out_path: Path, case_name: str, loop: Dict[str, np.ndarray]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "t_ms", "volume_ml", "lv_pressure_mmHg",
                         "arterial_pressure_mmHg", "atrial_pressure_mmHg"])
        for i in range(len(loop["time_ms"])):
            writer.writerow([
                case_name,
                f"{loop['time_ms'][i]:.2f}",
                f"{loop['volume_ml'][i]:.5f}",
                f"{loop['pressure_mmHg'][i]:.5f}",
                f"{loop['arterial_pressure_mmHg'][i]:.5f}",
                f"{loop['atrial_pressure_mmHg'][i]:.5f}",
            ])


def plot_pv_loop(out_path: Path, case_name: str, loop: Dict[str, np.ndarray],
                 summary: Dict[str, float]) -> None:
    volume = loop["volume_ml"]
    pressure = loop["pressure_mmHg"]

    fig, ax = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
    ax.plot(volume, pressure, color=net.NATURE_VERMILLION, linewidth=2.2)

    # A single arrow fixes the direction of travel: filling, then ejection.
    mid = len(volume) // 5
    ax.annotate("", xy=(volume[mid + 1], pressure[mid + 1]), xytext=(volume[mid], pressure[mid]),
                arrowprops=dict(arrowstyle="-|>", color=net.NATURE_VERMILLION, lw=2.2))

    end_diastolic = int(np.argmax(volume))
    end_systolic = int(np.argmin(volume))
    for index, label, offset in ((end_diastolic, "ED", (-26, -14)), (end_systolic, "ES", (10, 8))):
        ax.scatter([volume[index]], [pressure[index]], s=46, zorder=5,
                   color=net.NATURE_GREY, edgecolor="white", linewidth=0.8)
        ax.annotate(label, (volume[index], pressure[index]), textcoords="offset points",
                    xytext=offset, fontsize=10, color=net.NATURE_GREY)

    ax.set_xlabel("Left-ventricular volume (mL)")
    ax.set_ylabel("Left-ventricular pressure (mmHg)")
    ax.set_title(f"{case_name}: predicted pressure-volume loop", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.10, y=0.06)
    ax.set_ylim(bottom=min(0.0, float(pressure.min())))
    # Inside the loop, where nothing is drawn.
    ax.text(
        0.52, 0.40,
        f"EDV {summary['end_diastolic_volume_ml']:.1f} mL\n"
        f"ESV {summary['end_systolic_volume_ml']:.1f} mL\n"
        f"SV {summary['stroke_volume_ml']:.1f} mL\n"
        f"EF {summary['ejection_fraction_percent']:.1f} %\n"
        f"peak {summary['peak_pressure_mmHg']:.1f} mmHg",
        transform=ax.transAxes, ha="center", va="center", fontsize=9.5,
        linespacing=1.6, color=net.NATURE_GREY,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)


def write_vtk_series(
    out_dir: Path,
    mesh: Mesh,
    loop: Dict[str, np.ndarray],
    displacement_mm: np.ndarray,
    frames: Sequence[int],
) -> Path:
    """One .vtu per frame, plus a .pvd index carrying the physical times.

    The written points are the deformed configuration, so the series animates
    straight away; the reference coordinates travel along as point data, and
    the haemodynamic state of the frame as field data.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = np.hstack([np.full((mesh.tetra.shape[0], 1), 4, dtype=np.int64), mesh.tetra]).ravel()
    cell_types = np.full(mesh.tetra.shape[0], pv.CellType.TETRA, dtype=np.uint8)

    entries: List[str] = []
    for frame_number, step in enumerate(frames):
        grid = pv.UnstructuredGrid(cells, cell_types, mesh.points_mm + displacement_mm[:, frame_number, :])
        grid.point_data["reference_coordinates_mm"] = mesh.points_mm
        grid.point_data["displacement_mm"] = displacement_mm[:, frame_number, :]
        grid.point_data["displacement_magnitude_mm"] = np.linalg.norm(
            displacement_mm[:, frame_number, :], axis=1)
        for name, key in (("time_ms", "time_ms"),
                          ("volume_ml", "volume_ml"),
                          ("lv_pressure_mmHg", "pressure_mmHg"),
                          ("arterial_pressure_mmHg", "arterial_pressure_mmHg")):
            grid.field_data[name] = np.asarray([loop[key][step]], dtype=np.float64)

        name = f"cycle_{frame_number:04d}.vtu"
        grid.save(out_dir / name)
        entries.append(f'    <DataSet timestep="{loop["time_ms"][step]:.2f}" file="{name}"/>')

    index_path = out_dir / f"{mesh.case_name}.pvd"
    index_path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        '  <Collection>\n' + "\n".join(entries) + "\n"
        "  </Collection>\n"
        "</VTKFile>\n",
        encoding="utf-8",
    )
    return index_path


def render_movie(
    out_path: Path,
    mesh: Mesh,
    loop: Dict[str, np.ndarray],
    displacement_mm: np.ndarray,
    frames: Sequence[int],
    fps: int,
) -> None:
    """The beating ventricle beside the loop it is tracing out."""
    import imageio.v2 as imageio

    faces = mesh.surface_faces
    deformed = mesh.points_mm[None, :, :] + np.transpose(displacement_mm, (1, 0, 2))  # [F, N, 3]
    magnitude = np.linalg.norm(displacement_mm, axis=-1)                              # [N, F]
    lo, hi = deformed.reshape(-1, 3).min(axis=0), deformed.reshape(-1, 3).max(axis=0)
    centre, radius = (lo + hi) / 2.0, float((hi - lo).max()) / 2.0 * 1.02
    norm = colors.Normalize(vmin=0.0, vmax=float(magnitude.max()))
    cmap = matplotlib.colormaps["magma"]

    # 1280 x 560 px: both dimensions are multiples of 16, as the encoder prefers.
    # Axes are placed explicitly so that nothing shifts between frames.
    fig = plt.figure(figsize=(12.8, 5.6), dpi=100)
    ax_mesh = fig.add_axes([0.005, 0.04, 0.44, 0.84], projection="3d")
    ax_bar = fig.add_axes([0.455, 0.18, 0.011, 0.62])
    ax_loop = fig.add_axes([0.605, 0.13, 0.365, 0.76])
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_bar,
                 label="Displacement magnitude (mm)")
    fig.text(0.5, 0.96, f"{mesh.case_name}: predicted cardiac cycle",
             ha="center", fontsize=12.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8,
                            macro_block_size=1) as writer:
        for frame_number, step in enumerate(frames):
            ax_mesh.clear()
            ax_mesh.add_collection3d(Poly3DCollection(
                deformed[frame_number][faces],
                facecolors=cmap(norm(magnitude[faces, frame_number].mean(axis=1))),
                edgecolor=(0.15, 0.15, 0.15, 0.25), linewidths=0.18))
            ax_mesh.set_xlim(centre[0] - radius, centre[0] + radius)
            ax_mesh.set_ylim(centre[1] - radius, centre[1] + radius)
            ax_mesh.set_zlim(centre[2] - radius, centre[2] + radius)
            ax_mesh.set_box_aspect((1, 1, 1))
            ax_mesh.set_xlabel("x (mm)", labelpad=-6)
            ax_mesh.set_ylabel("y (mm)", labelpad=-6)
            ax_mesh.set_zticklabels([])
            ax_mesh.tick_params(axis="both", pad=-2, labelsize=7)
            ax_mesh.view_init(elev=18.0, azim=-62.0)
            ax_mesh.grid(True, alpha=0.16)
            for axis in (ax_mesh.xaxis, ax_mesh.yaxis, ax_mesh.zaxis):
                axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
                axis.pane.set_edgecolor((0.80, 0.80, 0.80, 0.40))

            ax_loop.clear()
            ax_loop.plot(loop["volume_ml"], loop["pressure_mmHg"],
                         color=net.NATURE_VERMILLION, linewidth=2.0)
            ax_loop.scatter([loop["volume_ml"][step]], [loop["pressure_mmHg"][step]],
                            color=net.NATURE_VERMILLION, edgecolor="white",
                            linewidth=0.8, s=70, zorder=5)
            ax_loop.set_xlabel("Volume (mL)")
            ax_loop.set_ylabel("Pressure (mmHg)")
            ax_loop.grid(True, alpha=0.25)
            ax_loop.set_title(
                f"t = {loop['time_ms'][step]:.0f} ms    "
                f"V = {loop['volume_ml'][step]:.1f} mL    "
                f"P = {loop['pressure_mmHg'][step]:.1f} mmHg", fontsize=10.5)

            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = net.CGFENet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    stats = net.NormalizationStats.from_dict(checkpoint["normalization"])
    mm_per_unit = MM_PER_UNIT
    pressure_scale = float(checkpoint["model_config"]["pressure_scale"])

    mesh = read_mesh(args.case_dir, stats, mm_per_unit, args.stiffness, args.tmax)
    # "test" is the split dictionary's own key for the group train.py holds out
    # and never reads; it is recorded in the checkpoint for exactly this check.
    held_out = checkpoint.get("split", {}).get("test", [])
    if mesh.case_name.split("_")[0] not in held_out:
        raise RuntimeError(
            f"Geometry {mesh.case_name.split('_')[0]} was not held out by this checkpoint")
    print(f"Mesh {args.case_dir / MESH_FILE_NAME}: {mesh.points_mm.shape[0]} nodes, "
          f"{mesh.tetra.shape[0]} tetrahedra")
    print(f"Physiology: passive stiffness C = {mesh.stiff:g}, Tmax = {mesh.tmax:g}k")
    print(f"Checkpoint {args.checkpoint} (epoch {checkpoint['epoch']}), geometry held out\n")

    volumes = np.linspace(args.volume_min, args.volume_max, args.volume_samples)
    times = np.arange(args.time_samples, dtype=np.float64)
    print(f"Predicting the pressure surface over {len(volumes)} volumes "
          f"({volumes[0]:.0f}-{volumes[-1]:.0f} mL) x {len(times)} ms ...")
    surface = predict_surface(model, stats, mesh, volumes, times, pressure_scale, device)

    print(f"Coupling to the circulation for {args.cycles} cycles "
          f"at {args.time_step:g} ms ...")
    history = simulate_circulation(surface, Circulation(), args.time_step, args.cycles)
    loop = last_cycle(history, surface.cycle_duration_ms, args.cycles)
    summary = loop_summary(loop)

    stride = max(1, len(loop["time_ms"]) // max(1, args.movie_frames))
    frames = list(range(0, len(loop["time_ms"]), stride))
    displacement_mm = displacement_along_loop(
        surface, loop, frames, safe_coord_std(stats.coord_std), mm_per_unit)

    pressure_dir = args.output_dir / "pressure"
    disp_dir = args.output_dir / "disp"
    write_loop_table(pressure_dir / "pv_loop.csv", mesh.case_name, loop)
    plot_pv_loop(pressure_dir / "pv_loop.png", mesh.case_name, loop, summary)
    index_path = write_vtk_series(disp_dir / "vtk", mesh, loop, displacement_mm, frames)
    render_movie(disp_dir / "displacement_cycle.mp4", mesh, loop, displacement_mm,
                 frames, args.movie_fps)

    print("\nPredicted haemodynamics")
    for label, key, unit in (
        ("End-diastolic volume", "end_diastolic_volume_ml", "mL"),
        ("End-systolic volume", "end_systolic_volume_ml", "mL"),
        ("Stroke volume", "stroke_volume_ml", "mL"),
        ("Ejection fraction", "ejection_fraction_percent", "%"),
        ("Peak LV pressure", "peak_pressure_mmHg", "mmHg"),
        ("End-diastolic pressure", "end_diastolic_pressure_mmHg", "mmHg"),
    ):
        print(f"  {label:<24}{summary[key]:8.1f} {unit}")
    print(f"  {'Peak wall displacement':<24}"
          f"{np.linalg.norm(displacement_mm, axis=-1).max():8.1f} mm")

    print(f"\nWrote {pressure_dir / 'pv_loop.csv'}")
    print(f"Wrote {pressure_dir / 'pv_loop.png'}")
    print(f"Wrote {index_path} and {len(frames)} .vtu frames")
    print(f"Wrote {disp_dir / 'displacement_cycle.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
