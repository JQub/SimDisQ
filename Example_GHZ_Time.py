# =========================
# Python path setup
# =========================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import io
import contextlib
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# =========================
# Distributed Quantum Computing Simulator + Timing
# =========================
from dqc_simulator import (
    DQCCircuit, DQCQPU, QPUManager,
    DQCTimingProvider, QiskitTimingProvider, NetworkTimingProvider,
    DecoherenceModel, fiber_latency,
    DEFAULT_GATE_TIMES, DEFAULT_EPR_TIME, DEFAULT_CLASSICAL_LATENCY,
)

# All durations are in seconds.
US = 1e6   # seconds -> us
MS = 1e3   # seconds -> ms


# ============================================================
# Helpers
# ============================================================
def ghz_circuit(numbits):
    qc0 = QuantumCircuit(numbits, numbits)
    qc0.h(0)
    for i in range(numbits - 1):
        qc0.cx(i, i + 1)
    for i in range(numbits):
        qc0.measure(i, i)
    return qc0


def score(counts: dict, n: int) -> float:
    """Hellinger fidelity to the ideal GHZ distribution."""
    total = sum(counts.values())
    return sum(np.sqrt(0.5 * counts.get(k, 0) / total) for k in ('0' * n, '1' * n))


def build_qpu_group(backend_name, num_qpus, timing=None, **link_kwargs):
    """Line topology 0 - 1 - ... - (num_qpus-1)."""
    group = QPUManager(timing=timing)
    for i in range(num_qpus):
        group.add_qpu(DQCQPU(i, backend_name))
    for i in range(num_qpus - 1):
        group.add_coonnection(i, i + 1, distance=5, **link_kwargs)
    return group


def simulate(qc, result_qc, numbits, shots=4000):
    sim = AerSimulator(noise_model=qc.get_noise_model())
    counts = sim.run(transpile(result_qc, sim), shots=shots).result().get_counts()
    counts_res = {}
    for bitstring, cnt in counts.items():
        bits = bitstring[:numbits]
        counts_res[bits] = counts_res.get(bits, 0) + cnt
    return score(counts_res, numbits)


def run_ghz(backend_name, numbits=12, partition=(3, 3, 3, 3), timing=None, idle_noise=False, quiet=True):
    """Full pipeline: partition -> execution -> scheduling -> (idle noise) -> simulation."""
    qc = DQCCircuit(ghz_circuit(numbits))
    group = build_qpu_group(backend_name, len(partition), timing=timing)
    log = io.StringIO()
    with contextlib.redirect_stdout(log) if quiet else contextlib.nullcontext():
        result_qc = qc.Execution(list(partition), group, comm_noise=True, idle_noise=idle_noise)
    return qc, simulate(qc, result_qc, numbits)


numbits = 12
Partition = [3, 3, 3, 3]


# ============================================================
# Step 1: Default timing interface
# ============================================================
print("=" * 60)
print("Step 1: Default timing interface")
print("=" * 60)
print("DEFAULT_GATE_TIMES        :", DEFAULT_GATE_TIMES)
print("DEFAULT_EPR_TIME          :", DEFAULT_EPR_TIME, "s")
print("DEFAULT_CLASSICAL_LATENCY :", DEFAULT_CLASSICAL_LATENCY, "s")

# QPUManager() builds DQCTimingProvider(QiskitTimingProvider, NetworkTimingProvider) by default.
QPUGROUP = build_qpu_group("FakeVigoV2", 4)

# Local gate times come from each QPU's Qiskit Target (same gate, different qubits -> different time).
print(f"CX(0,1) on QPU0   : {QPUGROUP.get_gate_time(0, 'cx', (0, 1)) * US:.4f} us")
print(f"CX(1,2) on QPU0   : {QPUGROUP.get_gate_time(0, 'cx', (1, 2)) * US:.4f} us")
print(f"Measure(0) on QPU0: {QPUGROUP.get_gate_time(0, 'measure', (0,)) * US:.4f} us")
# Not calibrated in FakeVigoV2 -> falls back to DEFAULT_GATE_TIMES.
print(f"Reset(0) on QPU0  : {QPUGROUP.get_gate_time(0, 'reset', (0,)) * US:.4f} us (default)")

# Network times come from the SimDisQ network model.
print(f"EPR QPU0<->QPU1   : {QPUGROUP.get_epr_time(0, 1) * MS:.3f} ms generation + "
      f"{QPUGROUP.get_epr_transmission_time(0, 1) * US:.3f} us transmission (5 km fiber)")
print(f"Classical QPU0->1 : {QPUGROUP.get_classical_latency(0, 1) * US:.3f} us")

# Unified entry point: get_operation_duration(operation, resource, context)
print("Unified API       :",
      QPUGROUP.get_operation_duration("cx", (0, 1), context=0),
      QPUGROUP.get_operation_duration("epr_generation", (0, 1)),
      QPUGROUP.get_operation_duration("classical_communication", (0, 1)))


# ============================================================
# Step 2: Custom timing (user gate times, per-link values, models, defaults)
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Custom timing")
print("=" * 60)

timing = DQCTimingProvider(
    QiskitTimingProvider(
        # User gate times override the backend (native gate names after transpilation).
        gate_times={
            "x": 10e-6,                 # X on every QPU
            "rz": 2e-6,                 # Z / RZ (no longer a virtual gate)
            (1, "cx"): 500e-9,          # CX on QPU 1 only
            (1, "cx", (0, 1)): 450e-9,  # CX(0,1) on QPU 1 only
        },
        # Used only when the backend has no calibration.
        default_times={"reset": 2e-6},
    ),
    NetworkTimingProvider(
        # T_EPR = f(distance): 1 ms base + 0.2 ms per unit distance.
        epr_model=lambda src, dst, dist: 1e-3 + 0.2e-3 * (dist or 0),
        # T_classical = T_propagation + T_processing over fiber (distance in km).
        classical_model=lambda src, dst, dist: fiber_latency(dist or 0, processing=5e-6),
    ),
)
CUSTOM = QPUManager(timing=timing)
for i in range(4):
    CUSTOM.add_qpu(DQCQPU(i, "FakeVigoV2"))
CUSTOM.add_coonnection(0, 1, distance=5)                     # from the models
CUSTOM.add_coonnection(1, 2, distance=5, epr_time=5e-3,      # per-link EPR overrides
                       epr_transmission_time=100e-6)
CUSTOM.add_coonnection(2, 3, distance=10, classical_latency=50e-6)  # per-link classical override

for a, b in [(0, 1), (1, 2), (2, 3)]:
    print(f"Link {a}-{b}: EPR {CUSTOM.get_epr_time(a, b) * MS:.3f} ms + "
          f"{CUSTOM.get_epr_transmission_time(a, b) * US:.3f} us transmission, "
          f"classical {CUSTOM.get_classical_latency(a, b) * US:.3f} us")
print(f"X(0) on QPU0      : {CUSTOM.get_gate_time(0, 'x', (0,)) * US:.3f} us (user, backend has 0.036 us)")
print(f"RZ(0) on QPU0     : {CUSTOM.get_gate_time(0, 'rz', (0,)) * US:.3f} us (user, backend has 0)")
print(f"SX(0) on QPU0     : {CUSTOM.get_gate_time(0, 'sx', (0,)) * US:.3f} us (backend)")
print(f"CX(0,1) QPU0/QPU1 : {CUSTOM.get_gate_time(0, 'cx', (0, 1)) * US:.3f} / "
      f"{CUSTOM.get_gate_time(1, 'cx', (0, 1)) * US:.3f} us")
print(f"Reset(0) on QPU0  : {CUSTOM.get_gate_time(0, 'reset', (0,)) * US:.3f} us (custom default)")


# ============================================================
# Step 3: Scheduling -> timeline
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Scheduling")
print("=" * 60)

qc = DQCCircuit(ghz_circuit(numbits))
result_qc = qc.Execution(Partition, CUSTOM, comm_noise=True)   # estimate_time=True by default

timeline = qc.timeline
timeline.print_summary()

# Timing statistics: critical-path decomposition, per gate / link / channel / QPU.
stats = timeline.statistics()
stats.report()
stats.to_json("GHZ_12qubit_DQC_timing.json")

print("Network events:")
for ev in timeline.events:
    if ev.kind == "epr":
        print(f"  epr       QPU {ev.qpus}  {ev.start * MS:8.4f} -> {ev.end * MS:8.4f} ms  "
              f"(generation {ev.segment('epr_generation') * MS:.3f} ms, "
              f"transmission {ev.segment('epr_transmission') * US:.3f} us, wait {ev.wait * US:.3f} us)")
    elif ev.kind == "classical":
        print(f"  classical QPU {ev.qpus}  {ev.start * MS:8.4f} -> {ev.end * MS:8.4f} ms")

# Waiting time is an output of the scheduler: idle gaps of each qubit.
q_worst = max(timeline.qubit_idle, key=timeline.qubit_idle.get)
print(f"Most idle qubit: q{q_worst}, idle {timeline.qubit_idle[q_worst] * MS:.3f} ms, "
      f"intervals {[(round(s * MS, 4), round(e * MS, 4)) for s, e in timeline.idle_intervals[q_worst]]}")

# The schedule can be recomputed with any other TimingProvider without re-compiling.
fast_net = DQCTimingProvider(network=NetworkTimingProvider(default_epr_time=0.1e-3))
print(f"Same circuit with 0.1 ms EPR: {qc.schedule(fast_net).makespan * MS:.3f} ms")
qc.schedule()   # restore the timeline of CUSTOM


# --- 3.1 Timeline plot (one row per QPU) ---
colors = {"gate": "tab:blue", "measure": "tab:green", "reset": "tab:gray",
          "feedforward": "tab:purple", "epr": "tab:orange", "classical": "tab:red"}
fig1, ax = plt.subplots(figsize=(10, 3.5))
for ev in timeline.events:
    rows = ev.qpus if ev.kind in ("epr", "classical") else ev.qpus[:1]
    for qpu in rows:
        ax.broken_barh([(ev.start * MS, max(ev.duration * MS, 1e-3))], (qpu - 0.35, 0.7),
                       color=colors[ev.kind], alpha=0.6 if ev.kind in ("epr", "classical") else 1.0)
ax.set_yticks(range(len(Partition)))
ax.set_yticklabels([f"QPU {i}" for i in range(len(Partition))])
ax.set_xlabel("time (ms)")
ax.set_title("GHZ 12-qubit DQC timeline")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()],
          labels=list(colors), ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=8)
fig1.tight_layout()
fig1.savefig("GHZ_12qubit_DQC_timeline.pdf")


# ============================================================
# Step 4: Waiting time -> decoherence -> fidelity
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Idle decoherence (default 2 ms EPR)")
print("=" * 60)

for backend_name in ["FakeVigoV2", "IonQ"]:
    for label, idle in [("none", False),
                        ("thermal", True),
                        ("dephasing", DecoherenceModel("dephasing"))]:
        qc_i, fidelity = run_ghz(backend_name, idle_noise=idle)
        print(f"{backend_name:<11} idle noise = {label:<9}  T = {qc_i.timeline.makespan * MS:7.3f} ms  "
              f"F = {fidelity:.3f}")
# Superconducting QPUs (T2 ~ 10 us) decohere almost completely under ms-level network latency.
# Pure dephasing does not change Z-basis counts, so it is hidden from this GHZ fidelity.


# ============================================================
# Step 5: Fidelity vs EPR generation time
# ============================================================
print("\n" + "=" * 60)
print("Step 5: EPR time sweep (thermal idle noise)")
print("=" * 60)

epr_times = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
results = {}
for backend_name in ["FakeVigoV2", "IonQ"]:
    results[backend_name] = []
    for t_epr in epr_times:
        sweep_timing = DQCTimingProvider(network=NetworkTimingProvider(default_epr_time=t_epr))
        qc_s, fidelity = run_ghz(backend_name, timing=sweep_timing, idle_noise=True)
        results[backend_name].append(fidelity)
        print(f"{backend_name:<11} EPR = {t_epr * MS:8.1f} ms  T = {qc_s.timeline.makespan * MS:9.3f} ms  "
              f"F = {fidelity:.3f}")

fig2, ax = plt.subplots(figsize=(5, 3.5))
for backend_name, fids in results.items():
    ax.semilogx(np.array(epr_times) * MS, fids, marker="o", label=backend_name)
ax.set_xlabel("EPR generation time (ms)")
ax.set_ylabel("GHZ Hellinger fidelity")
ax.set_ylim(0, 1)
ax.legend()
fig2.tight_layout()
fig2.savefig("GHZ_12qubit_DQC_fidelity_vs_epr.pdf")

plt.show()
