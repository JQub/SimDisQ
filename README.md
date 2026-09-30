# Distributed Quantum Circuit 

This repository demonstrates the creation, partitioning, execution, and visualization of a distributed quantum circuit using **DQCCircuit** and QPU simulation. It also shows how to save measurement results and circuit diagrams as images.

---

## Example Environment

- **Python**: 3.13.X
- **Qiskit**
- **Qiskit Aer**
- **Qiskit IBM Runtime**
- **Matplotlib**
- **pylatexenc**

Install dependencies with pip:

```bash
# Make sure you are using Python 3.13.5
pip install qiskit qiskit-aer qiskit-ibm-runtime matplotlib pylatexenc
```

## Quick Start Guide

### 1. Define Circuit Size
Set the number of qubits for your quantum circuit:
```python
numbits = 12  # Total logical qubits
```

### 2. Create Global Quantum Circuit
Build your quantum algorithm (e.g., GHZ state preparation):
```python
qc0 = QuantumCircuit(numbits, numbits)
qc0.h(0)
for i in range(numbits - 1):
    qc0.cx(i, i + 1)
# Add measurements
for i in range(numbits):
    qc0.measure(i, i)
```

### 3. Convert to Distributed Circuit
Transform the standard circuit into a distributed format:
```python
qc = DQCCircuit(qc0)
```

### 4. Define Circuit Partition
Specify how to distribute qubits across QPUs:
```python
# Option 1: Equal distribution by count
Partition = [3, 3, 3, 3]  # 4 QPUs, 3 qubits each

# Option 2: Explicit qubit assignment
Partition = [[0,1,2], [3,4,5], [6,7,8], [9,10,11]]
```

### 5. Configure QPU Manager
Set up heterogeneous QPU group with network topology:
```python
QPUGROUP = QPUManager()

# Add QPUs with different backends
QPUGROUP.add_qpu(DQCQPU(0, "FakeVigoV2"))
QPUGROUP.add_qpu(DQCQPU(1, "FakeLagosV2"))
QPUGROUP.add_qpu(DQCQPU(2, "FakeAthensV2"))
QPUGROUP.add_qpu(DQCQPU(3, "FakeManilaV2"))

# Define inter-QPU connections
dis = 5  # Communication distance/cost
QPUGROUP.add_coonnection(0, 1, distance=dis)
QPUGROUP.add_coonnection(1, 2, distance=dis)
QPUGROUP.add_coonnection(2, 3, distance=dis)
```

### 6. Execute Distributed Circuit
Run with communication noise modeling:
```python
result_qc = qc.Execution(
    Partition,
    QPUGROUP,
    comm_noise=True  # Enable teleportation noise
)
```

### 7. Timing and Scheduling
All durations go through one timing interface (`TimingProvider`, seconds):

- Local gate / measure / reset times are read from each QPU's Qiskit `Target`
  (`QiskitTimingProvider`); missing values fall back to `DEFAULT_GATE_TIMES`.
- EPR generation and classical latency come from `NetworkTimingProvider`
  (defaults: 2 ms and 20 us per link, overridable per link).
- Waiting time is not an input: `Execution` schedules the merged circuit (ASAP,
  event based, with link contention) and stores the result in `qc.timeline`.

```python
QPUGROUP.add_coonnection(0, 1, distance=5, epr_time=5e-3, classical_latency=50e-6)

QPUGROUP.get_gate_time(0, "cx", (0, 1))                        # from Qiskit Target
QPUGROUP.get_operation_duration("epr_generation", (0, 1))      # from network model

result_qc = qc.Execution(Partition, QPUGROUP, comm_noise=True)
qc.timeline.print_summary()      # makespan, EPR / classical / wait breakdown
qc.timeline.idle_intervals       # {qubit: [(start, end), ...]} for decoherence models
```

Custom defaults or models:
```python
from dqc_simulator import DQCTimingProvider, QiskitTimingProvider, NetworkTimingProvider, fiber_latency

timing = DQCTimingProvider(
    QiskitTimingProvider(default_times={"reset": 2e-6}),
    NetworkTimingProvider(
        default_epr_time=1e-3,
        classical_model=lambda a, b, dist: fiber_latency(dist, processing=5e-6),
    ),
)
QPUGROUP = QPUManager(timing=timing)
```

Idle decoherence (waiting time -> T1/T2 -> fidelity):
```python
from dqc_simulator import DecoherenceModel

# Insert thermal relaxation on every idle gap, T1/T2 from each QPU's Target
result_qc = qc.Execution(Partition, QPUGROUP, comm_noise=True, idle_noise=True)

# Or pure dephasing p_Z = (1 - exp(-dt/T_phi)) / 2
result_qc = qc.Execution(Partition, QPUGROUP, comm_noise=True,
                         idle_noise=DecoherenceModel("dephasing"))
```
Gate-time decoherence is already in `get_noise_model()`, so only idle gaps are added.
Note that pure dephasing is invisible to Z-basis counts; measure GHZ parity to observe it.

> **Superconducting QPUs decohere almost completely under ms-level network latency.**
> FakeV2 superconducting backends have T1 ≈ 100 µs and T2 ≈ 10–100 µs, while one EPR
> generation takes ~ms (default 2 ms). Qubits waiting for EPR pairs or classical
> messages therefore idle for many T2 periods. Trapped-ion QPUs (`IonQ`, T1 ≈ 10 s) are
> barely affected at the same latency.
>
> 12-qubit GHZ, 4 QPUs, 2 ms EPR, `comm_noise=True` (Hellinger fidelity):
>
> | Backend | no idle noise | `idle_noise=True` (thermal) |
> |---|---|---|
> | FakeVigoV2 (T1 ≈ 121 µs, T2 ≈ 17 µs) | 0.654 | 0.292 |
> | IonQ (T1 ≈ 10 s, T2 ≈ 1.5 s) | 0.699 | 0.699 |
>
> See `Example_GHZ_Time.py` for the full timing workflow and an EPR-time sweep.

### 8. Simulate and Analyze Results
Execute on AerSimulator with combined noise model:
```python
# Get combined noise model (local QPU + communication)
noise_model = qc.get_noise_model()
sim = AerSimulator(noise_model=noise_model)

# Transpile and run
compiled = transpile(result_qc, sim)
job = sim.run(compiled, shots=10_000)
result = job.result()
```

### 9. Visualize and Export
Generate publication-ready figures:
```python
# Measurement histogram
fig1 = plot_histogram(counts_res)
fig1.savefig("GHZ_12qubit_DQC_histogram.pdf")

# Circuit diagram
fig2 = result_qc.draw("mpl", scale=0.7, fold=100)
fig2.savefig("GHZ_12qubit_DQC_circuit.pdf")
```
