"""
Unified timing interface for SimDisQ.

    Qiskit backend / Target  ->  QiskitTimingProvider   (local gate, measure, reset)
    SimDisQ network config   ->  NetworkTimingProvider  (EPR generation + transmission, classical latency)
                                         |
                                  DQCTimingProvider      (unified API used by the scheduler)

All durations are in seconds (same convention as qiskit.transpiler.Target).
Waiting time is NOT a timing input: it is produced by the scheduler.
"""

import numpy as np


# =========================
# Defaults (seconds)
# =========================
# Used when a backend has no calibrated duration for an operation
# (e.g. FakeV2 backends report reset with no properties, or 'h' is not a basis gate).
DEFAULT_GATE_TIMES = {
    "1q": 50e-9,        # single-qubit gate
    "2q": 300e-9,       # two-qubit gate
    "measure": 1e-6,    # measurement
    "reset": 1e-6,      # reset
    "initialize": 1e-6, # local state preparation (treated like reset)
}

# Operations that take no time on hardware (virtual / compiler directives / noise channels).
ZERO_DURATION_OPS = {"barrier", "rz", "p", "u1", "kraus", "quantum_channel", "snapshot"}

DEFAULT_EPR_TIME = 2e-3             # 2 ms to generate one heralded EPR pair on one link
DEFAULT_CLASSICAL_LATENCY = 20e-6   # 20 us per classical message on one link
FIBER_SPEED = 2e8                   # m/s, light in optical fiber (distance is in km)

# Operation names accepted by get_operation_duration for network events.
EPR_OPS = {"epr", "epr_generation"}
EPR_TRANSMISSION_OPS = {"epr_transmission"}
CLASSICAL_OPS = {"classical", "classical_communication", "classical_latency"}

_DELAY_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}


def _op_name(operation):
    """Accept a gate name, a Qiskit Instruction or a CircuitInstruction."""
    if hasattr(operation, "operation"):
        operation = operation.operation
    name = operation if isinstance(operation, str) else getattr(operation, "name", str(operation))
    return name.lower()


def fiber_latency(distance_km, processing=0.0, speed=FIBER_SPEED):
    """Classical propagation latency over optical fiber (T_propagation + T_processing)."""
    return distance_km * 1e3 / speed + processing


# ============================================================
# Interface
# ============================================================
class TimingProvider:
    """
    Unified timing interface. The scheduler only talks to this class and does not care
    whether a duration comes from a Qiskit backend, real calibration data, or a network model.
    """

    def get_gate_time(self, qpu, gate, qubits):
        """Duration of a local operation `gate` on physical `qubits` of `qpu`."""
        raise NotImplementedError

    def get_epr_time(self, src_qpu, dst_qpu):
        """Time to generate one EPR pair between two directly linked QPUs."""
        raise NotImplementedError

    def get_epr_transmission_time(self, src_qpu, dst_qpu):
        """Time to deliver a generated EPR pair (photon propagation) to both QPUs."""
        return 0.0

    def get_classical_latency(self, src_qpu, dst_qpu):
        """Latency of one classical message from src_qpu to dst_qpu."""
        raise NotImplementedError

    def register_link(self, src_qpu, dst_qpu, distance=None, epr_time=None, classical_latency=None,
                      epr_transmission_time=None):
        """Hook called by QPUManager when a connection is added. Default: ignore."""
        pass

    def get_operation_duration(self, operation, resource, context=None):
        """
        Single entry point:
            get_operation_duration("cx", (0, 1), qpu)                  -> local gate time
            get_operation_duration("epr_generation", (qpu0, qpu1))     -> EPR generation time
            get_operation_duration("epr_transmission", (qpu0, qpu1))   -> EPR transmission time
            get_operation_duration("classical_communication", (a, b))  -> classical latency
        """
        name = _op_name(operation)
        if name in EPR_OPS:
            return self.get_epr_time(*resource)
        if name in EPR_TRANSMISSION_OPS:
            return self.get_epr_transmission_time(*resource)
        if name in CLASSICAL_OPS:
            return self.get_classical_latency(*resource)
        return self.get_gate_time(context, operation, resource)


# ============================================================
# Local QPU timing from Qiskit
# ============================================================
class QiskitTimingProvider(TimingProvider):
    """
    Read local gate / measure / reset durations from a Qiskit Target.

    Lookup order for (gate, qubits) on a QPU:
        1. gate_times set by the user (most specific key first), overrides everything
        2. target[gate][qubits].duration  (also tries reversed qubits for 2q gates)
        3. mean duration of that gate over all calibrated qubits of the target
        4. default_times by operation type

    Gate names are native (post-transpile) names, e.g. x, sx, rz, cx, measure, reset.
    """

    def __init__(self, gate_times=None, default_times=None, qpu_resolver=None):
        """
        :param gate_times: user durations that override the backend, keyed by
                           "x"                  -> every QPU
                           (qpu_id, "x")        -> one QPU
                           (qpu_id, "cx", (0,1))-> one QPU and qubits
        :param default_times: overrides for DEFAULT_GATE_TIMES (used only when nothing else is found)
        :param qpu_resolver: optional callable qpu_id -> DQCQPU, lets callers pass plain ids
        """
        self.gate_times = {}
        self._cache = {}
        for key, duration in (gate_times or {}).items():
            if isinstance(key, str):
                self.set_gate_time(key, duration)
            else:
                self.set_gate_time(key[1], duration, qpu=key[0], qubits=key[2] if len(key) > 2 else None)
        self.default_times = dict(DEFAULT_GATE_TIMES)
        if default_times:
            self.default_times.update(default_times)
        self.qpu_resolver = qpu_resolver

    def set_gate_time(self, gate, duration, qpu=None, qubits=None):
        """Override the duration of a native gate (optionally only on one QPU / qubits)."""
        qpu = getattr(qpu, "qpu_id", qpu)
        qubits = tuple(int(q) for q in qubits) if qubits is not None else None
        self.gate_times[(qpu, gate.lower(), qubits)] = duration
        self._cache.clear()

    def _user_time(self, qpu, name, qargs):
        qpu = getattr(qpu, "qpu_id", qpu)
        for key in ((qpu, name, qargs), (qpu, name, tuple(reversed(qargs))), (qpu, name, None),
                    (None, name, qargs), (None, name, None)):
            if key in self.gate_times:
                return self.gate_times[key]
        return None

    def _get_target(self, qpu):
        if isinstance(qpu, (int, np.integer)) and self.qpu_resolver is not None:
            qpu = self.qpu_resolver(qpu)
        if qpu is None:
            return None
        # Accept DQCQPU, BackendV2 or Target.
        if hasattr(qpu, "target"):
            return qpu.target
        if hasattr(qpu, "operation_names"):
            return qpu
        return None

    def _default(self, name, num_qubits):
        if name in self.default_times:
            return self.default_times[name]
        return self.default_times["2q"] if num_qubits >= 2 else self.default_times["1q"]

    def _from_target(self, target, name, qargs):
        if target is None or name not in target.operation_names:
            return None
        try:
            props = target[name]
        except KeyError:
            return None
        for key in (qargs, tuple(reversed(qargs))):
            p = props.get(key)
            if p is not None and p.duration is not None:
                return float(p.duration)
        known = [p.duration for p in props.values() if p is not None and p.duration is not None]
        return float(np.mean(known)) if known else None

    def _delay_time(self, gate, target):
        duration = gate.params[0] if gate.params else getattr(gate, "duration", 0)
        unit = getattr(gate, "unit", "dt")
        if unit == "dt":
            dt = getattr(target, "dt", None) if target is not None else None
            return float(duration) * (dt or 0.0)
        return float(duration) * _DELAY_UNITS.get(unit, 1.0)

    def get_gate_time(self, qpu, gate, qubits):
        name = _op_name(gate)
        qargs = tuple(int(q) for q in qubits)
        user = self._user_time(qpu, name, qargs) if self.gate_times else None
        if user is not None:
            return user
        if name in ZERO_DURATION_OPS:
            return 0.0

        target = self._get_target(qpu)
        if name == "delay":
            op = gate.operation if hasattr(gate, "operation") else gate
            return self._delay_time(op, target) if not isinstance(op, str) else 0.0

        key = (id(target), name, qargs)
        if key not in self._cache:
            duration = self._from_target(target, name, qargs)
            self._cache[key] = self._default(name, len(qargs)) if duration is None else duration
        return self._cache[key]


# ============================================================
# Network timing configured by SimDisQ
# ============================================================
class NetworkTimingProvider(TimingProvider):
    """
    EPR generation time and classical latency between QPUs.

    One EPR pair costs  T_EPR = T_generation + T_transmission:
        generation   : heralded entanglement generation, occupies the link
        transmission : delivery of the pair to both QPUs (default: fiber propagation of the
                       link distance in km); the link can start the next generation meanwhile

    Lookup order for a link (src, dst):
        1. per-link value set via set_link / QPUManager.add_coonnection(..., epr_time=..., ...)
        2. model(src, dst, distance) if provided (future: f(distance, loss, success prob, contention, ...))
        3. default (constant, or fiber propagation for transmission)
    """

    def __init__(self,
                 default_epr_time=DEFAULT_EPR_TIME,
                 default_classical_latency=DEFAULT_CLASSICAL_LATENCY,
                 epr_model=None,
                 classical_model=None,
                 transmission_model=None,
                 default_epr_transmission_time=None):
        """
        :param default_epr_transmission_time: constant transmission time; None -> fiber propagation
        """
        self.default_epr_time = default_epr_time
        self.default_classical_latency = default_classical_latency
        self.default_epr_transmission_time = default_epr_transmission_time
        self.epr_model = epr_model
        self.classical_model = classical_model
        self.transmission_model = transmission_model
        self.epr_time = {}               # {(src, dst): seconds}
        self.epr_transmission_time = {}  # {(src, dst): seconds}
        self.classical_latency = {}      # {(src, dst): seconds}
        self.distance = {}               # {(src, dst): distance}

    @staticmethod
    def _id(qpu):
        return getattr(qpu, "qpu_id", qpu)

    def set_link(self, src_qpu, dst_qpu, epr_time=None, classical_latency=None, distance=None,
                 epr_transmission_time=None):
        """Configure one bidirectional link."""
        a, b = self._id(src_qpu), self._id(dst_qpu)
        for key in ((a, b), (b, a)):
            if epr_time is not None:
                self.epr_time[key] = epr_time
            if epr_transmission_time is not None:
                self.epr_transmission_time[key] = epr_transmission_time
            if classical_latency is not None:
                self.classical_latency[key] = classical_latency
            if distance is not None:
                self.distance[key] = distance

    def register_link(self, src_qpu, dst_qpu, distance=None, epr_time=None, classical_latency=None,
                      epr_transmission_time=None):
        self.set_link(src_qpu, dst_qpu, epr_time, classical_latency, distance, epr_transmission_time)

    def _lookup(self, table, model, default, src_qpu, dst_qpu):
        key = (self._id(src_qpu), self._id(dst_qpu))
        if key in table:
            return table[key]
        if model is not None:
            return model(key[0], key[1], self.distance.get(key))
        return default

    def get_epr_time(self, src_qpu, dst_qpu):
        return self._lookup(self.epr_time, self.epr_model, self.default_epr_time, src_qpu, dst_qpu)

    def get_epr_transmission_time(self, src_qpu, dst_qpu):
        key = (self._id(src_qpu), self._id(dst_qpu))
        if key in self.epr_transmission_time or self.transmission_model is not None:
            return self._lookup(self.epr_transmission_time, self.transmission_model, 0.0, src_qpu, dst_qpu)
        if self.default_epr_transmission_time is not None:
            return self.default_epr_transmission_time
        return fiber_latency(self.distance.get(key) or 0.0)

    def get_classical_latency(self, src_qpu, dst_qpu):
        if self._id(src_qpu) == self._id(dst_qpu):
            return 0.0
        return self._lookup(self.classical_latency, self.classical_model,
                            self.default_classical_latency, src_qpu, dst_qpu)


# ============================================================
# Unified provider used by the scheduler
# ============================================================
class DQCTimingProvider(TimingProvider):
    """Route local operations to a local provider and network events to a network provider."""

    def __init__(self, local=None, network=None):
        self.local = local if local is not None else QiskitTimingProvider()
        self.network = network if network is not None else NetworkTimingProvider()

    def get_gate_time(self, qpu, gate, qubits):
        return self.local.get_gate_time(qpu, gate, qubits)

    def get_epr_time(self, src_qpu, dst_qpu):
        return self.network.get_epr_time(src_qpu, dst_qpu)

    def get_epr_transmission_time(self, src_qpu, dst_qpu):
        return self.network.get_epr_transmission_time(src_qpu, dst_qpu)

    def get_classical_latency(self, src_qpu, dst_qpu):
        return self.network.get_classical_latency(src_qpu, dst_qpu)

    def register_link(self, src_qpu, dst_qpu, distance=None, epr_time=None, classical_latency=None,
                      epr_transmission_time=None):
        self.network.register_link(src_qpu, dst_qpu, distance, epr_time, classical_latency,
                                   epr_transmission_time)
