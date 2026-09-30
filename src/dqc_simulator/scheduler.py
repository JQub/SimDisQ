"""
Event-based ASAP scheduler for merged distributed circuits.

Every duration comes from a TimingProvider. Waiting time is an output:
  - an EPR pair waits for its link to be free (link contention),
  - a feed-forward correction waits for the classical message from the remote QPU,
  - every qubit idles between its operations (idle_intervals -> decoherence model).
"""

from collections import defaultdict
from dataclasses import dataclass, field

from qiskit.circuit import Clbit


@dataclass
class TimelineEvent:
    name: str           # operation name
    kind: str           # 'gate' | 'measure' | 'reset' | 'epr' | 'classical' | 'feedforward'
    qpus: tuple         # QPU ids involved
    qubits: tuple       # global qubit indices in the merged circuit
    start: float
    end: float
    wait: float = 0.0   # time blocked by a network resource / message after qubits were free

    @property
    def duration(self):
        return self.end - self.start


@dataclass
class Timeline:
    events: list = field(default_factory=list)
    makespan: float = 0.0
    qubit_busy: dict = field(default_factory=dict)       # {q: busy seconds}
    idle_intervals: dict = field(default_factory=dict)   # {q: [(start, end), ...]}
    instruction_idle: dict = field(default_factory=dict) # {instr index: [(q, idle seconds), ...]} idle right before it

    @property
    def qubit_idle(self):
        """Total idle time of each qubit between its first and last operation."""
        return {q: sum(e - s for s, e in iv) for q, iv in self.idle_intervals.items()}

    def summary(self):
        by_kind = defaultdict(float)
        count = defaultdict(int)
        for ev in self.events:
            by_kind[ev.kind] += ev.duration
            count[ev.kind] += 1
        idle = self.qubit_idle
        return {
            "makespan": self.makespan,
            "duration_by_kind": dict(by_kind),     # summed, events may overlap in time
            "count_by_kind": dict(count),
            "total_network_wait": sum(ev.wait for ev in self.events),
            "max_qubit_idle": max(idle.values()) if idle else 0.0,
        }

    def print_summary(self):
        s = self.summary()
        print("=" * 60)
        print(f"Estimated execution time : {s['makespan'] * 1e6:.3f} us")
        for kind, t in s["duration_by_kind"].items():
            print(f"  {kind:<12} x{s['count_by_kind'][kind]:<5} total {t * 1e6:.3f} us")
        print(f"Network waiting time     : {s['total_network_wait'] * 1e6:.3f} us")
        print(f"Max qubit idle time      : {s['max_qubit_idle'] * 1e6:.3f} us")
        print("=" * 60)


class DQCScheduler:
    def __init__(self, timing):
        """:param timing: TimingProvider"""
        self.timing = timing

    def schedule(self, circuit, qubit_location):
        """
        :param circuit: merged QuantumCircuit
        :param qubit_location: {global_qubit: (qpu, physical_qubit)}, qpu is a DQCQPU
        :return: Timeline
        """
        timing = self.timing
        timeline = Timeline()

        qubit_ready = defaultdict(float)
        qubit_last = {}
        clbit_info = {}          # Clbit -> (time available, source qpu id)
        link_free = defaultdict(float)
        sent_messages = set()    # (clbit, dst qpu id), avoid duplicated classical events

        def qpu_id(qpu):
            return getattr(qpu, "qpu_id", qpu)

        def commit(ev, qubits, busy):
            for q in qubits:
                if q in qubit_last and ev.start > qubit_last[q]:
                    timeline.idle_intervals.setdefault(q, []).append((qubit_last[q], ev.start))
                    timeline.instruction_idle.setdefault(i, []).append((q, ev.start - qubit_last[q]))
                timeline.idle_intervals.setdefault(q, [])
                timeline.qubit_busy[q] = timeline.qubit_busy.get(q, 0.0) + busy
                qubit_ready[q] = ev.end
                qubit_last[q] = ev.end
            if ev.duration > 0 or ev.kind != "gate":
                timeline.events.append(ev)
            timeline.makespan = max(timeline.makespan, ev.end)

        for i, ci in enumerate(circuit.data):
            op = ci.operation
            name = op.name
            qs = [circuit.find_bit(q).index for q in ci.qubits]
            locs = [qubit_location[q] for q in qs]
            qpus = []
            for qpu, _ in locs:
                if qpu_id(qpu) not in [qpu_id(p) for p in qpus]:
                    qpus.append(qpu)
            ids = tuple(qpu_id(p) for p in qpus)
            ready = max((qubit_ready[q] for q in qs), default=0.0)

            # --- Barrier: synchronize, no duration. ---
            if name == "barrier":
                for q in qs:
                    qubit_ready[q] = ready
                continue

            # --- EPR pair shared by two QPUs. ---
            if name == "initialize" and len(qpus) == 2:
                link = tuple(sorted(ids))
                start = max(ready, link_free[link])
                end = start + timing.get_epr_time(qpus[0], qpus[1])
                link_free[link] = end
                commit(TimelineEvent("epr", "epr", ids, tuple(qs), start, end, start - ready), qs, end - start)
                continue

            # --- Classically controlled correction. ---
            if name == "if_else":
                dst = qpus[0]
                cond = op.condition[0]
                cond_bits = [cond] if isinstance(cond, Clbit) else list(cond)
                arrival = ready
                for cb in cond_bits:
                    if cb not in clbit_info:
                        continue
                    t, src = clbit_info[cb]
                    lat = timing.get_classical_latency(src, qpu_id(dst))
                    arrival = max(arrival, t + lat)
                    if lat > 0 and (cb, qpu_id(dst)) not in sent_messages:
                        sent_messages.add((cb, qpu_id(dst)))
                        timeline.events.append(
                            TimelineEvent("classical", "classical", (src, qpu_id(dst)), (), t, t + lat))
                # Body runs sequentially on the destination QPU.
                body = op.blocks[0]
                dur = 0.0
                for inner in body.data:
                    phys = [locs[body.find_bit(q).index][1] for q in inner.qubits]
                    dur += timing.get_gate_time(dst, inner.operation, phys)
                start = arrival
                commit(TimelineEvent(name, "feedforward", ids, tuple(qs), start, start + dur, start - ready),
                       qs, dur)
                continue

            # --- Local operation on one QPU. ---
            qpu = qpus[0] if qpus else None
            phys = [p for _, p in locs]
            dur = timing.get_gate_time(qpu, op, phys)
            kind = name if name in ("measure", "reset") else "gate"
            ev = TimelineEvent(name, kind, ids, tuple(qs), ready, ready + dur)
            commit(ev, qs, dur)
            if name == "measure":
                for cb in ci.clbits:
                    clbit_info[cb] = (ev.end, qpu_id(qpu))

        return timeline
