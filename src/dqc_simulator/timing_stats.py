"""
Timing statistics over a scheduled Timeline.

    T_execution = T_local + T_epr_generation + T_epr_transmission + T_classical + T_waiting

The decomposition follows the critical path (the chain of events that determines the
makespan), so the parts add up exactly to the makespan. Totals per gate / link / QPU are
also reported; those are summed over all events and may overlap in time.
All values are in seconds.
"""

import json
from collections import defaultdict


CATEGORIES = ("local", "epr_generation", "epr_transmission", "classical", "waiting")


class TimingStatistics:
    def __init__(self, timeline):
        self.timeline = timeline

    # ---------------- Critical path ----------------
    def critical_path(self):
        """
        Walk back from the last event through the events that released each other.

        :return: (breakdown {category: seconds}, [event indices, first -> last])
        """
        events = self.timeline.events
        breakdown = {c: 0.0 for c in CATEGORIES}
        if not events:
            return breakdown, []

        idx = max(range(len(events)), key=lambda k: events[k].end)
        upto = events[idx].end       # part of the current event that lies on the path
        cursor = upto                # path time already accounted for
        path = []
        while idx is not None:
            ev = events[idx]
            breakdown["waiting"] += max(cursor - upto, 0.0)
            remaining = upto - ev.start
            for category, duration in ev.segments:
                take = min(duration, max(remaining, 0.0))
                breakdown[category] += take
                remaining -= take
            path.append(idx)
            cursor = ev.start
            upto, idx = ev.pred_release, ev.pred
        breakdown["waiting"] += cursor   # before the first event on the path
        return breakdown, path[::-1]

    # ---------------- Local gates ----------------
    def gate_stats(self, by_qpu=False):
        """
        Count and execution time of each native local operation (x, sx, rz, cx, measure, reset, ...).
        Gates inside classically controlled corrections are included as if executed.
        """
        stats = {}
        for (qpu, name), (count, total) in self.timeline.op_stats.items():
            key = (qpu, name) if by_qpu else name
            entry = stats.setdefault(key, [0, 0.0])
            entry[0] += count
            entry[1] += total
        return {k: {"count": c, "total": t, "mean": t / c if c else 0.0}
                for k, (c, t) in sorted(stats.items(), key=lambda kv: str(kv[0]))}

    # ---------------- Network ----------------
    def epr_stats(self):
        """Per link: EPR count, generation / transmission / total time, time waiting for the link."""
        stats = {}
        for ev in self.timeline.events:
            if ev.kind != "epr":
                continue
            st = stats.setdefault(tuple(sorted(ev.qpus)), defaultdict(float))
            st["count"] += 1
            st["generation"] += ev.segment("epr_generation")
            st["transmission"] += ev.segment("epr_transmission")
            st["total"] += ev.duration
            st["link_wait"] += ev.wait
        return {link: self._finish(st) for link, st in sorted(stats.items())}

    def classical_stats(self):
        """Per directed channel (src, dst): message count and latency."""
        stats = {}
        for ev in self.timeline.events:
            if ev.kind != "classical":
                continue
            st = stats.setdefault(tuple(ev.qpus), defaultdict(float))
            st["count"] += 1
            st["total"] += ev.duration
        return {ch: self._finish(st) for ch, st in sorted(stats.items())}

    @staticmethod
    def _finish(st):
        st = dict(st)
        st["count"] = int(st["count"])
        st["mean"] = st["total"] / st["count"] if st["count"] else 0.0
        return st

    # ---------------- QPUs and qubits ----------------
    def qpu_stats(self):
        """Per QPU: local operation time, feed-forward wait, qubit idle time, active span."""
        tl = self.timeline
        stats = defaultdict(lambda: {"local_time": 0.0, "feedforward_wait": 0.0, "qubit_idle": 0.0,
                                     "start": float("inf"), "end": 0.0})
        for (qpu, _), (_, total) in tl.op_stats.items():
            stats[qpu]["local_time"] += total
        for ev in tl.events:
            for qpu in ev.qpus:
                stats[qpu]["start"] = min(stats[qpu]["start"], ev.start)
                stats[qpu]["end"] = max(stats[qpu]["end"], ev.end)
            if ev.kind == "feedforward":
                stats[ev.qpus[0]]["feedforward_wait"] += ev.wait
        for q, idle in tl.qubit_idle.items():
            stats[tl.qubit_qpu[q]]["qubit_idle"] += idle
        return dict(sorted(stats.items()))

    def idle_stats(self):
        """Per qubit: total / max idle gap and number of gaps."""
        return {q: {"total": sum(e - s for s, e in iv),
                    "max": max((e - s for s, e in iv), default=0.0),
                    "gaps": len(iv),
                    "qpu": self.timeline.qubit_qpu.get(q)}
                for q, iv in sorted(self.timeline.idle_intervals.items())}

    # ---------------- Export ----------------
    def to_dict(self):
        breakdown, path = self.critical_path()
        key = lambda k: "-".join(map(str, k)) if isinstance(k, tuple) else str(k)
        return {
            "makespan": self.timeline.makespan,
            "critical_path": breakdown,
            "critical_path_events": len(path),
            "gates": {key(k): v for k, v in self.gate_stats().items()},
            "gates_by_qpu": {key(k): v for k, v in self.gate_stats(by_qpu=True).items()},
            "epr": {key(k): v for k, v in self.epr_stats().items()},
            "classical": {key(k): v for k, v in self.classical_stats().items()},
            "qpus": {key(k): v for k, v in self.qpu_stats().items()},
            "qubit_idle": {key(k): v for k, v in self.idle_stats().items()},
        }

    def to_json(self, path=None, indent=2):
        text = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            with open(path, "w") as f:
                f.write(text)
        return text

    # ---------------- Report ----------------
    def report(self, by_qpu=False):
        us = 1e6
        makespan = self.timeline.makespan
        breakdown, path = self.critical_path()

        print("=" * 64)
        print(f"Execution time (makespan): {makespan * us:.3f} us")
        print(f"Critical path ({len(path)} events):")
        for category in CATEGORIES:
            t = breakdown[category]
            share = t / makespan * 100 if makespan else 0.0
            print(f"  {category:<17}{t * us:>14.3f} us {share:>7.2f} %")

        print("-" * 64)
        print(f"{'QPU':<5}" * by_qpu + f"{'gate':<10}{'count':>7}{'mean (us)':>13}{'total (us)':>13}")
        for k, st in self.gate_stats(by_qpu).items():
            qpu, name = k if by_qpu else (None, k)
            print(f"{qpu!s:<5}" * by_qpu + f"{name:<10}{st['count']:>7}{st['mean'] * us:>13.4f}{st['total'] * us:>13.4f}")

        print("-" * 64)
        print(f"{'link':<8}{'EPR':>5}{'gen (us)':>14}{'trans (us)':>12}{'total (us)':>14}{'wait (us)':>11}")
        for link, st in self.epr_stats().items():
            print(f"{'-'.join(map(str, link)):<8}{st['count']:>5}{st['generation'] * us:>14.3f}"
                  f"{st['transmission'] * us:>12.3f}{st['total'] * us:>14.3f}{st['link_wait'] * us:>11.3f}")

        print("-" * 64)
        print(f"{'channel':<8}{'msgs':>5}{'mean (us)':>14}{'total (us)':>12}")
        for ch, st in self.classical_stats().items():
            print(f"{'->'.join(map(str, ch)):<8}{st['count']:>5}{st['mean'] * us:>14.3f}{st['total'] * us:>12.3f}")

        print("-" * 64)
        print(f"{'QPU':<5}{'local (us)':>12}{'ff wait (us)':>14}{'qubit idle (us)':>17}{'span (us)':>13}")
        for qpu, st in self.qpu_stats().items():
            span = st["end"] - st["start"] if st["end"] >= st["start"] else 0.0
            print(f"{qpu!s:<5}{st['local_time'] * us:>12.3f}{st['feedforward_wait'] * us:>14.3f}"
                  f"{st['qubit_idle'] * us:>17.3f}{span * us:>13.3f}")
        print("=" * 64)
