"""Trich xuat calibration snapshot tu Qiskit Fake Backends -> data/calibration.json.
Chay 1 lan; sau do toan bo pipeline KHONG can qiskit nua (tai lap duoc, khong phu thuoc phien ban)."""
import json, warnings, datetime
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

ROOT = Path(__file__).resolve().parent.parent
SYSTEMS = {
    "sherbrooke": dict(qubits=127, qv=32,  clops=904),
    "washington": dict(qubits=127, qv=64,  clops=850),
    "kolkata":    dict(qubits=27,  qv=128, clops=2000),
    "mumbai":     dict(qubits=27,  qv=128, clops=1800),
    "cairo":      dict(qubits=27,  qv=64,  clops=2400),
    "hanoi":      dict(qubits=27,  qv=64,  clops=2300),
    "guadalupe":  dict(qubits=16,  qv=32,  clops=2400),
    "perth":      dict(qubits=7,   qv=32,  clops=2900),
    "lagos":      dict(qubits=7,   qv=32,  clops=2700),
    "nairobi":    dict(qubits=7,   qv=32,  clops=2600),
}
MODEL = {127: "ibmq127", 27: "ibmq27", 16: "ibmq16", 7: "ibmq7"}


def med(values, default):
    values = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.median(values)) if values else float(default)


def main():
    by_name = {b.name: b for b in FakeProviderForBackendV2().backends()}
    out = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
           "source": "qiskit_ibm_runtime.fake_provider (static calibration snapshots)", "nodes": {}}
    for name, sysm in SYSTEMS.items():
        b = by_name["fake_" + name]
        t = b.target
        ops = set(t.operation_names)
        g2 = "cx" if "cx" in ops else ("ecr" if "ecr" in ops else "cz")
        e2 = [p.error for p in t[g2].values() if p is not None]
        d2 = [p.duration for p in t[g2].values() if p is not None and p.duration]
        e1 = [p.error for p in t["sx"].values() if p is not None]
        ero = [t["measure"][(q,)].error for q in range(b.num_qubits)]
        t1 = [b.qubit_properties(q).t1 for q in range(b.num_qubits)]
        t2 = [b.qubit_properties(q).t2 for q in range(b.num_qubits)]
        assert b.num_qubits == sysm["qubits"], (name, b.num_qubits)
        out["nodes"][name] = {
            "name": "ibm_" + name, "qubits": sysm["qubits"], "model": MODEL[sysm["qubits"]],
            "qv": sysm["qv"], "clops": sysm["clops"], "d1cps": sysm["clops"] * 19.95,
            "e_2q": med(e2, 0.01), "e_1q": med(e1, 3e-4), "e_ro": med(ero, 0.02),
            "t1_us": med(t1, 100e-6) * 1e6, "t2_us": med(t2, 100e-6) * 1e6,
            "dur_2q_ns": med(d2, 400e-9) * 1e9, "gate_2q": g2,
        }
        n = out["nodes"][name]
        print(f"{name:11s} q={n['qubits']:3d} 2q={g2:3s} e2q={n['e_2q']:.4f} e1q={n['e_1q']:.5f} "
              f"ero={n['e_ro']:.4f} T1={n['t1_us']:.0f}us T2={n['t2_us']:.0f}us")
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "calibration.json").write_text(json.dumps(out, indent=2))
    print("Saved data/calibration.json")


if __name__ == "__main__":
    main()
