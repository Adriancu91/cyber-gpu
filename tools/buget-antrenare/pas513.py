#!/usr/bin/env python3
"""C-157 punctul 2 dupa os-395: unde s-ar duce secunda pasului #513 (lot 2 x 512) — DEDUS, doar socoteala.

Numara operatiile (FLOP = 2 x inmultiri-adunari) ale pasului F32 pe grupe (inmultirile modelului, capul legat,
atentia, restul) si arata cat ar lua fiecare la cateva viteze presupuse. Nimic de aici nu e masurat; masuratoarea
e profilul pasului pe programe (`nvidia_ga104::train_profile`) facut de OS pe placa.

    python3 tools/buget-antrenare/pas513.py
"""

V, T, C, H, L = 16384, 512, 768, 12, 12
B = 2
N = B * T  # simboluri pe pas

def mm(m, n, k):
    return 2.0 * m * n * k

# inainte, pe strat: qkv, out_proj, fc, proj
fwd_layer = mm(N, 3 * C, C) + mm(N, C, C) + mm(N, 4 * C, C) + mm(N, C, 4 * C)
groups = {
    "inmultiri straturi (inainte)": L * fwd_layer,
    "inmultiri straturi (inapoi: dX + dW)": 2 * L * fwd_layer,
    "capul legat (inainte)": mm(N, V, C),
    "capul legat (inapoi: dX + dW)": 2 * mm(N, V, C),
    # atentia: S = Q K^T si P V, pe fiecare secventa, patratul intreg (programele de baza nu sar triunghiul)
    "atentia (inainte)": L * B * (mm(T, T, C) + mm(T, C, T)),
    "atentia (inapoi)": L * B * 2 * (mm(T, T, C) + mm(T, C, T)),
}
total = sum(groups.values())
measured_s = 1.123236  # os-395, placa, median pe pas
print(f"pasul #513, lot {B} x {T} = {N} simboluri: {total / 1e9:.0f} GFLOP")
print(f"masurat (os-395): {measured_s * 1e3:.0f} ms pe pas -> {total / measured_s / 1e12:.2f} TFLOP/s efectiv")
for name, f in groups.items():
    print(f"  {name:40s} {f / 1e9:7.1f} GFLOP  {100 * f / total:5.1f} %")
for rate in (0.6, 2.6, 8.0, 17.0):
    mmf = sum(v for k, v in groups.items() if not k.startswith("atentia"))
    print(f"inmultirile la {rate:5.1f} TFLOP/s: {mmf / rate / 1e9:7.1f} ms")
