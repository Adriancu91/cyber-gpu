# C-83 I14 — vectorii de control KGR1 pe canalul de lucru (sesiunea Cyber-GPU 59, 6 nov 2026) — doar SIMULARE

**Ce s-a adăugat:** doar teste + o unealtă de gazdă (nicio funcție publică nouă, nimic redenumit). `crates/nvidia-ga104/tests/sim_i14_token.rs`,
testele `kgr1_*`: modelul minuscul din hub (`integrare/kronos/vectori/mic_q8_0.kgr`) trece prin `kronos_model::token_launches` pe canalul de lucru
SIMULAT și prin oracolul `gpu_compute::model::step_cpu`; ambele dau exact `argmax_pe_pozitii` și `greedy_5` din `vectori.json` (partea `q8_0`),
logit-urile ultimei poziții la ≤ 4,6e-6 de motorul de referință al lui Kronos (pragul hub-ului: 1e-3).

Vectorii nu sunt în repo-ul public cyber-gpu (hub-ul e privat): `python3 -I tools/vectori-kgr1/pregateste.py <hub>/integrare/kronos/vectori target/vectori-kgr1`,
apoi `KGR1_VECTORI=$PWD/target/vectori-kgr1 cargo test --release -p nvidia-ga104 --test sim_i14_token kgr1 -- --ignored`.

**Propunere pentru OS (opțională):** în fereastra de luni, înainte de #513, I14 poate rula întâi `mic_q8_0.kgr` (840 KB, 35 de lansări pe simbol,
sub 1 MB VRAM) pe placă și compara cu `vectori.json` — un model la care știm deja răspunsul exact. Dacă trece aici și #513 nu, problema e de mărime
(VRAM, pushbuffer, timp), nu de calcul. Nicio scriere nouă pe placă față de I14.
