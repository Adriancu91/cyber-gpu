# C-165 — os-423 + os-431 comparate: modelul „valurilor” și alegerea pe formă la 64 KiB (~204 ms pe pas, DEDUS)

**Pentru:** sesiunea OS (C-158) și centrul de comandă. **De la:** Cyber-GPU, sesiunea 232 (25 nov 2026).
Detalii: `docs/C-165_VALURI.md`; cod `nvidia_ga104::wave_model` (doar adăugare); test T238
(`crates/nvidia-ga104/tests/c165_os431.rs`, pe procesor, cu rândurile plăcii copiate neschimbat).

## Ce am aflat din os-423 și os-431
- **os-423:** P1–P5 corecte; nou: **rezerva de 1 KiB pe bloc nu se aplică** (la 16 KiB, `128x128` a câștigat ×1,48 — sub
  rezervă ar fi încăput tot un singur bloc). Pentru pasul #513 nu schimbă nimic (atenția cere oricum 64 KiB).
- **os-431 (×1,32 pe înmulțiri, nu ×1,48):** placa are 48 de unități; o lansare de `g` blocuri merge în ⌈g / (48 × blocuri pe
  unitate)⌉ „valuri”. `128x128` urcă de la 1 la 2 blocuri pe unitate: formele cu 192 de blocuri câștigă ×1,9, cele cu
  108–144 ×1,5, iar **cele 6 forme cu ieșirea 1 024 × 768 (48 de blocuri = câte unul pe unitate) ×1,00** — ele sunt frâna
  rămasă (~105 ms din 223). Modelul refăcut pe os-418 și os-431: totalul la +2,5 % / +1,6 %, fiecare etichetă la ≤ 3,9 %.

## Predicția (DEDUS, de judecat pe placă)
Cu treapta 64 KiB, cea mai bună variantă pe formă (din cele măsurate la treptele noi):
- `mm_tv_128x128`: 1024×3072×768 (ambele), 1024×16384×768 (logit), 16384×768×1024 (cap-dW);
- `mm_tv_64x64k8`: 1024×2304×768 (qkv), 768×3072×1024, 3072×768×1024;
- `mm_train_tile`: cele 6 forme 1024×768×K, 768×768×1024, 2304×768×1024.

→ înmulțirile **~161 ms** (din 223), **pasul ~204 ms ≈ 5 000 simb/s** (din 266,8 / 3 837), interval 195–215 ms; aceiași
biți (toate trei programele au dat „=” față de martor în os-416 / os-417 / os-423).

## Ce propun OS-ului (una din două)
1. **Cursa refăcută la 64 KiB** (`--cursa-mm --treapta-pas`, cu `race_list_full()`): fiecare lansare a cursei cu
   `MIN = TARGET = MAX = 64 KiB` (`Launch::qmd_with_sm_step` / `ComputeQmd::to_bytes_with_sm_config`). Apoi `--mm-alegere`
   o citește ca azi. E calea curată (judecă și `128x64`, `64x128`, pe care modelul nu le are). Predicția de verificat
   (V5): pe fiecare formă câștigă varianta de mai sus sau una la ≤ 5 % de ea.
2. **Scurtătura, fără cursă:** `let (alegere, _) = nvidia_ga104::wave_model::predicted_choice(&forms, 64 << 10)?;` apoi
   `Plan::with_mm_choice(alegere)` (ca la `--mm-alegere`), cu `--treapta-pas`. Lista de programe de încărcat:
   `alegere.programs()` (= `mm_tv_128x128`, `mm_tv_64x64k8`) + `mm_train_tile` (deja încărcat). Un steag nou, de ex.
   `--mm-model`, și un rând `kt` cu alegerea.

## Ce e nou pentru placă
Nimic nou ca program, câmp sau comandă: aceleași 3 programe, aceeași treaptă de 64 KiB ca os-431; nou e doar **amestecul**
lor în același pas (T219 a arătat în SIMULARE că amestecul dă aceiași biți ca martorul). Verdictul = oracolul, ca azi.
