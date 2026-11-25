# C-157: `token32` — „un singur program pe simbol" pentru lista de generare de azi (sesiunea 239, 26 nov 2026, cloud, FĂRĂ placă)

**Stare:** COMPILAT (Mesa NAK 25.0.7) + SIMULARE (T248, 4/4). **NU RULAT pe placă.** Nimic nou trimis la placă.

## 1. De ce

`token.comp` (C-87, 7 nov) rulează toată lista unui simbol într-o singură lansare, dar știe doar programele de atunci (lista
„simplă”, 195 de lansări la #513). Placa generează azi cu `lipita-ln-32` + argmax pe placă (+ `embed_dev` la loturi): `layernorm256`,
`q8_fused32`, `q8_matvec32`, `argmax`, `embed_dev` — pe care `token.comp` le refuză (`kronos_token::work_list`). Proba barierei
(os-451, sesiunea 237) e pasul dinaintea lui `token.comp` pe placă; fără un program care știe lista de azi, proba n-ar avea ce să
deblocheze. `token32` e acel program. `token.comp` rămâne neschimbat.

## 2. Ce face

- Aceeași listă de lucru (256 de octeți pe operație: bufferul constant al lansării separate + operația la 0x20 + grila la 0x24),
  același buffer constant (lista 0x28, contorul 0x30, n 0x38, **G la 0x0c** — `token_cbuf_for`), aceeași barieră între operații.
- 13 operații (numerele din `kronos_token::op_number`): cele 7 vechi + `q8_fused` (7), `layernorm256` (8), `argmax` (9),
  `q8_fused32` (10), `q8_matvec32` (11), `embed_dev` (12).
- Programele de 32 de fire: același corp. Cele de 256 de fire pe elemente (`embed_dev`): 8 runde de 32.
- `layernorm256` (256 de fire = 8 warp-uri): firul t joacă, pe rând, firul 32·w + t al warp-ului w (w = 0..7): suma lui (k = 0..3,
  de la 0), același „fluture” în warp (shuffle xor 16..1), apoi cele 8 sume adunate în ordinea w = 0..7. x, g, b citite din nou din
  memorie în loc de registre (aceleași valori).
- `argmax`: cel mai mare logit finit, indicele cel mai mic la egalitate, numărul celor nefinite — exacte, nu depind de ce fir a
  văzut ce element → aceleași 4 cuvinte cu 32 de fire.
- Tot ce scrie un grup și citește altul după barieră (activările, cache-urile K/V, căsuța argmax) trece prin bufferele `coherent`.

## 3. Predicțiile (scrise în capul testului înainte de rulare)

| | Predicție | Ieșit |
|---|---|---|
| P1 | un simbol din lista de azi + argmax într-o lansare = cele 25 de lansări separate (modelul mic) / 11 (lățimea lui Kronos): aceiași biți ai logit-urilor, aceleași 4 cuvinte argmax, la G = 1, 7, 48 | **DA** (T248a, T248b) |
| P2 | tot textul (3 + 12 simboluri = 14 poziții, simbolurile generate citite pe placă) într-o singură lansare (350 de operații) = câte o poziție pe sonerie: aceleași simboluri, aceleași cuvinte argmax, aceleași cache-uri K/V bit cu bit | **DA** (T248c) |
| P3 | `token.comp` refuză lista de azi; operația 13 oprește modelul; bufferul constant fără G la 0x0c e refuzat | **DA** (T248d + testul din `kronos_token`) |

Cifrele compilării **nu** au fost prezise dinainte.

## 4. Compilarea (Mesa NAK 25.0.7, reconstruit în cloud; `token.comp` a ieșit identic: `ac176819…77d8a79c`)

| Ce | Valoare |
|---|---|
| `token32.sm86.sass` | **7 438 de instrucțiuni, 119 008 B**, SHA-256 `40d9749cc8074f639ca0eb3fafae9686b2ca4d2491e868b3b2544ec7faa1ad5f` |
| Registre | **92** (`token`: 35), 0 memorie locală (nicio vărsare) |
| Memoria partajată | 2 176 B (scorurile atenției + 32 de sume), ca `token` |
| G | citit din `c[0x0][0xc]` (2 locuri, ca la `token`) |
| Citiri | activări / cache-uri / căsuța argmax: `ld.global.strong.sys` (262 × 32 biți, 12 × 128 biți — 8 ale copierii, 4 ale lui x în sferturi); greutățile și lista: `.constant` (216); scrieri `.strong.sys`; contorul: `atom.add.strong.sys` |

**Verificarea listării față de programele separate** (contopirile decid biții):
- `layernorm256`: aceeași formă ca `layernorm256.sm86.txt` — `x − r·(1/n)` într-un singur `ffma` (media nu e rotunjită separat, în
  nicio variantă), `d·d`: `fmul` la primul element, `ffma` după, fluturele 5 × (`shfl.bfly` + `fadd`), sumele warp-urilor adunate în
  ordine, `r·(1/n) + eps` într-un `ffma`, `mufu.rsq`, ieșirea `fmul` + `ffma` cu g și b; primul element al fiecărei sume fără `0 + x`
  (la fel în amândouă).
- `q8_fused32` / `q8_matvec32`: secvența de 17 operații a unei sferturi (`i2f` ×8, `fmul`, `ffma` ×7, `ffma` cu scara) din
  `q8_matvec32.sm86.txt` apare **de 2 ori** în `token32` (o dată pe program).
- `argmax`: doar comparații pe chei întregi, nicio rotunjire.
Concluzia „`token32` dă biții programelor separate” rămâne **DEDUSĂ** (sursă + listare); modelul din `gpu-sim` rulează modelele
programelor separate, deci nu o poate dovedi. Pe placă: aceleași 16 simboluri / aceeași amprentă ca lista de azi.

## 5. Ce NU e dovedit și ce urmează pe placă

1. **Bariera** (toate cele G grupuri pe placă deodată, coerența fără L1): proba os-451, G = 48 și 96.
2. **Viteza — risc mare, DEDUS:** la G = 48 fiecare SM are **un singur warp**; lansările separate țin până la 16 blocuri pe SM
   (lățimea de bandă vine din multe citiri în zbor). Înmulțirile (cea mai mare parte a plăcii la generare) ar putea fi **mai lente**
   decât azi, nu mai rapide. Câștigul (~31 % goluri de lansare, DEDUS din os-346) cere G mare. Ocuparea pentru `token32` (92 de
   registre, 2 176 B; `occupancy::SM86`): registrele permit 16 blocuri pe SM (limita de blocuri); memoria partajată: 3 / 7 / 15 / 16
   blocuri la treptele 8 / 16 / 32 / 64 KiB (2 / 5 / 10 / 16 cu rezerva de 1 KiB) → **G = 768 doar la treapta 64 KiB**. Costul unei
   bariere cu 384–768 de grupuri nu e măsurat (B3/B4 din proba barierei sunt la 48 / 96).
3. **Ordinea propusă:** (a) proba barierei la G = 48 și 96 (os-451, deja GATA DE TEST); (b) dacă trece, aceeași probă la G = 384 și
   768 (doar parametrul G — `grid_barrier` are 19 registre, fără memorie partajată: 16 pe SM încap); (c) abia apoi `token32` cu G ales
   din (b), de sub o opțiune oprită implicit, comparat cu lista de azi (aceleași 16 simboluri) și cu timpul de azi.
   `token32_qmd` lasă treapta implicită (cea mai mică); pentru G > 3·48 OS-ul trebuie să ceară treapta 64 KiB
   (`ComputeQmd::to_bytes_with_sm_config`) — adăugare pentru atunci, nefăcută azi.

## 6. Cod (doar adăugări)

- `tools/nak/token32.comp`, `tools/nak/build.sh token32`, `tools/nak/out/token32.sm86.{sass,txt}`.
- `nvidia_ga104::kronos_token`: `TOKEN32`, `TOKEN32_REGISTERS` (92), `work_list32`, `token32_qmd` (`token_qmd` neschimbat în
  comportament: aceeași funcție internă).
- `gpu_sim::cmdproc`: `token32_on_cpu`, `TOKEN32_NEEDS`.
- Teste: `crates/nvidia-ga104/tests/sim_c157_token32.rs` (T248a–d), `common::token_sim::TokenSim::submit_token32`,
  testul `cbuf_qmd_and_groups` extins.

**Nicio schimbare incompatibilă.**
