# nvidia-rm570

Valori luate **automat** din antetele publicate de NVIDIA (open-gpu-kernel-modules, eticheta **570.144**, licență MIT — notificarea în
`LICENSE-NVIDIA`). Comanda C-20, pasul B.

- `src/regs.rs` — offset-urile registrelor GA102 (falcon, GSP, SEC2, RISC-V, FBIF, GC6/GFW) + `NV_PFB_PRI_MMU_WPR2_*` (TU102, folosit și pe Ampere).
- `src/structs.rs` — `GspFwWprMeta`, `msgqTxHeader`/`msgqRxHeader`, `GSP_ARGUMENTS_CACHED`, `MESSAGE_QUEUE_INIT_ARGUMENTS`, cu verificările de
  mărime și poziție calculate de clang (bindgen) la compilare.
- `src/lib.rs` (scris de noi) — tipurile `Field`/`Range` și testele.

**Doar date.** Nu citește și nu scrie nimic în placă; nu conține cod care rulează pe GPU. Fișierele generate nu se editează de mână.

## Regenerare (doar pe calculatorul gazdă)
```sh
cargo install bindgen-cli --version 0.71.1 --locked     # o singură dată; are nevoie de libclang
sh tools/gen-rm570/generate.sh /tmp/ogkm-570.144          # clonează antetele (commit fixat) și rescrie regs.rs + structs.rs
git diff crates/nvidia-rm570                              # trebuie să fie gol dacă nimic nu s-a schimbat
```
Commit-ul antetelor: `8ec351aeb96a93a4bb69ccc12a542bf8a8df2b6f` (tag `570.144`). Antetele nu intră în repo.

## Teste
`cargo test -p nvidia-rm570` (valorile citite de mână în studiul din hub) și `cargo test -p nvidia-ga104 --test rm570_crosscheck`
(offset-urile scrise de mână în `nvidia-ga104` în sesiunea 5 = cele din antete).

## Depinde de
Nimic (no_std, fără dependențe). Folosit de `nvidia-ga104` doar în teste, deocamdată.
