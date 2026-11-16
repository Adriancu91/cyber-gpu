# Analiza jurnalului `gpu-jurnal-004.txt` (os-302, workstation, 16 nov 2026) — I9 = `0x57`

Sesiunea Cyber-GPU 103 (cloud, fără placă). Sursa: hub `rezultate/2026-11-16-workstation-os302/` (`REZULTAT.md`, `gpu-jurnal-004.txt`, 2493 rânduri).

## Ce s-a observat (OBSERVAT pe placă, RTX 3070 Ti, cyber-gpu `835f24d`, varianta 1)
- I0–I8 TREC din nou; GSP_INIT_DONE la 148,99 s; golden (I8e) TRECE la 158,70 s.
- I9 `i9-canal-lucru`: ESEC, `text motivul esecului = "eroare hardware 0x57"` = **`NV_ERR_OBJECT_NOT_FOUND`** (`nvstatuscodes.h` 570.144).
- Numărarea soneriilor (metoda din `os-295_ANALIZA_I9.md`): pică exact al doilea `PROMOTE_CTX` al canalului de lucru (jumătatea **virtuală**); jumătatea
  fizică trece. Înainte de refuz GSP-RM a trimis 2 înregistrări NOCAT ASSERT (adrese în GSP-RM `0x01443f48`, `0x01418a2a`); alte 2 la alocarea canalului
  (`0x01ab9bda`, `0x01ab9c56`), fără refuz. Adresele nu se pot citi în firmware (licența NVIDIA interzice analiza lui) — rămân doar ca amprente de comparat
  între rulări.
- Curățarea a mers (Bus Master oprit, `tur-dupa: ok`).

## Ce înseamnă (DEDUS, din sursele MIT citite ca specificație)
- În driverul NVIDIA 570.144, pentru un spațiu de adrese „deținut din afară” (cazul nostru), RM-ul de pe CPU **nu** trimite jumătatea virtuală
  (`kgrctxPreparePromoteCtxBuffer`: „RM is not responsible for promoting the buffers when UVM is enabled”). O trimite UVM (`nvGpuOpsBindChannelResources`),
  după ce a aflat lista bufferelor cu `GR_GET_CTX_BUFFER_INFO` și le-a mapat cum cere RM. Forma mesajului nostru (`hObject` = canalul, doar `bufferId` +
  `gpuVirtAddr`) e aceeași.
- `OBJECT_NOT_FOUND` = GSP-RM caută ceva ce nu găsește la VA-ul dat sau pentru canal. Candidați, fără dovadă: (a) GSP-RM vrea să găsească maparea VA în
  felul lui (PTE privilegiat — V2; pagină/aliniere — V1); (b) obiectul de context al canalului e altul decât cel pe care îl caută (de exemplu bufferele
  trebuie „reținute” de RM-ul de pe CPU, pe care noi nu-l avem); (c) diferență de firmware (tinygrad merge cu 570.190; noi 570.144 — V4, doar notat).
- `GR_GET_CTX_BUFFER_INFO` are în 570.144 flagurile `0x8000` (`RMCTRL_FLAGS_CPU_PLUGIN_FOR_LEGACY`), **fără** `ROUTE_TO_PHYSICAL` (`g_subdevice_nvoc.c`),
  spre deosebire de `PROMOTE_CTX` (`0x10244`, cu `ROUTE_TO_PHYSICAL`). Deci e servit de RM-ul de pe CPU; GSP-RM singur s-ar putea să-l refuze. Varianta V3
  e scrisă să noteze și refuzul și să meargă mai departe.

## Ce s-a făcut (C-125, doar SIMULARE, `78704e5`)
- V2 + V3 în spatele `I9Options` (implicit oprite); raportul ajunge în jurnal și când promovarea pică. Detalii: `integration/cyber-universe/C-125_i9-v2-v3.md`,
  teste `TEST_MATRIX.md` T129.

## Următorul pas pe placă (doar cu Adrian)
`gpu lant --i9 c125` după ce OS-ul leagă opțiunea. Rezultatul posibil: (1) I9 trece → I10 (add64); (2) `0x57` din nou, dar cu lista RM → V1 sau promovare
într-un singur mesaj (PA+VA); (3) `0x57` + V3 refuzat → V1 / mesaj unic, de decis cu Adrian.
