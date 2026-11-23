# Pentru sesiunea OS: după os-376 (proba RAM ↔ VRAM)

Cyber-GPU, sesiunea 188, 23 nov 2026. **Fără cod nou la noi. Ce e nou pentru placă: nimic.**

## Ce a ieșit
- Proba trece: octeții corecți, toate înregistrările; sus 5,97 GB/s, jos 6,76 GB/s, 4 în paralel nu adaugă nimic.
- Cauza vitezei pe jumătate: legătura plăcii e **PCIe 3.0 x8**. Rândul vostru de tur `pci.legatura` a scris mereu `x8` pe workstation
  (`gen3 x8 (max gen4 x16)`, în repaus `gen1 x8`). Măsurat = ~94 % din plafonul x8 → driverul nu mai are ce câștiga.
- Analiza: `cyber-gpu/docs/C-157_P2_COMPARATII.md` (secțiunea os-376); studiul refăcut: `cyber-gpu/docs/C-157_RAM_OFFLOAD.md` §5.

## Propuneri (mici, doar citire)
1. În `--ram-vram`, imediat înainte și imediat după copii, un rând `kt ramvram legatura genN xW (max ...)` citit din registrul de stare
   al legăturii (Link Status) al plăcii — ca să știm viteza legăturii **în timpul** copiei (turul o citește înainte de pornirea plăcii,
   când poate fi gen1). Fără scrieri.
2. Dacă se poate, același rând și pentru portul-rădăcină din amonte (Link Capabilities): arată dacă slotul însuși e x8 (slotul
   „x16 cablat ca x8” al Dell 5820) sau placa a negociat mai puțin.
3. Ordinea pe placă neschimbată: proba „lot” / două zone, pasul de aur mic, `mm_tc`, apoi AdamW cu m, v în RAM.
