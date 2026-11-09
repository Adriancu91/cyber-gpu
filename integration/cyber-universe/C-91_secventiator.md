# C-91 — reparația secvențiatorului (I8)

- **Commit cyber-gpu:** vezi `stare/cyber-gpu.md` din hub (ramura `g0-g2-fundatie`, după `fb38d18`).
- **Ce s-a schimbat:** doar `nvidia_ga104::gsp::rpc::decode_sequencer`. Comenzile = cuvintele `0..cmdIndex`; refuzat dacă `bufferSizeDWord == 0`,
  `cmdIndex > bufferSizeDWord` sau `cmdIndex` > cuvintele din mesaj. Interfața publică e aceeași (aceleași câmpuri în `SequencerHeader`;
  `buffer_size_dwords` înseamnă acum explicit „capacitatea”, `commands.len() == cmd_index`).
- **Ce are de făcut OS-ul (P6-68):** doar `kernel/Cargo.toml` pe noul commit; niciun apel nou, nicio schimbare de cod OS.
- **Testat:** doar SIMULARE (417 teste PASS, build no_std PASS). Pe placă se vede la următoarea rulare `gpu lant`: dacă trece de secvențiator, următorul
  pas de urmărit e `GSP_INIT_DONE` (sau o eroare nouă, cinstită, la primul pas diferit de simulare).
