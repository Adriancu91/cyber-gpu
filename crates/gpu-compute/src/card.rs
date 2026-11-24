//! C-157 point 8 — the model card: what a KGR1 file says about its model (architecture, weight kind, layers, heads,
//! n_emb, context, vocabulary, LayerNorm epsilon, tied head), read from the header and the key-values only, never the
//! tensors. Adrian's decisions of 20 Nov 19:41 / 19:45 (hub `DECIZII.md`): everything Kronos-specific in the driver is
//! built from this card, and the card picks the compute pack (`nvidia_ga104::packs`).
//!
//! The KGR1 format is Kronos's (hub `integrare/kronos/format_greutati.md`, read as a specification); this reader is
//! ours. Pure parsing, no I/O, no panics on any input.

use alloc::string::String;
use alloc::vec::Vec;

use crate::layer::Shape;

/// The model family the file declares (`arhitectura`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Arch {
    /// `kronos_gpt`: the only family the driver knows.
    KronosGpt,
    /// Anything else, kept so the refusal can name it.
    Other(String),
}

/// The weight kind the file declares (`fel`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Quant {
    F32,
    Q8_0,
    Other(String),
}

/// What a KGR1 file says about its model.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelCard {
    pub arch: Arch,
    pub quant: Quant,
    pub layers: usize,
    pub heads: usize,
    pub e: usize,
    pub context: usize,
    pub vocab: usize,
    pub eps: f32,
    pub tied_head: bool,
}

/// Why a card could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CardError {
    /// The file ends before the header or a key-value is complete.
    Truncated,
    /// Not `KGR1`, or a version other than 1.
    Magic,
    Version(u32),
    /// A value kind other than 1 (i64), 2 (f64), 3 (text).
    ValueKind(u32),
    /// A key the card needs is missing or has the wrong kind (the key's name).
    Missing(&'static str),
    /// A number that does not fit (negative, too large).
    Range(&'static str),
    /// A text that is not UTF-8.
    Text,
}

impl CardError {
    /// One line for the OS console, in Romanian.
    pub fn line(&self) -> String {
        use alloc::format;
        match self {
            CardError::Truncated => {
                "fisa KGR1: fisierul se termina inainte de antet sau de chei".into()
            }
            CardError::Magic => "fisa KGR1: nu incepe cu \"KGR1\"".into(),
            CardError::Version(v) => format!("fisa KGR1: versiunea {v}, stiu doar 1"),
            CardError::ValueKind(k) => format!("fisa KGR1: fel de valoare necunoscut ({k})"),
            CardError::Missing(k) => format!("fisa KGR1: lipseste cheia `{k}` (sau are alt fel)"),
            CardError::Range(k) => {
                format!("fisa KGR1: cheia `{k}` are o valoare in afara limitelor")
            }
            CardError::Text => "fisa KGR1: un text nu e UTF-8".into(),
        }
    }
}

struct Cursor<'a> {
    f: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], CardError> {
        let end = self.at.checked_add(n).ok_or(CardError::Truncated)?;
        let s = self.f.get(self.at..end).ok_or(CardError::Truncated)?;
        self.at = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, CardError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64, CardError> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }
    fn text(&mut self) -> Result<&'a [u8], CardError> {
        let n = usize::try_from(self.u64()?).map_err(|_| CardError::Truncated)?;
        self.take(n)
    }
}

enum Value<'a> {
    Int(i64),
    Float(f64),
    Text(&'a [u8]),
}

impl ModelCard {
    /// Reads the card from the start of a KGR1 file (the header and the key-values; the tensor list and the data
    /// are not needed, so the first few hundred KiB of the file are enough — the tokenizer text is skipped).
    pub fn from_kgr1(f: &[u8]) -> Result<ModelCard, CardError> {
        let mut c = Cursor { f, at: 0 };
        if c.take(4)? != b"KGR1" {
            return Err(CardError::Magic);
        }
        let version = c.u32()?;
        if version != 1 {
            return Err(CardError::Version(version));
        }
        let n_kv = c.u64()?;
        let _n_tensors = c.u64()?;
        let mut kv: Vec<(&[u8], Value<'_>)> = Vec::new();
        for _ in 0..n_kv {
            let k = c.text()?;
            let v = match c.u32()? {
                1 => Value::Int(c.u64()? as i64),
                2 => Value::Float(f64::from_bits(c.u64()?)),
                3 => Value::Text(c.text()?),
                other => return Err(CardError::ValueKind(other)),
            };
            kv.push((k, v));
        }
        // A key written twice: the last one counts.
        let get = |k: &'static str| {
            kv.iter()
                .rev()
                .find(|(n, _)| *n == k.as_bytes())
                .map(|(_, v)| v)
        };
        let int = |k: &'static str| -> Result<usize, CardError> {
            match get(k) {
                Some(Value::Int(v)) => usize::try_from(*v).map_err(|_| CardError::Range(k)),
                _ => Err(CardError::Missing(k)),
            }
        };
        let text = |k: &'static str| -> Result<String, CardError> {
            match get(k) {
                Some(Value::Text(t)) => core::str::from_utf8(t)
                    .map(String::from)
                    .map_err(|_| CardError::Text),
                _ => Err(CardError::Missing(k)),
            }
        };
        let arch = match text("arhitectura")?.as_str() {
            "kronos_gpt" => Arch::KronosGpt,
            other => Arch::Other(other.into()),
        };
        let quant = match text("fel")?.as_str() {
            "f32" => Quant::F32,
            "q8_0" => Quant::Q8_0,
            other => Quant::Other(other.into()),
        };
        let eps = match get("ln_eps") {
            Some(Value::Float(v)) if v.is_finite() && *v > 0.0 && *v < 1.0 => *v as f32,
            Some(Value::Float(_)) => return Err(CardError::Range("ln_eps")),
            _ => return Err(CardError::Missing("ln_eps")),
        };
        let tied_head = match int("cap_legat")? {
            0 => false,
            1 => true,
            _ => return Err(CardError::Range("cap_legat")),
        };
        Ok(ModelCard {
            arch,
            quant,
            layers: int("n_straturi")?,
            heads: int("n_capete")?,
            e: int("n_emb")?,
            context: int("context")?,
            vocab: int("vocabular")?,
            eps,
            tied_head,
        })
    }

    /// The layer shape the programs take.
    pub fn shape(&self) -> Shape {
        Shape {
            e: self.e,
            heads: self.heads,
            eps: self.eps,
        }
    }

    /// The values per head (`n_emb / n_capete`), or `None` when the heads do not divide n_emb.
    pub fn head_dim(&self) -> Option<usize> {
        if self.heads == 0 || !self.e.is_multiple_of(self.heads) {
            None
        } else {
            Some(self.e / self.heads)
        }
    }

    /// The numbers that make two cards "the same form" for a proof on the board (weights may differ).
    pub fn form(&self) -> (usize, usize, usize, usize, usize) {
        (self.layers, self.heads, self.e, self.context, self.vocab)
    }
}

/// Builds the header and key-values of a KGR1 file (no tensors): for tests and host tools.
pub fn kgr1_header(keys: &[(&str, KeyValue<'_>)]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"KGR1");
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&(keys.len() as u64).to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    let text = |b: &mut Vec<u8>, t: &[u8]| {
        b.extend_from_slice(&(t.len() as u64).to_le_bytes());
        b.extend_from_slice(t);
    };
    for (k, v) in keys {
        text(&mut b, k.as_bytes());
        match v {
            KeyValue::Int(i) => {
                b.extend_from_slice(&1u32.to_le_bytes());
                b.extend_from_slice(&i.to_le_bytes());
            }
            KeyValue::Float(f) => {
                b.extend_from_slice(&2u32.to_le_bytes());
                b.extend_from_slice(&f.to_bits().to_le_bytes());
            }
            KeyValue::Text(t) => {
                b.extend_from_slice(&3u32.to_le_bytes());
                text(&mut b, t.as_bytes());
            }
        }
    }
    b
}

/// One key-value for [`kgr1_header`].
#[derive(Clone, Copy, Debug)]
pub enum KeyValue<'a> {
    Int(i64),
    Float(f64),
    Text(&'a str),
}

/// The keys of a Kronos card with these numbers (Q8_0, tied head, eps 1e-5) — the form `exporta_greutati.py` writes.
pub fn kronos_keys(
    layers: i64,
    heads: i64,
    e: i64,
    context: i64,
    vocab: i64,
) -> Vec<(&'static str, KeyValue<'static>)> {
    alloc::vec![
        ("arhitectura", KeyValue::Text("kronos_gpt")),
        ("fel", KeyValue::Text("q8_0")),
        ("sursa", KeyValue::Text("test")),
        ("n_straturi", KeyValue::Int(layers)),
        ("n_capete", KeyValue::Int(heads)),
        ("n_emb", KeyValue::Int(e)),
        ("context", KeyValue::Int(context)),
        ("vocabular", KeyValue::Int(vocab)),
        ("ln_eps", KeyValue::Float(1e-5)),
        ("cap_legat", KeyValue::Int(1)),
        (
            "tokenizer.bpe",
            KeyValue::Text("{\"versiune\":2,\"reguli\":\"ws\",\"imbinari\":[],\"speciale\":[]}")
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k513() -> Vec<u8> {
        kgr1_header(&kronos_keys(12, 12, 768, 512, 16_384))
    }

    #[test]
    fn reads_513() {
        let c = ModelCard::from_kgr1(&k513()).unwrap();
        assert_eq!(c.arch, Arch::KronosGpt);
        assert_eq!(c.quant, Quant::Q8_0);
        assert_eq!(c.form(), (12, 12, 768, 512, 16_384));
        assert_eq!(c.head_dim(), Some(64));
        assert!(c.tied_head);
        assert_eq!(c.eps, 1e-5);
        assert_eq!(c.shape().e, 768);
    }

    #[test]
    fn every_truncation_is_an_error_not_a_panic() {
        let f = k513();
        for n in 0..f.len() {
            assert!(ModelCard::from_kgr1(&f[..n]).is_err(), "taiat la {n}");
        }
        // Trailing bytes (the tensor list) are ignored.
        let mut g = f.clone();
        g.extend_from_slice(&[0xAB; 100]);
        assert!(ModelCard::from_kgr1(&g).is_ok());
    }

    #[test]
    fn errors() {
        let mut f = k513();
        f[0] = b'X';
        assert_eq!(ModelCard::from_kgr1(&f), Err(CardError::Magic));
        let mut f = k513();
        f[4] = 2;
        assert_eq!(ModelCard::from_kgr1(&f), Err(CardError::Version(2)));
        let keys: Vec<_> = kronos_keys(12, 12, 768, 512, 16_384)
            .into_iter()
            .filter(|(k, _)| *k != "vocabular")
            .collect();
        assert_eq!(
            ModelCard::from_kgr1(&kgr1_header(&keys)),
            Err(CardError::Missing("vocabular"))
        );
        let mut keys = kronos_keys(12, 12, 768, 512, 16_384);
        keys.push(("n_capete", KeyValue::Int(-1)));
        assert_eq!(
            ModelCard::from_kgr1(&kgr1_header(&keys)),
            Err(CardError::Range("n_capete"))
        );
        let mut keys = kronos_keys(12, 12, 768, 512, 16_384);
        keys.push(("n_emb", KeyValue::Text("768")));
        assert_eq!(
            ModelCard::from_kgr1(&kgr1_header(&keys)),
            Err(CardError::Missing("n_emb"))
        );
        let mut keys = kronos_keys(12, 12, 768, 512, 16_384);
        keys.push(("cap_legat", KeyValue::Int(2)));
        assert_eq!(
            ModelCard::from_kgr1(&kgr1_header(&keys)),
            Err(CardError::Range("cap_legat"))
        );
        let mut keys = kronos_keys(12, 12, 768, 512, 16_384);
        keys.push(("ln_eps", KeyValue::Float(f64::NAN)));
        assert_eq!(
            ModelCard::from_kgr1(&kgr1_header(&keys)),
            Err(CardError::Range("ln_eps"))
        );
        // An unknown value kind.
        let mut f = kgr1_header(&[("x", KeyValue::Int(1))]);
        let at = 4 + 4 + 8 + 8 + 8 + 1;
        f[at] = 9;
        assert_eq!(ModelCard::from_kgr1(&f), Err(CardError::ValueKind(9)));
        for e in [
            CardError::Truncated,
            CardError::Magic,
            CardError::Version(3),
            CardError::ValueKind(9),
            CardError::Missing("n_emb"),
            CardError::Range("context"),
            CardError::Text,
        ] {
            assert!(e.line().starts_with("fisa KGR1"));
        }
    }

    #[test]
    fn other_arch_and_kind_are_kept_for_the_refusal() {
        let mut keys = kronos_keys(12, 12, 768, 512, 16_384);
        keys.push(("arhitectura", KeyValue::Text("llama")));
        keys.push(("fel", KeyValue::Text("f32")));
        let c = ModelCard::from_kgr1(&kgr1_header(&keys)).unwrap();
        assert_eq!(c.arch, Arch::Other("llama".into()));
        assert_eq!(c.quant, Quant::F32);
    }

    #[test]
    fn huge_lengths_do_not_overflow() {
        let mut f = Vec::new();
        f.extend_from_slice(b"KGR1");
        f.extend_from_slice(&1u32.to_le_bytes());
        f.extend_from_slice(&1u64.to_le_bytes());
        f.extend_from_slice(&0u64.to_le_bytes());
        f.extend_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(ModelCard::from_kgr1(&f), Err(CardError::Truncated));
    }
}
