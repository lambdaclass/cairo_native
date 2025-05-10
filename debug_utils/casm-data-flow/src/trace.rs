use bincode::{de::read::Reader, error::DecodeError};
use cairo_vm::vm::trace::trace_entry::RelocatedTraceEntry;
use std::ops::Deref;

#[derive(Debug)]
pub struct Trace(Vec<RelocatedTraceEntry>);

impl Trace {
    pub fn decode(mut data: impl Reader) -> Self {
        let mut trace = Vec::new();

        let mut buf = [0u8; 8];
        loop {
            match data.read(&mut buf) {
                Ok(_) => {}
                Err(DecodeError::UnexpectedEnd { additional: 8 }) => break,
                e @ Err(_) => e.unwrap(),
            }
            let ap = u64::from_le_bytes(buf) as usize;

            data.read(&mut buf).unwrap();
            let fp = u64::from_le_bytes(buf) as usize;

            data.read(&mut buf).unwrap();
            let pc = u64::from_le_bytes(buf) as usize;

            trace.push(RelocatedTraceEntry { pc, ap, fp });
        }

        Self(trace)
    }
}

impl Deref for Trace {
    type Target = [RelocatedTraceEntry];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
