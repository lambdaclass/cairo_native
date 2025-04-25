use bincode::{de::read::Reader, error::DecodeError};
use starknet_types_core::felt::Felt;
use std::ops::Deref;

#[derive(Debug)]
pub struct Memory(Vec<Option<Felt>>);

impl Memory {
    pub fn decode(mut data: impl Reader) -> Self {
        let mut memory = Vec::new();

        let mut addr_data = [0u8; 8];
        let mut value_data = [0u8; 32];
        loop {
            match data.read(&mut addr_data) {
                Ok(_) => {}
                Err(DecodeError::UnexpectedEnd { additional: 8 }) => break,
                e @ Err(_) => e.unwrap(),
            }
            data.read(&mut value_data).unwrap();

            let addr = u64::from_le_bytes(addr_data);
            let value = Felt::from_bytes_le(&value_data);

            if addr >= memory.len() as u64 {
                memory.resize(addr as usize + 1, None);
            }

            match &mut memory[addr as usize] {
                Some(_) => panic!("duplicated memory cell"),
                x @ None => *x = Some(value),
            }
        }

        Self(memory)
    }
}

impl Deref for Memory {
    type Target = [Option<Felt>];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
