#![cfg(target_arch = "aarch64")]

use super::AbiArgument;
use crate::{starknet::U256, utils::get_integer_layout};
use num_traits::ToBytes;
use starknet_types_core::felt::Felt;

fn align_to(buffer: &mut Vec<u8>, align: usize) {
    buffer.resize(buffer.len().next_multiple_of(align), 0);
}

impl AbiArgument for u8 {
    fn to_bytes(&self, buffer: &mut Vec<u8>) {
        if buffer.len() < 64 {
            buffer.extend_from_slice(&(*self as u64).to_ne_bytes());
        } else {
            align_to(buffer, get_integer_layout(8).align());
            buffer.push(*self);
        }
    }
}

impl AbiArgument for i8 {
    fn to_bytes(&self, buffer: &mut Vec<u8>) {
        if buffer.len() < 64 {
            buffer.extend_from_slice(&(*self as u64).to_ne_bytes());
        } else {
            align_to(buffer, get_integer_layout(8).align());
            buffer.extend_from_slice(&self.to_ne_bytes());
        }
    }
}

impl AbiArgument for u16 {
    fn to_bytes(&self, buffer: &mut Vec<u8>) {
        if buffer.len() < 64 {
            buffer.extend_from_slice(&(*self as u64).to_ne_bytes());
        } else {
            align_to(buffer, get_integer_layout(16).align());
            buffer.extend_from_slice(&self.to_ne_bytes());
        }
    }
}

impl AbiArgument for i16 {
    fn to_bytes(&self, buffer: &mut Vec<u8>) {
        if buffer.len() < 64 {
            buffer.extend_from_slice(&(*self as u64).to_ne_bytes());
        } else {
            align_to(buffer, get_integer_layout(16).align());
            buffer.extend_from_slice(&self.to_ne_bytes());
        }
    }
}

impl AbiArgument for u32 {
    fn to_bytes(&self, buffer: &mut Vec<u8>) {
        if buffer.len() < 64 {
            buffer.extend_from_slice(&(*self as u64).to_ne_bytes());
        } else {
            align_to(buffer, get_integer_layout(32).align());
            buffer.extend_from_slice(&self.to_ne_bytes());
        }
    }
}

impl AbiArgument for i32 {
    fn to_bytes(&self, buffer: &mut Vec<u8>) {
        if buffer.len() < 64 {
            buffer.extend_from_slice(&(*self as u64).to_ne_bytes());
        } else {
            align_to(buffer, get_integer_layout(32).align());
            buffer.extend_from_slice(&self.to_ne_bytes());
        }
    }
}

impl AbiArgument for u64 {
    fn to_bytes(&self, buffer: &mut Vec<u8>) {
        if buffer.len() < 64 {
            buffer.extend_from_slice(&self.to_ne_bytes());
        } else {
            align_to(buffer, get_integer_layout(64).align());
            buffer.extend_from_slice(&self.to_ne_bytes());
        }
    }
}

impl AbiArgument for i64 {
    fn to_bytes(&self, buffer: &mut Vec<u8>) {
        if buffer.len() < 64 {
            buffer.extend_from_slice(&self.to_ne_bytes());
        } else {
            align_to(buffer, get_integer_layout(64).align());
            buffer.extend_from_slice(&self.to_ne_bytes());
        }
    }
}

impl AbiArgument for u128 {
    fn to_bytes(&self, buffer: &mut Vec<u8>) {
        if buffer.len() < 56 {
            buffer.extend_from_slice(&self.to_ne_bytes());
        } else {
            align_to(buffer, get_integer_layout(128).align());
            buffer.extend_from_slice(&self.to_ne_bytes());
        }
    }
}

impl AbiArgument for i128 {
    fn to_bytes(&self, buffer: &mut Vec<u8>) {
        if buffer.len() < 56 {
            buffer.extend_from_slice(&self.to_ne_bytes());
        } else {
            align_to(buffer, get_integer_layout(128).align());
            buffer.extend_from_slice(&self.to_ne_bytes());
        }
    }
}

impl AbiArgument for Felt {
    fn to_bytes(&self, buffer: &mut Vec<u8>) {
        if buffer.len() < 56 {
            buffer.extend_from_slice(&self.to_bytes_le());
        } else {
            align_to(buffer, get_integer_layout(252).align());
            buffer.extend_from_slice(&self.to_bytes_le());
        }
    }
}

impl AbiArgument for U256 {
    fn to_bytes(&self, buffer: &mut Vec<u8>) {
        if buffer.len() < 56 {
            buffer.extend_from_slice(&self.lo.to_le_bytes());
            buffer.extend_from_slice(&self.hi.to_le_bytes());
        } else {
            align_to(buffer, get_integer_layout(256).align());
            buffer.extend_from_slice(&self.lo.to_le_bytes());
            buffer.extend_from_slice(&self.hi.to_le_bytes());
        }
    }
}

impl AbiArgument for [u8; 31] {
    fn to_bytes(&self, buffer: &mut Vec<u8>) {
        if buffer.len() < 56 {
            buffer.extend_from_slice(self);
            buffer.push(0);
        } else {
            align_to(buffer, get_integer_layout(252).align());
            buffer.extend_from_slice(self);
            buffer.push(0);
        }
    }
}

impl<T> AbiArgument for *const T {
    fn to_bytes(&self, buffer: &mut Vec<u8>) {
        <u64 as AbiArgument>::to_bytes(&(*self as u64), buffer)
    }
}

impl<T> AbiArgument for *mut T {
    fn to_bytes(&self, buffer: &mut Vec<u8>) {
        <u64 as AbiArgument>::to_bytes(&(*self as u64), buffer)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn u128_stack_split() {
        let mut buffer = vec![0; 56];
        u128::MAX.to_bytes(&mut buffer);
        assert_eq!(
            buffer,
            [0; 64].into_iter().chain([0xFF; 16]).collect::<Vec<_>>()
        );
    }

    #[test]
    fn felt_stack_split() {
        // Only a single u64 spilled into the stack.
        let mut buffer = vec![0; 40];
        Felt::from_hex("0x00ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
            .unwrap()
            .to_bytes(&mut buffer);
        assert_eq!(
            buffer,
            [0; 40]
                .into_iter()
                .chain([0xFF; 31])
                .chain([0x00])
                .collect::<Vec<_>>()
        );

        // Half the felt spilled into the stack.
        let mut buffer = vec![0; 48];
        Felt::from_hex("0x00ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
            .unwrap()
            .to_bytes(&mut buffer);
        assert_eq!(
            buffer,
            [0; 48]
                .into_iter()
                .chain([0xFF; 31])
                .chain([0x00])
                .collect::<Vec<_>>()
        );

        // All the felt spilled into the stack (with padding).
        let mut buffer = vec![0; 56];
        Felt::from_hex("0x00ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
            .unwrap()
            .to_bytes(&mut buffer);
        assert_eq!(
            buffer,
            [0; 64]
                .into_iter()
                .chain([0xFF; 31])
                .chain([0x00])
                .collect::<Vec<_>>()
        );
    }
}
