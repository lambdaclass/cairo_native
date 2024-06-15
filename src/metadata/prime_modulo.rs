//! # Finite field prime modulo
//!
//! Contains the prime modulo number of a finite field.
//!
//! Currently used only for `Felt`.

use num_bigint::BigUint;
use std::marker::PhantomData;

/// Prime modulo number metadata.
#[derive(Debug)]
pub struct PrimeModuloMeta<T> {
    prime: BigUint,
    phantom: PhantomData<T>,
}

impl<T> PrimeModuloMeta<T> {
    /// Create the metadata from the prime number.
    pub fn new(prime: BigUint) -> Self {
        Self {
            prime,
            phantom: PhantomData,
        }
    }

    /// Return the stored prime number.
    pub fn prime(&self) -> &BigUint {
        &self.prime
    }
}
// PLT: it's not yet clear to me what this is used for. Need to check.
// PLT: ACK
