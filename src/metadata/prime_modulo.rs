use num_bigint::BigUint;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct PrimeModulo<T> {
    prime: BigUint,
    phantom: PhantomData<T>,
}

impl<T> PrimeModulo<T> {
    pub fn new(prime: BigUint) -> Self {
        Self {
            prime,
            phantom: PhantomData,
        }
    }

    pub fn prime(&self) -> &BigUint {
        &self.prime
    }
}
