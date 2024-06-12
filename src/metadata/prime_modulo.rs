////! # Finite field prime modulo
//! # Finite field prime modulo
////!
//!
////! Contains the prime modulo number of a finite field.
//! Contains the prime modulo number of a finite field.
////!
//!
////! Currently used only for `Felt`.
//! Currently used only for `Felt`.
//

//use num_bigint::BigUint;
use num_bigint::BigUint;
//use std::marker::PhantomData;
use std::marker::PhantomData;
//

///// Prime modulo number metadata.
/// Prime modulo number metadata.
//#[derive(Debug)]
#[derive(Debug)]
//pub struct PrimeModuloMeta<T> {
pub struct PrimeModuloMeta<T> {
//    prime: BigUint,
    prime: BigUint,
//    phantom: PhantomData<T>,
    phantom: PhantomData<T>,
//}
}
//

//impl<T> PrimeModuloMeta<T> {
impl<T> PrimeModuloMeta<T> {
//    /// Create the metadata from the prime number.
    /// Create the metadata from the prime number.
//    pub fn new(prime: BigUint) -> Self {
    pub fn new(prime: BigUint) -> Self {
//        Self {
        Self {
//            prime,
            prime,
//            phantom: PhantomData,
            phantom: PhantomData,
//        }
        }
//    }
    }
//

//    /// Return the stored prime number.
    /// Return the stored prime number.
//    pub fn prime(&self) -> &BigUint {
    pub fn prime(&self) -> &BigUint {
//        &self.prime
        &self.prime
//    }
    }
//}
}
