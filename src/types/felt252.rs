//! # `felt252` type
//!
//! A `felt252` is a 252-bit number within a
//! [finite field](https://en.wikipedia.org/wiki/Finite_field) modulo
//! [a prime number](struct@self::PRIME).

use super::{TypeBuilder, WithSelf};
use crate::{
    error::types::{Error, Result},
    metadata::{prime_modulo::PrimeModuloMeta, MetadataStorage},
};
use cairo_lang_sierra::{
    extensions::{types::InfoOnlyConcreteType, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry,
};
use lazy_static::lazy_static;
use melior::{
    ir::{r#type::IntegerType, Module, Type},
    Context,
};
use num_bigint::BigUint;

lazy_static! {
    /// The `felt252` prime modulo.
    pub static ref PRIME: BigUint =
        "3618502788666131213697322783095070105623107215331596699973092056135872020481"
            .parse()
            .unwrap();
}

/// Marker type for the [PrimeModuloMeta] metadata.
// TODO: Maybe we should use the JIT value (in `crate::values::felt252`) instead of defining a dummy
//   type?
#[derive(Debug)]
pub struct Felt252;

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    metadata: &mut MetadataStorage,
    _info: WithSelf<InfoOnlyConcreteType>,
) -> Result<Type<'ctx>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = Error>,
{
    match metadata.get::<PrimeModuloMeta<Felt252>>() {
        Some(x) => assert_eq!(x.prime(), &*PRIME),
        None => {
            register_prime_modulo_meta(metadata);
        }
    }

    Ok(IntegerType::new(context, 252).into())
}

pub fn register_prime_modulo_meta(metadata: &mut MetadataStorage) -> &mut PrimeModuloMeta<Felt252> {
    metadata
        .insert(PrimeModuloMeta::<Felt252>::new(PRIME.clone()))
        .unwrap()
}
