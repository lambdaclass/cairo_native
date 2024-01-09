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
use num_bigint::{BigInt, BigUint};
use starknet_types_core::felt::Felt;

lazy_static! {
    /// The `felt252` prime modulo.
    pub static ref PRIME: BigUint =
        "3618502788666131213697322783095070105623107215331596699973092056135872020481"
            .parse()
            .unwrap();
    pub static ref HALF_PRIME: BigInt =
        "1809251394333065606848661391547535052811553607665798349986546028067936010240"
            .parse()
            .unwrap();
}

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
    match metadata.get::<PrimeModuloMeta<Felt>>() {
        Some(x) => assert_eq!(x.prime(), &*PRIME),
        None => {
            register_prime_modulo_meta(metadata);
        }
    }

    Ok(IntegerType::new(context, 252).into())
}

pub fn register_prime_modulo_meta(metadata: &mut MetadataStorage) -> &mut PrimeModuloMeta<Felt> {
    metadata
        .insert(PrimeModuloMeta::<Felt>::new(PRIME.clone()))
        .unwrap()
}
