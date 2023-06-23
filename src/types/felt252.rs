//! # `felt252` type
//!
//! A `felt252` is a 252-bit number within a
//! [finite field](https://en.wikipedia.org/wiki/Finite_field) modulo
//! [a prime number](struct@self::PRIME).

use super::TypeBuilder;
use crate::metadata::{prime_modulo::PrimeModulo, MetadataStorage};
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

/// Marker type for the [PrimeModulo] metadata.
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
    _info: &InfoOnlyConcreteType,
) -> Result<Type<'ctx>, std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
{
    match metadata.get::<PrimeModulo<Felt252>>() {
        Some(x) => assert_eq!(x.prime(), &*PRIME),
        None => {
            metadata
                .insert(PrimeModulo::<Felt252>::new(PRIME.clone()))
                .unwrap();
        }
    }

    Ok(IntegerType::new(context, 252).into())
}
