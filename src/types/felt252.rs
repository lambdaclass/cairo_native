////! # `felt252` type
//! # `felt252` type
////!
//!
////! A `felt252` is a 252-bit number within a
//! A `felt252` is a 252-bit number within a
////! [finite field](https://en.wikipedia.org/wiki/Finite_field) modulo
//! [finite field](https://en.wikipedia.org/wiki/Finite_field) modulo
////! [a prime number](struct@self::PRIME).
//! [a prime number](struct@self::PRIME).
//

//use super::WithSelf;
use super::WithSelf;
//use crate::{
use crate::{
//    error::Result,
    error::Result,
//    metadata::{prime_modulo::PrimeModuloMeta, MetadataStorage},
    metadata::{prime_modulo::PrimeModuloMeta, MetadataStorage},
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        types::InfoOnlyConcreteType,
        types::InfoOnlyConcreteType,
//    },
    },
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use lazy_static::lazy_static;
use lazy_static::lazy_static;
//use melior::{
use melior::{
//    ir::{r#type::IntegerType, Module, Type},
    ir::{r#type::IntegerType, Module, Type},
//    Context,
    Context,
//};
};
//use num_bigint::{BigInt, BigUint};
use num_bigint::{BigInt, BigUint};
//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
//

//lazy_static! {
lazy_static! {
//    /// The `felt252` prime modulo.
    /// The `felt252` prime modulo.
//    pub static ref PRIME: BigUint =
    pub static ref PRIME: BigUint =
//        "3618502788666131213697322783095070105623107215331596699973092056135872020481"
        "3618502788666131213697322783095070105623107215331596699973092056135872020481"
//            .parse()
            .parse()
//            .unwrap();
            .unwrap();
//    pub static ref HALF_PRIME: BigInt =
    pub static ref HALF_PRIME: BigInt =
//        "1809251394333065606848661391547535052811553607665798349986546028067936010240"
        "1809251394333065606848661391547535052811553607665798349986546028067936010240"
//            .parse()
            .parse()
//            .unwrap();
            .unwrap();
//}
}
//

///// Build the MLIR type.
/// Build the MLIR type.
/////
///
///// Check out [the module](self) for more info.
/// Check out [the module](self) for more info.
//pub fn build<'ctx>(
pub fn build<'ctx>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _module: &Module<'ctx>,
    _module: &Module<'ctx>,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    _info: WithSelf<InfoOnlyConcreteType>,
    _info: WithSelf<InfoOnlyConcreteType>,
//) -> Result<Type<'ctx>> {
) -> Result<Type<'ctx>> {
//    match metadata.get::<PrimeModuloMeta<Felt>>() {
    match metadata.get::<PrimeModuloMeta<Felt>>() {
//        Some(x) => assert_eq!(x.prime(), &*PRIME),
        Some(x) => assert_eq!(x.prime(), &*PRIME),
//        None => {
        None => {
//            register_prime_modulo_meta(metadata);
            register_prime_modulo_meta(metadata);
//        }
        }
//    }
    }
//

//    Ok(IntegerType::new(context, 252).into())
    Ok(IntegerType::new(context, 252).into())
//}
}
//

//pub fn register_prime_modulo_meta(metadata: &mut MetadataStorage) -> &mut PrimeModuloMeta<Felt> {
pub fn register_prime_modulo_meta(metadata: &mut MetadataStorage) -> &mut PrimeModuloMeta<Felt> {
//    metadata
    metadata
//        .insert(PrimeModuloMeta::<Felt>::new(PRIME.clone()))
        .insert(PrimeModuloMeta::<Felt>::new(PRIME.clone()))
//        .unwrap()
        .unwrap()
//}
}
