use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        utils::Range,
    },
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use num_bigint::BigInt;
use num_traits::{Bounded, One, ToPrimitive};
use starknet_types_core::felt::CAIRO_PRIME_BIGINT;

use crate::Value;

/// Receives a vector of values, filters any which is non numeric and returns a `Vec<BigInt>`
/// Useful when a binary operation takes generic values (like with bounded ints).
pub fn get_numeric_args_as_bigints(args: &[Value]) -> Vec<BigInt> {
    args.iter()
        .map(|v| match v {
            Value::BoundedInt { value, .. } => value.to_owned(),
            Value::I8(value) => BigInt::from(*value),
            Value::I16(value) => BigInt::from(*value),
            Value::I32(value) => BigInt::from(*value),
            Value::I64(value) => BigInt::from(*value),
            Value::I128(value) => BigInt::from(*value),
            Value::U8(value) => BigInt::from(*value),
            Value::U16(value) => BigInt::from(*value),
            Value::U32(value) => BigInt::from(*value),
            Value::U64(value) => BigInt::from(*value),
            Value::U128(value) => BigInt::from(*value),
            Value::Felt(value) => value.to_bigint(),
            Value::Bytes31(value) => value.to_bigint(),
            value => panic!("argument should be an integer: {:?}", value),
        })
        .collect()
}

pub fn get_value_from_integer(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    ty_id: &ConcreteTypeId,
    value: BigInt,
) -> Value {
    let ty = registry.get_type(ty_id).unwrap();

    match ty {
        CoreTypeConcrete::NonZero(info) => get_value_from_integer(registry, &info.ty, value),
        CoreTypeConcrete::Sint8(_) => Value::I8(value.to_i8().unwrap()),
        CoreTypeConcrete::Sint16(_) => Value::I16(value.to_i16().unwrap()),
        CoreTypeConcrete::Sint32(_) => Value::I32(value.to_i32().unwrap()),
        CoreTypeConcrete::Sint64(_) => Value::I64(value.to_i64().unwrap()),
        CoreTypeConcrete::Sint128(_) => Value::I128(value.to_i128().unwrap()),
        CoreTypeConcrete::Uint8(_) => Value::U8(value.to_u8().unwrap()),
        CoreTypeConcrete::Uint16(_) => Value::U16(value.to_u16().unwrap()),
        CoreTypeConcrete::Uint32(_) => Value::U32(value.to_u32().unwrap()),
        CoreTypeConcrete::Uint64(_) => Value::U64(value.to_u64().unwrap()),
        CoreTypeConcrete::Uint128(_) => Value::U128(value.to_u128().unwrap()),
        _ => panic!("cannot get integer value for a non-integer type"),
    }
}

pub fn integer_range(
    ty: &ConcreteTypeId,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
) -> Range {
    fn range_of<T>() -> Range
    where
        T: Bounded + Into<BigInt>,
    {
        Range {
            lower: T::min_value().into(),
            upper: T::max_value().into() + BigInt::one(),
        }
    }

    let ty = registry.get_type(ty).unwrap();

    match ty {
        CoreTypeConcrete::Uint8(_) => range_of::<u8>(),
        CoreTypeConcrete::Uint16(_) => range_of::<u16>(),
        CoreTypeConcrete::Uint32(_) => range_of::<u32>(),
        CoreTypeConcrete::Uint64(_) => range_of::<u64>(),
        CoreTypeConcrete::Uint128(_) => range_of::<u128>(),
        CoreTypeConcrete::Felt252(_) => Range {
            lower: BigInt::ZERO,
            upper: CAIRO_PRIME_BIGINT.clone(),
        },
        CoreTypeConcrete::Sint8(_) => range_of::<i8>(),
        CoreTypeConcrete::Sint16(_) => range_of::<i16>(),
        CoreTypeConcrete::Sint32(_) => range_of::<i32>(),
        CoreTypeConcrete::Sint64(_) => range_of::<i64>(),
        CoreTypeConcrete::Sint128(_) => range_of::<i128>(),
        CoreTypeConcrete::BoundedInt(info) => info.range.clone(),
        CoreTypeConcrete::Bytes31(_) => Range {
            lower: BigInt::ZERO,
            upper: BigInt::one() << 248,
        },
        CoreTypeConcrete::Const(info) => integer_range(&info.inner_ty, registry),
        CoreTypeConcrete::NonZero(info) => integer_range(&info.ty, registry),
        _ => panic!("cannot get integer range value for a non-integer type"),
    }
}
