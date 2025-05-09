use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType, CoreTypeConcrete},
    program_registry::ProgramRegistry,
};
use num_bigint::BigInt;
use num_traits::ToPrimitive;

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
    ty: &CoreTypeConcrete,
    value: BigInt,
) -> Value {
    match ty {
        CoreTypeConcrete::NonZero(info) => {
            let ty = registry.get_type(&info.ty).unwrap();
            get_value_from_integer(registry, ty, value)
        }
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
        CoreTypeConcrete::BoundedInt(info) => {
            let range = &info.range;
            Value::BoundedInt {
                range: range.lower.clone()..range.upper.clone(),
                value,
            }
        }
        CoreTypeConcrete::Felt252(_) => Value::Felt(value.into()),
        _ => panic!("cannot get integer value for a non-integer type"),
    }
}
