use num_bigint::BigInt;

use crate::Value;

/// Receives a vector of values, filters any which is non numeric and returns a `Vec<BigInt>`
/// Useful when a binary operation takes generic values (like with bounded ints).
pub fn get_numberic_args_as_bigints(args: Vec<Value>) -> Vec<BigInt> {
    args.into_iter()
        .filter(|v| !matches!(v, Value::Unit))
        .map(|v| match v {
            Value::BoundedInt { value, .. } => value,
            Value::I8(value) => BigInt::from(value),
            Value::I16(value) => BigInt::from(value),
            Value::I32(value) => BigInt::from(value),
            Value::I64(value) => BigInt::from(value),
            Value::I128(value) => BigInt::from(value),
            Value::U8(value) => BigInt::from(value),
            Value::U16(value) => BigInt::from(value),
            Value::U32(value) => BigInt::from(value),
            Value::U64(value) => BigInt::from(value),
            Value::U128(value) => BigInt::from(value),
            Value::Felt(value) => value.to_bigint(),
            Value::Bytes31(value) => value.to_bigint(),
            value => panic!("{:?}", value),
        })
        .collect()
}
