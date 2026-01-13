use std::{error::Error, io::Write};

use arbitrary::{Arbitrary, Unstructured};
use cairo_lang_sierra::extensions::{
    circuit::CircuitTypeConcrete, core::CoreTypeConcrete, starknet::StarknetTypeConcrete,
    ConcreteType,
};
use cairo_native::Value;
use starknet_types_core::felt::Felt;

pub fn is_builtin(ty: &CoreTypeConcrete) -> bool {
    matches!(
        ty,
        CoreTypeConcrete::Bitwise(_)
            | CoreTypeConcrete::EcOp(_)
            | CoreTypeConcrete::GasBuiltin(_)
            | CoreTypeConcrete::BuiltinCosts(_)
            | CoreTypeConcrete::RangeCheck(_)
            | CoreTypeConcrete::RangeCheck96(_)
            | CoreTypeConcrete::Pedersen(_)
            | CoreTypeConcrete::Poseidon(_)
            | CoreTypeConcrete::Coupon(_)
            | CoreTypeConcrete::Starknet(StarknetTypeConcrete::System(_))
            | CoreTypeConcrete::SegmentArena(_)
            | CoreTypeConcrete::Circuit(CircuitTypeConcrete::AddMod(_))
            | CoreTypeConcrete::Circuit(CircuitTypeConcrete::MulMod(_))
    )
}

pub fn is_supported(ty: &CoreTypeConcrete) -> bool {
    matches!(
        ty,
        CoreTypeConcrete::Felt252(_)
            | CoreTypeConcrete::Uint8(_)
            | CoreTypeConcrete::Uint16(_)
            | CoreTypeConcrete::Uint32(_)
            | CoreTypeConcrete::Uint64(_)
            | CoreTypeConcrete::Uint128(_)
            | CoreTypeConcrete::Sint8(_)
            | CoreTypeConcrete::Sint16(_)
            | CoreTypeConcrete::Sint32(_)
            | CoreTypeConcrete::Sint64(_)
            | CoreTypeConcrete::Sint128(_)
    )
}

pub fn random_value(ty: &CoreTypeConcrete) -> Value {
    match ty {
        CoreTypeConcrete::Felt252(_) => {
            Value::Felt252(Felt::from_bytes_le(&rand::random::<[u8; 32]>()))
        }
        CoreTypeConcrete::Uint8(_) => Value::Uint8(rand::random()),
        CoreTypeConcrete::Uint16(_) => Value::Uint16(rand::random()),
        CoreTypeConcrete::Uint32(_) => Value::Uint32(rand::random()),
        CoreTypeConcrete::Uint64(_) => Value::Uint64(rand::random()),
        CoreTypeConcrete::Uint128(_) => Value::Uint128(rand::random()),
        CoreTypeConcrete::Sint8(_) => Value::Sint8(rand::random()),
        CoreTypeConcrete::Sint16(_) => Value::Sint16(rand::random()),
        CoreTypeConcrete::Sint32(_) => Value::Sint32(rand::random()),
        CoreTypeConcrete::Sint64(_) => Value::Sint64(rand::random()),
        CoreTypeConcrete::Sint128(_) => Value::Sint128(rand::random()),
        x => todo!("random {:?}", x.info()),
    }
}

pub fn arbitrary_value(
    ty: &CoreTypeConcrete,
    u: &mut Unstructured,
) -> Result<Value, Box<dyn Error>> {
    Ok(match ty {
        CoreTypeConcrete::Felt252(_) => {
            Value::Felt252(Felt::from_bytes_le(&Arbitrary::arbitrary(u)?))
        }
        CoreTypeConcrete::Uint128(_) => {
            Value::Uint128(u128::from_le_bytes(Arbitrary::arbitrary(u)?))
        }
        CoreTypeConcrete::Sint128(_) => {
            Value::Sint128(i128::from_le_bytes(Arbitrary::arbitrary(u)?))
        }
        CoreTypeConcrete::Uint8(_) => Value::Uint8(u8::from_le_bytes(Arbitrary::arbitrary(u)?)),
        CoreTypeConcrete::Uint16(_) => Value::Uint16(u16::from_le_bytes(Arbitrary::arbitrary(u)?)),
        CoreTypeConcrete::Uint32(_) => Value::Uint32(u32::from_le_bytes(Arbitrary::arbitrary(u)?)),
        CoreTypeConcrete::Uint64(_) => Value::Uint64(u64::from_le_bytes(Arbitrary::arbitrary(u)?)),
        CoreTypeConcrete::Sint8(_) => Value::Sint8(i8::from_le_bytes(Arbitrary::arbitrary(u)?)),
        CoreTypeConcrete::Sint16(_) => Value::Sint16(i16::from_le_bytes(Arbitrary::arbitrary(u)?)),
        CoreTypeConcrete::Sint32(_) => Value::Sint32(i32::from_le_bytes(Arbitrary::arbitrary(u)?)),
        CoreTypeConcrete::Sint64(_) => Value::Sint64(i64::from_le_bytes(Arbitrary::arbitrary(u)?)),
        x => todo!("random {:?}", x.info()),
    })
}

pub fn encode_value(v: &Value, mut w: impl Write) -> Result<(), Box<dyn Error>> {
    match v {
        Value::Felt252(v) => w.write_all(&v.to_bytes_le())?,
        Value::Uint8(v) => w.write_all(&v.to_le_bytes())?,
        Value::Uint16(v) => w.write_all(&v.to_le_bytes())?,
        Value::Uint32(v) => w.write_all(&v.to_le_bytes())?,
        Value::Uint64(v) => w.write_all(&v.to_le_bytes())?,
        Value::Uint128(v) => w.write_all(&v.to_le_bytes())?,
        Value::Sint8(v) => w.write_all(&v.to_le_bytes())?,
        Value::Sint16(v) => w.write_all(&v.to_le_bytes())?,
        Value::Sint32(v) => w.write_all(&v.to_le_bytes())?,
        Value::Sint64(v) => w.write_all(&v.to_le_bytes())?,
        Value::Sint128(v) => w.write_all(&v.to_le_bytes())?,
        x => todo!("encode {:?}", x),
    };

    Ok(())
}
