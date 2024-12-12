use super::U256;
use crate::Value;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Secp256r1Point {
    pub x: U256,
    pub y: U256,
}

impl Secp256r1Point {
    #[allow(unused)]
    pub fn into_value(self) -> Value {
        Value::Struct(vec![
            Value::Struct(vec![Value::U128(self.x.lo), Value::U128(self.x.hi)]),
            Value::Struct(vec![Value::U128(self.y.lo), Value::U128(self.y.hi)]),
        ])
    }

    pub fn from_value(v: Value) -> Self {
        let Value::Struct(mut v) = v else { panic!() };

        let y = U256::from_value(v.remove(1));
        let x = U256::from_value(v.remove(0));

        Self { x, y }
    }
}
