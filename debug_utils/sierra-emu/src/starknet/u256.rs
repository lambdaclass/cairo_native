use crate::Value;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct U256 {
    pub lo: u128,
    pub hi: u128,
}

impl U256 {
    #[allow(unused)]
    pub(crate) fn into_value(self) -> Value {
        Value::Struct(vec![Value::U128(self.lo), Value::U128(self.hi)])
    }

    pub fn from_value(v: Value) -> Self {
        let Value::Struct(v) = v else { panic!() };
        let Value::U128(lo) = v[0] else { panic!() };
        let Value::U128(hi) = v[1] else { panic!() };

        Self { lo, hi }
    }
}
