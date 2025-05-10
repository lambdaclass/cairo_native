use super::EvalAction;
use crate::{
    vm::uint256::{u256_to_biguint, u256_to_value, u516_to_value},
    Value,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        int::unsigned512::Uint512Concrete,
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use smallvec::smallvec;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &Uint512Concrete,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        Uint512Concrete::DivModU256(info) => eval_divmod(registry, info, args),
    }
}

pub fn eval_divmod(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [range_check @ Value::Unit, Value::Struct(lhs), Value::Struct(rhs)]: [Value; 3] =
        args.try_into().unwrap()
    else {
        panic!()
    };

    let [Value::U128(div_0), Value::U128(div_1), Value::U128(div_2), Value::U128(div_3)]: [Value;
        4] = lhs.try_into().unwrap()
    else {
        panic!()
    };

    let lhs = u256_to_biguint(div_0, div_1) | (u256_to_biguint(div_2, div_3) << 256);

    let [Value::U128(divisor_0), Value::U128(divisor_1)]: [Value; 2] = rhs.try_into().unwrap()
    else {
        panic!()
    };

    let rhs = u256_to_biguint(divisor_0, divisor_1);

    let div = &lhs / &rhs;
    let modulo = lhs % rhs;

    EvalAction::NormalBranch(
        0,
        smallvec![
            range_check,
            u516_to_value(div),
            u256_to_value(modulo),
            Value::Unit,
            Value::Unit,
            Value::Unit,
            Value::Unit,
            Value::Unit,
        ],
    )
}
