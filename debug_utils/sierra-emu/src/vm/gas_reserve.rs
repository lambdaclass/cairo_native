use cairo_lang_sierra::extensions::gas_reserve::GasReserveConcreteLibfunc;
use smallvec::smallvec;

use crate::{vm::EvalAction, Value};

pub fn eval(selector: &GasReserveConcreteLibfunc, args: Vec<Value>) -> EvalAction {
    match selector {
        GasReserveConcreteLibfunc::Create(_) => eval_gas_reserve_create(args),
        GasReserveConcreteLibfunc::Utilize(_) => eval_gas_reserve_utilize(args),
    }
}

fn eval_gas_reserve_create(args: Vec<Value>) -> EvalAction {
    let [range_check @ Value::Unit, Value::U64(gas), Value::U128(amount)]: [Value; 3] =
        args.try_into().unwrap()
    else {
        panic!()
    };

    if amount <= gas.into() {
        let spare_gas = gas - amount as u64;
        EvalAction::NormalBranch(
            0,
            smallvec![range_check, Value::U64(spare_gas), Value::U128(amount)],
        )
    } else {
        EvalAction::NormalBranch(1, smallvec![range_check, Value::U64(gas)])
    }
}

fn eval_gas_reserve_utilize(args: Vec<Value>) -> EvalAction {
    let [Value::U64(gas), Value::U128(amount)]: [Value; 2] = args.try_into().unwrap() else {
        panic!()
    };

    let updated_gas = gas + amount as u64;

    EvalAction::NormalBranch(0, smallvec![Value::U64(updated_gas)])
}
