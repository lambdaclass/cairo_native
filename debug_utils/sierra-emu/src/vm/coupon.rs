use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        coupon::CouponConcreteLibfunc,
        function_call::SignatureAndFunctionConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use smallvec::smallvec;

use super::EvalAction;
use crate::Value;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &CouponConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        CouponConcreteLibfunc::Buy(info) => eval_buy(registry, info, args),
        CouponConcreteLibfunc::Refund(info) => eval_refund(registry, info, args),
    }
}

fn eval_buy(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureAndFunctionConcreteLibfunc,
    _args: Vec<Value>,
) -> EvalAction {
    EvalAction::NormalBranch(0, smallvec![Value::Unit])
}

fn eval_refund(
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _info: &SignatureAndFunctionConcreteLibfunc,
    _args: Vec<Value>,
) -> EvalAction {
    EvalAction::NormalBranch(0, smallvec![])
}
