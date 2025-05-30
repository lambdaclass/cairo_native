use super::EvalAction;
use crate::Value;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        function_call::SignatureAndFunctionConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};

pub fn eval_function_call(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureAndFunctionConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    assert_eq!(args.len(), info.function.params.len());
    assert!(args
        .iter()
        .zip(&info.function.params)
        .all(|(value, param)| value.is(registry, &param.ty)));

    EvalAction::FunctionCall(info.function.id.clone(), args.into_iter().collect())
}

pub fn eval_coupon_call(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureAndFunctionConcreteLibfunc,
    mut args: Vec<Value>,
) -> EvalAction {
    // Don't check the last arg since it is not actually an argument for the function call itself
    assert_eq!(args.len() - 1, info.function.params.len());
    assert!(args
        .iter()
        .zip(&info.function.params)
        .all(|(value, param)| value.is(registry, &param.ty)));

    args.pop();

    EvalAction::FunctionCall(info.function.id.clone(), args.into_iter().collect())
}
