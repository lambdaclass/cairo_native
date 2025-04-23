use super::EvalAction;
use crate::Value;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        function_call::SignatureAndFunctionConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};

pub fn eval(
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
