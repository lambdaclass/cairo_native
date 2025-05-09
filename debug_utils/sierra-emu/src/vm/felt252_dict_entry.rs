use super::EvalAction;
use crate::Value;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        felt252_dict::Felt252DictEntryConcreteLibfunc,
        lib_func::SignatureAndTypeConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use smallvec::smallvec;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &Felt252DictEntryConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        Felt252DictEntryConcreteLibfunc::Get(info) => eval_get(registry, info, args),
        Felt252DictEntryConcreteLibfunc::Finalize(info) => eval_finalize(registry, info, args),
    }
}

pub fn eval_get(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::FeltDict {
        ty,
        mut data,
        mut count,
    }, Value::Felt(key)]: [Value; 2] = args.try_into().unwrap()
    else {
        panic!()
    };
    assert_eq!(info.ty, ty);

    let value = data
        .entry(key)
        .or_insert_with(|| {
            count += 1;

            Value::default_for_type(registry, &ty)
        })
        .to_owned();

    EvalAction::NormalBranch(0, smallvec![Value::FeltDict { ty, data, count }, value])
}

pub fn eval_finalize(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureAndTypeConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [Value::FeltDictEntry {
        ty,
        mut data,
        count,
        key,
    }, value]: [Value; 2] = args.try_into().unwrap()
    else {
        panic!()
    };
    assert_eq!(info.ty, ty);
    assert!(value.is(registry, &ty));

    data.insert(key, value);

    EvalAction::NormalBranch(0, smallvec![Value::FeltDict { ty, data, count }])
}
