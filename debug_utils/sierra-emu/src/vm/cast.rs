use super::EvalAction;
use crate::{utils::get_numberic_args_as_bigints, Value};
use cairo_lang_sierra::{
    extensions::{
        casts::{CastConcreteLibfunc, DowncastConcreteLibfunc},
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteType,
    },
    program_registry::ProgramRegistry,
};
use smallvec::smallvec;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &CastConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        CastConcreteLibfunc::Downcast(info) => eval_downcast(registry, info, args),
        CastConcreteLibfunc::Upcast(info) => eval_upcast(registry, info, args),
    }
}

pub fn eval_downcast(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &DowncastConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value] = get_numberic_args_as_bigints(args).try_into().unwrap();

    let range = info.to_range.lower.clone()..info.to_range.upper.clone();
    if range.contains(&value) {
        EvalAction::NormalBranch(
            0,
            smallvec![
                Value::Unit, // range_check
                match registry.get_type(&info.to_ty).unwrap() {
                    CoreTypeConcrete::Sint8(_) => Value::I8(value.try_into().unwrap()),
                    CoreTypeConcrete::Sint128(_) => Value::I128(value.try_into().unwrap()),
                    CoreTypeConcrete::Uint8(_) => Value::U8(value.try_into().unwrap()),
                    CoreTypeConcrete::Uint16(_) => Value::U16(value.try_into().unwrap()),
                    CoreTypeConcrete::Uint32(_) => Value::U32(value.try_into().unwrap()),
                    CoreTypeConcrete::Uint64(_) => Value::U64(value.try_into().unwrap()),
                    CoreTypeConcrete::Uint128(_) => Value::U128(value.try_into().unwrap()),
                    CoreTypeConcrete::Felt252(_) => Value::Felt(value.into()),
                    CoreTypeConcrete::BoundedInt(_) => Value::BoundedInt { range, value },
                    x => todo!("{:?}", x.info()),
                }
            ],
        )
    } else {
        EvalAction::NormalBranch(1, smallvec![Value::Unit])
    }
}

pub fn eval_upcast(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [value] = get_numberic_args_as_bigints(args).try_into().unwrap();

    EvalAction::NormalBranch(
        0,
        smallvec![match registry
            .get_type(&info.signature.branch_signatures[0].vars[0].ty)
            .unwrap()
        {
            CoreTypeConcrete::Sint8(_) => Value::I8(value.try_into().unwrap()),
            CoreTypeConcrete::Sint16(_) => Value::U16(value.try_into().unwrap()),
            CoreTypeConcrete::Sint32(_) => Value::I32(value.try_into().unwrap()),
            CoreTypeConcrete::Sint64(_) => Value::U64(value.try_into().unwrap()),
            CoreTypeConcrete::Sint128(_) => Value::I128(value.try_into().unwrap()),
            CoreTypeConcrete::Uint8(_) => Value::U8(value.try_into().unwrap()),
            CoreTypeConcrete::Uint16(_) => Value::U16(value.try_into().unwrap()),
            CoreTypeConcrete::Uint32(_) => Value::U32(value.try_into().unwrap()),
            CoreTypeConcrete::Uint64(_) => Value::U64(value.try_into().unwrap()),
            CoreTypeConcrete::Uint128(_) => Value::U128(value.try_into().unwrap()),
            CoreTypeConcrete::Felt252(_) => Value::Felt(value.into()),
            CoreTypeConcrete::BoundedInt(info) => {
                Value::BoundedInt {
                    range: info.range.lower.clone()..info.range.upper.clone(),
                    value,
                }
            }
            _ => todo!(),
        }],
    )
}
