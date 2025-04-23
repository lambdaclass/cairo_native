use super::EvalAction;
use crate::Value;
use cairo_lang_sierra::{
    extensions::{
        const_type::{
            ConstAsBoxConcreteLibfunc, ConstAsImmediateConcreteLibfunc, ConstConcreteLibfunc,
        },
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
    },
    ids::ConcreteTypeId,
    program::GenericArg,
    program_registry::ProgramRegistry,
};
use smallvec::smallvec;

pub fn eval(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    selector: &ConstConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    match selector {
        ConstConcreteLibfunc::AsBox(info) => eval_as_box(registry, info, args),
        ConstConcreteLibfunc::AsImmediate(info) => eval_as_immediate(registry, info, args),
    }
}

pub fn eval_as_immediate(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &ConstAsImmediateConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [] = args.try_into().unwrap();

    let const_ty = match registry.get_type(&info.const_type).unwrap() {
        CoreTypeConcrete::Const(x) => x,
        _ => unreachable!(),
    };
    EvalAction::NormalBranch(
        0,
        smallvec![inner(registry, &const_ty.inner_ty, &const_ty.inner_data)],
    )
}

pub fn eval_as_box(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &ConstAsBoxConcreteLibfunc,
    args: Vec<Value>,
) -> EvalAction {
    let [] = args.try_into().unwrap();

    let const_ty = match registry.get_type(&info.const_type).unwrap() {
        CoreTypeConcrete::Const(x) => x,
        _ => unreachable!(),
    };
    EvalAction::NormalBranch(
        0,
        smallvec![inner(registry, &const_ty.inner_ty, &const_ty.inner_data)],
    )
}

fn inner(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    type_id: &ConcreteTypeId,
    inner_data: &[GenericArg],
) -> Value {
    match registry.get_type(type_id).unwrap() {
        CoreTypeConcrete::BoundedInt(info) => match inner_data {
            [GenericArg::Type(type_id)] => match registry.get_type(type_id).unwrap() {
                CoreTypeConcrete::Const(info) => inner(registry, &info.inner_ty, &info.inner_data),
                _ => unreachable!(),
            },
            [GenericArg::Value(value)] => {
                assert!(value >= &info.range.lower && value < &info.range.upper);
                Value::BoundedInt {
                    range: info.range.lower.clone()..info.range.upper.clone(),
                    value: value.clone(),
                }
            }
            _ => unreachable!(),
        },
        CoreTypeConcrete::Felt252(_) => match inner_data {
            [GenericArg::Value(value)] => Value::Felt(value.into()),
            _ => unreachable!(),
        },
        CoreTypeConcrete::NonZero(_) => match inner_data {
            [GenericArg::Type(type_id)] => match registry.get_type(type_id).unwrap() {
                CoreTypeConcrete::Const(info) => inner(registry, &info.inner_ty, &info.inner_data),
                _ => unreachable!(),
            },
            _ => unreachable!(),
        },
        CoreTypeConcrete::Sint128(_) => match inner_data {
            [GenericArg::Value(value)] => Value::I128(value.try_into().unwrap()),
            _ => unreachable!(),
        },
        CoreTypeConcrete::Sint64(_) => match inner_data {
            [GenericArg::Value(value)] => Value::U64(value.try_into().unwrap()),
            _ => unreachable!(),
        },
        CoreTypeConcrete::Sint32(_) => match inner_data {
            [GenericArg::Value(value)] => Value::I32(value.try_into().unwrap()),
            _ => unreachable!(),
        },
        CoreTypeConcrete::Sint16(_) => match inner_data {
            [GenericArg::Value(value)] => Value::I16(value.try_into().unwrap()),
            _ => unreachable!(),
        },
        CoreTypeConcrete::Sint8(_) => match inner_data {
            [GenericArg::Value(value)] => Value::I8(value.try_into().unwrap()),
            _ => unreachable!(),
        },
        CoreTypeConcrete::Uint128(_) => match inner_data {
            [GenericArg::Value(value)] => Value::U128(value.try_into().unwrap()),
            [GenericArg::Type(type_id)] => match registry.get_type(type_id).unwrap() {
                CoreTypeConcrete::Const(info) => inner(registry, &info.inner_ty, &info.inner_data),
                _ => unreachable!(),
            },
            _ => unreachable!(),
        },
        CoreTypeConcrete::Uint64(_) => match inner_data {
            [GenericArg::Value(value)] => Value::U64(value.try_into().unwrap()),
            _ => unreachable!(),
        },
        CoreTypeConcrete::Uint32(_) => match inner_data {
            [GenericArg::Value(value)] => Value::U32(value.try_into().unwrap()),
            [GenericArg::Type(type_id)] => match registry.get_type(type_id).unwrap() {
                CoreTypeConcrete::Const(info) => inner(registry, &info.inner_ty, &info.inner_data),
                _ => unreachable!(),
            },
            _ => unreachable!(),
        },
        CoreTypeConcrete::Uint16(_) => match inner_data {
            [GenericArg::Value(value)] => Value::U16(value.try_into().unwrap()),
            _ => unreachable!(),
        },
        CoreTypeConcrete::Uint8(_) => match inner_data {
            [GenericArg::Value(value)] => Value::U8(value.try_into().unwrap()),
            _ => unreachable!(),
        },
        CoreTypeConcrete::Struct(_) => {
            let mut fields = Vec::new();

            for field in inner_data {
                match field {
                    GenericArg::Type(const_field_ty) => {
                        let field_type = registry.get_type(const_field_ty).unwrap();

                        match &field_type {
                            CoreTypeConcrete::Const(const_ty) => {
                                let field_value =
                                    inner(registry, &const_ty.inner_ty, &const_ty.inner_data);
                                fields.push(field_value);
                            }
                            _ => unreachable!(),
                        };
                    }
                    _ => unreachable!(),
                }
            }

            Value::Struct(fields)
        }
        _ => todo!("{}", type_id),
    }
}
