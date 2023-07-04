//! # Boolean libfuncs

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::MetadataStorage,
    types::TypeBuilder,
};
use cairo_lang_sierra::{
    extensions::{
        boolean::BoolConcreteLibfunc, lib_func::SignatureOnlyConcreteLibfunc, ConcreteLibfunc,
        GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith, llvm},
    ir::{attribute::DenseI64ArrayAttribute, r#type::IntegerType, Attribute, Block, Location},
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &BoolConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        BoolConcreteLibfunc::And(info) => build_bool_binary(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            info,
            BoolOp::And,
        ),
        BoolConcreteLibfunc::Not(info) => {
            build_bool_not(context, registry, entry, location, helper, metadata, info)
        }
        BoolConcreteLibfunc::Xor(info) => build_bool_binary(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            info,
            BoolOp::Xor,
        ),
        BoolConcreteLibfunc::Or(info) => build_bool_binary(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            info,
            BoolOp::Or,
        ),
        BoolConcreteLibfunc::ToFelt252(_) => todo!(),
    }
}

#[derive(Debug, Clone, Copy)]
enum BoolOp {
    And,
    Xor,
    Or,
}

/// Generate MLIR operations for the `bool_not_impl` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_bool_binary<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
    binop: BoolOp,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let enum_ty = registry.get_type(&info.param_signatures()[0].ty)?;
    let tag_bits = enum_ty
        .variants()
        .unwrap()
        .len()
        .next_power_of_two()
        .trailing_zeros();
    let tag_ty = IntegerType::new(context, tag_bits).into();

    let lhs = entry.argument(0)?.into();
    let rhs = entry.argument(0)?.into();

    let op = entry.append_operation(llvm::extract_value(
        context,
        lhs,
        DenseI64ArrayAttribute::new(context, &[0]),
        tag_ty,
        location,
    ));
    let lhs_tag = op.result(0)?.into();
    let op = entry.append_operation(llvm::extract_value(
        context,
        rhs,
        DenseI64ArrayAttribute::new(context, &[0]),
        tag_ty,
        location,
    ));
    let rhs_tag = op.result(0)?.into();

    let op = match binop {
        BoolOp::And => entry.append_operation(arith::andi(lhs_tag, rhs_tag, location)),
        BoolOp::Xor => entry.append_operation(arith::xori(lhs_tag, rhs_tag, location)),
        BoolOp::Or => entry.append_operation(arith::ori(lhs_tag, rhs_tag, location)),
    };
    let new_tag_value = op.result(0)?.into();

    let op = entry.append_operation(llvm::insert_value(
        context,
        lhs,
        DenseI64ArrayAttribute::new(context, &[0]),
        new_tag_value,
        location,
    ));
    let value = op.result(0)?.into();

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

/// Generate MLIR operations for the `bool_not_impl` libfunc.
pub fn build_bool_not<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let enum_ty = registry.get_type(&info.param_signatures()[0].ty)?;
    let tag_bits = enum_ty
        .variants()
        .unwrap()
        .len()
        .next_power_of_two()
        .trailing_zeros();
    let tag_ty = IntegerType::new(context, tag_bits).into();

    let value = entry.argument(0)?.into();

    let op = entry.append_operation(llvm::extract_value(
        context,
        value,
        DenseI64ArrayAttribute::new(context, &[0]),
        tag_ty,
        location,
    ));
    let tag_value = op.result(0)?.into();

    let op = entry.append_operation(arith::constant(
        context,
        Attribute::parse(context, &format!("1 : {tag_ty}")).unwrap(),
        location,
    ));
    let const_1 = op.result(0)?.into();

    let op = entry.append_operation(arith::xori(tag_value, const_1, location));
    let new_tag_value = op.result(0)?.into();

    let op = entry.append_operation(llvm::insert_value(
        context,
        value,
        DenseI64ArrayAttribute::new(context, &[0]),
        new_tag_value,
        location,
    ));
    let value = op.result(0)?.into();

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::{load_cairo, run_program};
    use serde_json::json;

    #[test]
    fn run_not() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> bool {
                true == true
            }
        );
        let result = run_program(&program, "run_test", json!([]));

        assert_eq!(result, json!([[1, []]]));

        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> bool {
                true == false
            }
        );
        let result = run_program(&program, "run_test", json!([]));

        assert_eq!(result, json!([[0, []]]));
    }

    #[test]
    fn run_and() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> bool {
                true && true
            }
        );
        let result = run_program(&program, "run_test", json!([]));

        assert_eq!(result, json!([[1, []]]));

        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> bool {
                true && false
            }
        );
        let result = run_program(&program, "run_test", json!([]));

        assert_eq!(result, json!([[0, []]]));
    }

    #[test]
    fn run_xor() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> bool {
                true ^ true
            }
        );
        let result = run_program(&program, "run_test", json!([]));

        assert_eq!(result, json!([[0, []]]));

        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> bool {
                true ^ false
            }
        );
        let result = run_program(&program, "run_test", json!([]));

        assert_eq!(result, json!([[0, []]]));
    }

    #[test]
    fn run_or() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> bool {
                true || true
            }
        );
        let result = run_program(&program, "run_test", json!([]));

        assert_eq!(result, json!([[1, []]]));

        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> bool {
                true || false
            }
        );
        let result = run_program(&program, "run_test", json!([]));

        assert_eq!(result, json!([[1, []]]));
    }
}
