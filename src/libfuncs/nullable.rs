//! # Nullable libfuncs

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
        lib_func::{SignatureAndTypeConcreteLibfunc, SignatureOnlyConcreteLibfunc},
        nullable::NullableConcreteLibfunc,
        ConcreteLibfunc, GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        cf,
        llvm::{self, r#type::opaque_pointer},
    },
    ir::{
        attribute::IntegerAttribute, operation::OperationBuilder, r#type::IntegerType, Block,
        Identifier, Location,
    },
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
    selector: &NullableConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        NullableConcreteLibfunc::Null(info) => {
            build_null(context, registry, entry, location, helper, metadata, info)
        }
        NullableConcreteLibfunc::NullableFromBox(info) => {
            build_nullable_from_box(context, registry, entry, location, helper, metadata, info)
        }
        NullableConcreteLibfunc::MatchNullable(info) => {
            build_match_nullable(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `null` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_null<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let op = entry.append_operation(llvm::nullptr(opaque_pointer(context), location));

    entry.append_operation(helper.br(0, &[op.result(0)?.into()], location));

    Ok(())
}

/// Generate MLIR operations for the `nullable_from_box` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_nullable_from_box<'ctx, 'this, TType, TLibfunc>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));

    Ok(())
}

/// Generate MLIR operations for the `nullable_from_box` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_match_nullable<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let param_type = registry.get_type(&info.param_signatures()[0].ty)?;
    let param_ty = param_type.build(context, helper, registry, metadata)?;

    let arg = entry.argument(0)?.into();

    let op = entry.append_operation(llvm::nullptr(param_ty, location));
    let nullptr = op.result(0)?.into();

    let op = entry.append_operation(
        OperationBuilder::new("llvm.icmp", location)
            .add_operands(&[arg, nullptr])
            .add_attributes(&[(
                Identifier::new(context, "predicate"),
                IntegerAttribute::new(0, IntegerType::new(context, 64).into()).into(),
            )])
            .add_results(&[IntegerType::new(context, 1).into()])
            .build(),
    );

    let is_null_ptr = op.result(0)?.into();

    let block_is_null = helper.append_block(Block::new(&[]));
    let block_is_not_null = helper.append_block(Block::new(&[]));

    entry.append_operation(cf::cond_br(
        context,
        is_null_ptr,
        block_is_null,
        block_is_not_null,
        &[],
        &[],
        location,
    ));

    block_is_null.append_operation(helper.br(0, &[], location));

    block_is_not_null.append_operation(helper.br(1, &[arg], location));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::{load_cairo, run_program};
    use serde_json::json;

    #[test]
    fn run_null() {
        let program = load_cairo!(
            use nullable::null;
            use nullable::match_nullable;
            use nullable::FromNullableResult;
            use nullable::nullable_from_box;
            use box::BoxTrait;

            fn run_test() {
                let a: Nullable<u8> = null();
            }
        );

        let result = run_program(&program, "run_test", json!([]));
        assert_eq!(result, json!([[]]));
    }

    #[test]
    fn run_not_null() {
        let program = load_cairo!(
            use nullable::null;
            use nullable::match_nullable;
            use nullable::FromNullableResult;
            use nullable::nullable_from_box;
            use box::BoxTrait;

            fn run_test(x: u8) -> u8 {
                let b: Box<u8> = BoxTrait::new(x);
                let c = if x == 0 {
                    null()
                } else {
                    nullable_from_box(b)
                };
                let d = match match_nullable(c) {
                    FromNullableResult::Null(_) => 99_u8,
                    FromNullableResult::NotNull(value) => value.unbox()
                };
                d
            }
        );

        let result = run_program(&program, "run_test", json!([4]));
        assert_eq!(result, json!([4]));

        let result = run_program(&program, "run_test", json!([0]));
        assert_eq!(result, json!([99]));
    }

    #[test]
    fn run_null_serialize_roundtrip() {
        let program = load_cairo!(
            fn run_test(x: Nullable<u8>) -> Nullable<u8> {
                x
            }
        );

        /*
        let result = run_program(&program, "run_test", json!([null]));
        assert_eq!(result, json!([null]));
        */

        let result = run_program(&program, "run_test", json!([Some(2)]));
        assert_eq!(result, json!([Some(2)]));
    }
}
