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
        arith, cf,
        llvm::{self, AllocaOptions, LoadStoreOptions},
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
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let target_ty = registry
        .get_type(&info.output_types()[0][0])?
        .build(context, helper, registry, metadata)?;

    let op = entry.append_operation(llvm::nullptr(target_ty, location));

    entry.append_operation(helper.br(0, &[op.result(0)?.into()], location));

    Ok(())
}

/// Generate MLIR operations for the `nullable_from_box` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_nullable_from_box<'ctx, 'this, TType, TLibfunc>(
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
    let inner_type = registry.get_type(&info.ty)?;
    let inner_layout = inner_type.layout(registry)?;

    let target_type = registry.get_type(&info.output_types()[0][0])?;

    let target_ty = target_type.build(context, helper, registry, metadata)?;

    let op = helper.init_block.append_operation(arith::constant(
        context,
        IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
        location,
    ));

    let const_1 = op.result(0)?.into();

    let op = helper.init_block.append_operation(llvm::alloca(
        context,
        const_1,
        target_ty,
        location,
        AllocaOptions::new().align(Some(IntegerAttribute::new(
            inner_layout.align() as i64,
            IntegerType::new(context, 64).into(),
        ))),
    ));
    let ptr = op.result(0)?.into();

    entry.append_operation(llvm::store(
        context,
        entry.argument(0)?.into(),
        ptr,
        location,
        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
            inner_layout.align() as i64,
            IntegerType::new(context, 64).into(),
        ))),
    ));

    entry.append_operation(helper.br(0, &[ptr], location));

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
    let inner_type = registry.get_type(&info.ty)?;
    let inner_layout = inner_type.layout(registry)?;
    let inner_ty = inner_type.build(context, helper, registry, metadata)?;

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

    let op = block_is_not_null.append_operation(llvm::load(
        context,
        arg,
        inner_ty,
        location,
        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
            inner_layout.align() as i64,
            IntegerType::new(context, 64).into(),
        ))),
    ));
    let value = op.result(0)?.into();

    block_is_not_null.append_operation(helper.br(1, &[value], location));

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
}
