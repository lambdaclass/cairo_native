//! # Nullable type
//!
//! Nullable is represented as a pointer, usually the null value will point to a alloca in the stack.
//!
//! A nullable is functionally equivalent to Rust's `Option<Box<T>>`. Since it's always paired with
//! `Box<T>` we can reuse its pointer, just leaving it null when there's no value.

use super::{TypeBuilder, WithSelf};
use crate::{
    error::Result,
    libfuncs::LibfuncHelper,
    metadata::{
        realloc_bindings::ReallocBindingsMeta, snapshot_clones::SnapshotClonesMeta, MetadataStorage,
    },
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        types::InfoAndTypeConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        llvm::{self, r#type::pointer},
        ods, scf,
    },
    ir::{
        attribute::IntegerAttribute, operation::OperationBuilder, r#type::IntegerType, Block,
        Location, Module, Region, Type, Value,
    },
    Context,
};

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: WithSelf<InfoAndTypeConcreteType>,
) -> Result<Type<'ctx>> {
    metadata
        .get_or_insert_with::<SnapshotClonesMeta>(SnapshotClonesMeta::default)
        .register(
            info.self_ty().clone(),
            snapshot_take,
            InfoAndTypeConcreteType {
                info: info.info.clone(),
                ty: info.ty.clone(),
            },
        );

    // nullable is represented as a pointer, like a box, used to check if its null (when it can be null).
    Ok(llvm::r#type::pointer(context, 0))
}

#[allow(clippy::too_many_arguments)]
fn snapshot_take<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: WithSelf<InfoAndTypeConcreteType>,
    src_value: Value<'ctx, 'this>,
) -> Result<(&'this Block<'ctx>, Value<'ctx, 'this>)> {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }

    let elem_layout = registry.get_type(&info.ty)?.layout(registry)?;

    let k0 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
            location,
        ))
        .result(0)?
        .into();
    let null_ptr = entry
        .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
        .result(0)?
        .into();

    let ptr_value = entry
        .append_operation(
            OperationBuilder::new("llvm.ptrtoint", location)
                .add_operands(&[src_value])
                .add_results(&[IntegerType::new(context, 64).into()])
                .build()?,
        )
        .result(0)?
        .into();

    let is_null = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Eq,
            ptr_value,
            k0,
            location,
        ))
        .result(0)?
        .into();

    let value = entry
        .append_operation(scf::r#if(
            is_null,
            &[llvm::r#type::pointer(context, 0)],
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                block.append_operation(scf::r#yield(&[null_ptr], location));
                region
            },
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                let alloc_len = block
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(
                            IntegerType::new(context, 64).into(),
                            elem_layout.size() as i64,
                        )
                        .into(),
                        location,
                    ))
                    .result(0)?
                    .into();

                let cloned_ptr = block
                    .append_operation(ReallocBindingsMeta::realloc(
                        context, null_ptr, alloc_len, location,
                    ))
                    .result(0)?
                    .into();

                block.append_operation(
                    ods::llvm::intr_memcpy(
                        context,
                        cloned_ptr,
                        src_value,
                        alloc_len,
                        IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
                        location,
                    )
                    .into(),
                );

                block.append_operation(scf::r#yield(&[cloned_ptr], location));
                region
            },
            location,
        ))
        .result(0)?
        .into();

    Ok((entry, value))
}
