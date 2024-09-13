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
    utils::BlockExt,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        types::InfoAndTypeConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::dialect::cf;
use melior::{
    dialect::{
        llvm::{self, r#type::pointer},
        ods,
    },
    ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, Location, Module, Type, Value},
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

    let inner_snapshot_take = metadata
        .get::<SnapshotClonesMeta>()
        .and_then(|meta| meta.wrap_invoke(&info.ty));

    let inner_type = registry.get_type(&info.ty)?;
    let inner_layout = inner_type.layout(registry)?;
    let inner_ty = inner_type.build(context, helper, registry, metadata, info.self_ty())?;

    let null_ptr = entry
        .append_op_result(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())?;

    let is_null = entry.append_op_result(
        ods::llvm::icmp(
            context,
            IntegerType::new(context, 1).into(),
            src_value,
            null_ptr,
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
            location,
        )
        .into(),
    )?;

    let mut block_not_null = helper.append_block(Block::new(&[]));
    let block_finish = helper.append_block(Block::new(&[(pointer(context, 0), location)]));

    entry.append_operation(cf::cond_br(
        context,
        is_null,
        block_finish,
        block_not_null,
        &[null_ptr],
        &[],
        location,
    ));

    {
        let value_len =
            block_not_null.const_int(context, location, inner_layout.pad_to_align().size(), 64)?;

        let dst_ptr = block_not_null.append_op_result(ReallocBindingsMeta::realloc(
            context, null_ptr, value_len, location,
        ))?;

        match inner_snapshot_take {
            Some(inner_snapshot_take) => {
                let value = block_not_null.load(context, location, src_value, inner_ty)?;

                let (next_block, value) = inner_snapshot_take(
                    context,
                    registry,
                    block_not_null,
                    location,
                    helper,
                    metadata,
                    value,
                )?;
                block_not_null = next_block;

                block_not_null.store(context, location, dst_ptr, value)?;
            }
            None => {
                block_not_null.append_operation(
                    ods::llvm::intr_memcpy(
                        context,
                        dst_ptr,
                        src_value,
                        value_len,
                        IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
                        location,
                    )
                    .into(),
                );
            }
        }
        block_not_null.append_operation(cf::br(block_finish, &[dst_ptr], location));
    }

    let value = block_finish.argument(0)?.into();

    Ok((block_finish, value))
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{jit_enum, jit_struct, load_cairo, run_program},
        values::Value,
    };
    use pretty_assertions_sorted::assert_eq;

    #[test]
    fn test_nullable_deep_clone() {
        let program = load_cairo! {
            use core::array::ArrayTrait;
            use core::NullableTrait;

            fn run_test() -> @Nullable<Array<felt252>> {
                let mut x = NullableTrait::new(array![1, 2, 3]);
                let x_s = @x;

                let mut y = NullableTrait::deref(x);
                y.append(4);

                x_s
            }

        };
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            jit_enum!(
                0,
                jit_struct!(Value::Array(vec![
                    Value::Felt252(1.into()),
                    Value::Felt252(2.into()),
                    Value::Felt252(3.into()),
                ]))
            ),
        );
    }
}
