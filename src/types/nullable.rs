//! # Nullable type
//!
//! Nullable is represented as a pointer, usually the null value will point to a alloca in the stack.
//!
//! A nullable is functionally equivalent to Rust's `Option<Box<T>>`. Since it's always paired with
//! `Box<T>` we can reuse its pointer, just leaving it null when there's no value.

use super::{TypeBuilder, WithSelf};
use crate::{
    error::Result,
    metadata::{
        dup_overrides::DupOverrideMeta, realloc_bindings::ReallocBindingsMeta, MetadataStorage,
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
use melior::{
    dialect::{cf, func},
    ir::Region,
};
use melior::{
    dialect::{llvm, ods},
    ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, Location, Module, Type},
    Context,
};

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: WithSelf<InfoAndTypeConcreteType>,
) -> Result<Type<'ctx>> {
    DupOverrideMeta::register_with(
        context,
        module,
        registry,
        metadata,
        info.self_ty(),
        |metadata| {
            // There's no need to build the type here because it'll always be built within
            // `snapshot_take`.

            Ok(Some(build_dup(context, module, registry, metadata, &info)?))
        },
    )?;

    // A nullable is represented by a pointer (equivalent to a box). A null value means no value.
    Ok(llvm::r#type::pointer(context, 0))
}

fn build_dup<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: &WithSelf<InfoAndTypeConcreteType>,
) -> Result<Region<'ctx>> {
    let location = Location::unknown(context);
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, module));
    }

    let inner_ty = registry.get_type(&info.ty)?;
    let inner_len = inner_ty.layout(registry)?.pad_to_align().size();
    let inner_ty = inner_ty.build(context, module, registry, metadata, &info.ty)?;

    let region = Region::new();
    let entry = region.append_block(Block::new(&[(llvm::r#type::pointer(context, 0), location)]));

    let null_ptr =
        entry.append_op_result(llvm::zero(llvm::r#type::pointer(context, 0), location))?;
    let inner_len_val = entry.const_int(context, location, inner_len, 64)?;

    let src_value = entry.argument(0)?.into();
    let src_is_null = entry.append_op_result(
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

    let block_realloc = region.append_block(Block::new(&[]));
    let block_finish =
        region.append_block(Block::new(&[(llvm::r#type::pointer(context, 0), location)]));
    entry.append_operation(cf::cond_br(
        context,
        src_is_null,
        &block_finish,
        &block_realloc,
        &[null_ptr],
        &[],
        location,
    ));

    {
        let dst_value = block_realloc.append_op_result(ReallocBindingsMeta::realloc(
            context,
            null_ptr,
            inner_len_val,
            location,
        ))?;

        match metadata.get::<DupOverrideMeta>() {
            Some(dup_override_meta) if dup_override_meta.is_overriden(&info.ty) => {
                let value = block_realloc.load(context, location, src_value, inner_ty)?;
                let values = dup_override_meta.invoke_override(
                    context,
                    &block_realloc,
                    location,
                    &info.ty,
                    value,
                )?;
                block_realloc.store(context, location, src_value, values.0)?;
                block_realloc.store(context, location, dst_value, values.1)?;
            }
            _ => {
                block_realloc.append_operation(
                    ods::llvm::intr_memcpy_inline(
                        context,
                        dst_value,
                        src_value,
                        IntegerAttribute::new(
                            IntegerType::new(context, 64).into(),
                            inner_len as i64,
                        ),
                        IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
                        location,
                    )
                    .into(),
                );
            }
        }

        block_realloc.append_operation(cf::br(&block_finish, &[dst_value], location));
    }

    block_finish.append_operation(func::r#return(
        &[src_value, block_finish.argument(0)?.into()],
        location,
    ));
    Ok(region)
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
