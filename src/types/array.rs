//! # Array type
//!
//! An array type is a dynamically allocated list of items.
//!
//! ## Layout
//!
//! Being dynamically allocated, we just need to keep the pointer to the data, its length and
//! its capacity:
//!
//! | Index | Type           | Description              |
//! | ----- | -------------- | ------------------------ |
//! |   0   | `!llvm.ptr<T>` | Pointer to the data[^1]. |
//! |   1   | `i32`          | Array start offset[^2].  |
//! |   1   | `i32`          | Array end offset[^2].    |
//! |   2   | `i32`          | Allocated capacity[^2].  |
//!
//! The pointer to the allocation (which is **not the data**) contains:
//!   1. Reference counter.
//!   2. Padding.
//!   3. Array data. Its address is the pointer to the data stored in the type.
//!
//! [^1]: When capacity is zero, this field is not guaranteed to be valid.
//! [^2]: Those numbers are number of items, **not bytes**.

use super::{TypeBuilder, WithSelf};
use crate::{
    error::Result,
    metadata::{
        drop_overrides::DropOverridesMeta, dup_overrides::DupOverridesMeta,
        realloc_bindings::ReallocBindingsMeta, MetadataStorage,
    },
    utils::{get_integer_layout, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        types::InfoAndTypeConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith, cf, llvm},
    ir::{r#type::IntegerType, Block, Location, Module, Type},
    Context,
};
use melior::{
    dialect::{arith::CmpiPredicate, func, scf},
    ir::BlockLike,
};
use melior::{
    helpers::{ArithBlockExt, BuiltinBlockExt, GepIndex, LlvmBlockExt},
    ir::Region,
};
use std::alloc::Layout;

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
    DupOverridesMeta::register_with(
        context,
        module,
        registry,
        metadata,
        info.self_ty(),
        |metadata, region, entry_block, return_block| {
            build_dup(
                context,
                module,
                region,
                entry_block,
                return_block,
                registry,
                metadata,
                &info,
            )
        },
    )?;
    DropOverridesMeta::register_with(
        context,
        module,
        registry,
        metadata,
        info.self_ty(),
        |metadata| {
            // There's no need to build the type here because it'll always be built within
            // `build_drop`.

            Ok(Some(build_drop(
                context, module, registry, metadata, &info,
            )?))
        },
    )?;

    let ptr_ty = llvm::r#type::pointer(context, 0);
    let len_ty = IntegerType::new(context, 32).into();

    Ok(llvm::r#type::r#struct(
        context,
        &[ptr_ty, len_ty, len_ty, len_ty],
        false,
    ))
}

/// This function clones the array shallowly. That is, it'll increment the reference counter but not
/// actually clone anything. The deep clone implementation is provided in `src/libfuncs/array.rs` as
/// part of some libfuncs's implementations.
#[allow(clippy::too_many_arguments)]
fn build_dup<'ctx>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    _region: &Region<'ctx>,
    entry: &Block<'ctx>,
    return_block: &Block<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _metadata: &mut MetadataStorage,
    info: &WithSelf<InfoAndTypeConcreteType>,
) -> Result<()> {
    let location = Location::unknown(context);

    let elem_layout = registry.get_type(&info.ty)?.layout(registry)?;
    let refcount_offset = calc_data_prefix_offset(elem_layout);

    let array_cap = entry.extract_value(
        context,
        location,
        entry.argument(0)?.into(),
        IntegerType::new(context, 32).into(),
        3,
    )?;
    let k0 = entry.const_int(context, location, 0, 32)?;
    let is_empty = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        array_cap,
        k0,
        location,
    ))?;

    entry.append_operation(scf::r#if(
        is_empty,
        &[],
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            block.append_operation(scf::r#yield(&[], location));
            region
        },
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let array_ptr_ptr = block.extract_value(
                context,
                location,
                entry.argument(0)?.into(),
                llvm::r#type::pointer(context, 0),
                0,
            )?;
            let array_ptr = block.load(
                context,
                location,
                array_ptr_ptr,
                llvm::r#type::pointer(context, 0),
            )?;

            let refcount_ptr = block.gep(
                context,
                location,
                array_ptr,
                &[GepIndex::Const(-(refcount_offset as i32))],
                IntegerType::new(context, 8).into(),
            )?;
            let ref_count = block.load(
                context,
                location,
                refcount_ptr,
                IntegerType::new(context, 32).into(),
            )?;

            let k1 = block.const_int(context, location, 1, 32)?;
            let ref_count = block.append_op_result(arith::addi(ref_count, k1, location))?;
            block.store(context, location, refcount_ptr, ref_count)?;

            block.append_operation(scf::r#yield(&[], location));
            region
        },
        location,
    ));

    entry.append_operation(cf::br(
        return_block,
        &[entry.argument(0)?.into(), entry.argument(0)?.into()],
        location,
    ));

    Ok(())
}

/// This function decreases the reference counter of the array by one.
/// If the reference counter reaches zero, then all the resources are freed.
fn build_drop<'ctx>(
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

    let value_ty = registry.build_type(context, module, metadata, info.self_ty())?;

    let elem_ty = registry.get_type(&info.ty)?;
    let elem_stride = elem_ty.layout(registry)?.pad_to_align().size();
    let elem_ty = elem_ty.build(context, module, registry, metadata, &info.ty)?;
    let elem_layout = registry.get_type(&info.ty)?.layout(registry)?;
    let refcount_offset = calc_data_prefix_offset(elem_layout);

    let region = Region::new();
    let entry = region.append_block(Block::new(&[(value_ty, location)]));

    let array_ptr_ptr = entry.extract_value(
        context,
        location,
        entry.argument(0)?.into(),
        llvm::r#type::pointer(context, 0),
        0,
    )?;

    let array_cap = entry.extract_value(
        context,
        location,
        entry.argument(0)?.into(),
        IntegerType::new(context, 32).into(),
        3,
    )?;
    let k0 = entry.const_int(context, location, 0, 32)?;
    let zero_capacity = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        array_cap,
        k0,
        location,
    ))?;

    entry.append_operation(scf::r#if(
        zero_capacity,
        &[],
        {
            // if the array has no capacity, do nothing, as there is no allocation

            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            block.append_operation(scf::r#yield(&[], location));
            region
        },
        {
            // if the array has capacity, decrease the reference counter
            // and, in case it reaches zero, free all the resources.

            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            // obtain the reference counter
            let array_ptr = block.load(
                context,
                location,
                array_ptr_ptr,
                llvm::r#type::pointer(context, 0),
            )?;
            let refcount_ptr = block.gep(
                context,
                location,
                array_ptr,
                &[GepIndex::Const(-(refcount_offset as i32))],
                IntegerType::new(context, 8).into(),
            )?;
            let ref_count = block.load(
                context,
                location,
                refcount_ptr,
                IntegerType::new(context, 32).into(),
            )?;

            // if the reference counter is greater than 1, then it's shared
            let k1 = block.const_int(context, location, 1, 32)?;
            let is_shared = block.append_op_result(arith::cmpi(
                context,
                CmpiPredicate::Ne,
                ref_count,
                k1,
                location,
            ))?;

            block.append_operation(scf::r#if(
                is_shared,
                &[],
                {
                    // if the array is shared, decrease the reference counter by one
                    let region = Region::new();
                    let block = region.append_block(Block::new(&[]));

                    let ref_count = block.append_op_result(arith::subi(ref_count, k1, location))?;
                    block.store(context, location, refcount_ptr, ref_count)?;

                    block.append_operation(scf::r#yield(&[], location));
                    region
                },
                {
                    // if the array is not shared, drop all elements and free the memory
                    let region = Region::new();
                    let block = region.append_block(Block::new(&[]));

                    if DropOverridesMeta::is_overriden(metadata, &info.ty) {
                        let k0 = block.const_int(context, location, 0, 64)?;
                        let elem_stride = block.const_int(context, location, elem_stride, 64)?;

                        let max_len_ptr = block.gep(
                            context,
                            location,
                            array_ptr,
                            &[GepIndex::Const(
                                -((refcount_offset - size_of::<u32>()) as i32),
                            )],
                            IntegerType::new(context, 8).into(),
                        )?;
                        let max_len = block.load(
                            context,
                            location,
                            max_len_ptr,
                            IntegerType::new(context, 32).into(),
                        )?;
                        let max_len =
                            block.extui(max_len, IntegerType::new(context, 64).into(), location)?;
                        let offset_end = block.muli(max_len, elem_stride, location)?;

                        // Drop each element in the array.
                        block.append_operation(scf::r#for(
                            k0,
                            offset_end,
                            elem_stride,
                            {
                                let region = Region::new();
                                let block = region.append_block(Block::new(&[(
                                    IntegerType::new(context, 64).into(),
                                    location,
                                )]));

                                let elem_offset = block.argument(0)?.into();
                                let elem_ptr = block.gep(
                                    context,
                                    location,
                                    array_ptr,
                                    &[GepIndex::Value(elem_offset)],
                                    IntegerType::new(context, 8).into(),
                                )?;
                                let elem_val = block.load(context, location, elem_ptr, elem_ty)?;

                                DropOverridesMeta::invoke_override(
                                    context, registry, module, &block, &block, location, metadata,
                                    &info.ty, elem_val,
                                )?;

                                block.append_operation(scf::r#yield(&[], location));
                                region
                            },
                            location,
                        ));
                    }

                    // finally, free the array allocation
                    block.append_operation(ReallocBindingsMeta::free(
                        context,
                        refcount_ptr,
                        location,
                    )?);
                    block.append_operation(ReallocBindingsMeta::free(
                        context,
                        array_ptr_ptr,
                        location,
                    )?);

                    block.append_operation(scf::r#yield(&[], location));
                    region
                },
                location,
            ));

            block.append_operation(scf::r#yield(&[], location));
            region
        },
        location,
    ));

    entry.append_operation(func::r#return(&[], location));
    Ok(region)
}

pub fn calc_data_prefix_offset(layout: Layout) -> usize {
    get_integer_layout(32)
        .extend(get_integer_layout(32))
        .expect("creating a layout of two i32 should never fail")
        .0
        .align_to(layout.align())
        .expect("layout size rounded up to the next multiple of layout alignment should never be greater than ISIZE::MAX")
        .pad_to_align()
        .size()
}

#[cfg(test)]
mod test {
    use crate::{load_cairo, utils::testing::run_program, values::Value};
    use pretty_assertions_sorted::assert_eq;

    #[test]
    fn test_array_snapshot_deep_clone() {
        let program = load_cairo! {
            fn run_test() -> @Array<Array<felt252>> {
                let mut inputs: Array<Array<felt252>> = ArrayTrait::new();
                inputs.append(array![1, 2, 3]);
                inputs.append(array![4, 5, 6]);

                @inputs
            }
        };
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            Value::Array(vec![
                Value::Array(vec![
                    Value::Felt252(1.into()),
                    Value::Felt252(2.into()),
                    Value::Felt252(3.into()),
                ]),
                Value::Array(vec![
                    Value::Felt252(4.into()),
                    Value::Felt252(5.into()),
                    Value::Felt252(6.into()),
                ]),
            ]),
        );
    }
}
