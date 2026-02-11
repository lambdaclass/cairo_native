//! # Array type
//!
//! An array type is a dynamically allocated list of items.
//!
//! ## Layout
//!
//! Being dynamically allocated, we just need to keep the pointer to the metadata, its length and
//! its capacity:
//!
//! | Index | Type        | Description                   |
//! | ----- | ----------- | ----------------------------- |
//! |   0   | `!llvm.ptr` | Pointer to ArrayMetadata[^1]. |
//! |   1   | `i32`       | Array start offset[^2].       |
//! |   2   | `i32`       | Array end offset[^2].         |
//! |   3   | `i32`       | Allocated capacity[^2].       |
//!
//! The ArrayMetadata struct contains:
//!   1. Reference counter (`u32`).
//!   2. Array max length (`u32`).
//!   3. Pointer to array data (`*mut u8`).
//!
//! [^1]: This pointer is null when the array has not yet been allocated (i.e., initially).
//! [^2]: Those numbers are number of items, **not bytes**.

use super::{TypeBuilder, WithSelf};
use crate::{
    error::Result,
    metadata::{
        drop_overrides::DropOverridesMeta, dup_overrides::DupOverridesMeta,
        realloc_bindings::ReallocBindingsMeta, MetadataStorage,
    },
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        types::InfoAndTypeConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith, llvm, ods},
    ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, Location, Module, Type},
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
        |metadata| {
            // There's no need to build the type here because it'll always be built within
            // `build_dup`.

            Ok(Some(build_dup(context, module, registry, metadata, &info)?))
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
pub fn build_dup<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: &WithSelf<InfoAndTypeConcreteType>,
) -> Result<Region<'ctx>> {
    let location = Location::unknown(context);
    let value_ty = registry.build_type(context, module, metadata, info.self_ty())?;

    let region = Region::new();
    let entry = region.append_block(Block::new(&[(value_ty, location)]));

    let metadata_ptr = entry.extract_value(
        context,
        location,
        entry.argument(0)?.into(),
        llvm::r#type::pointer(context, 0),
        0,
    )?;

    let null_ptr =
        entry.append_op_result(llvm::zero(llvm::r#type::pointer(context, 0), location))?;
    let is_empty = entry.append_op_result(
        ods::llvm::icmp(
            context,
            IntegerType::new(context, 1).into(),
            metadata_ptr,
            null_ptr,
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
            location,
        )
        .into(),
    )?;

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

            // Metadata struct: { refcount: u32, max_len: u32, data_ptr: *mut u8 }
            // Access refcount field (index 0)
            let refcount_ptr = block.gep(
                context,
                location,
                metadata_ptr,
                &[GepIndex::Const(0), GepIndex::Const(0)],
                get_metadata_llvm_type(context),
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

    entry.append_operation(func::r#return(
        &[entry.argument(0)?.into(), entry.argument(0)?.into()],
        location,
    ));
    Ok(region)
}

/// This function decreases the reference counter of the array by one.
/// If the reference counter reaches zero, then all the resources are freed.
pub fn build_drop<'ctx>(
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

    let region = Region::new();
    let entry = region.append_block(Block::new(&[(value_ty, location)]));

    let metadata_ptr = entry.extract_value(
        context,
        location,
        entry.argument(0)?.into(),
        llvm::r#type::pointer(context, 0),
        0,
    )?;

    let null_ptr =
        entry.append_op_result(llvm::zero(llvm::r#type::pointer(context, 0), location))?;
    let is_null = entry.append_op_result(
        ods::llvm::icmp(
            context,
            IntegerType::new(context, 1).into(),
            metadata_ptr,
            null_ptr,
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
            location,
        )
        .into(),
    )?;

    entry.append_operation(scf::r#if(
        is_null,
        &[],
        {
            // if the metadata pointer is null, do nothing

            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            block.append_operation(scf::r#yield(&[], location));
            region
        },
        {
            // if metadata exists, decrease the reference counter
            // and, in case it reaches zero, free all the resources.

            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            // Metadata struct: { refcount: u32, max_len: u32, data_ptr: *mut u8 }
            // Access refcount field (index 0)
            let refcount_ptr = block.gep(
                context,
                location,
                metadata_ptr,
                &[GepIndex::Const(0), GepIndex::Const(0)],
                get_metadata_llvm_type(context),
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

                        // Load max_len from metadata (field index 1)
                        let max_len_ptr = block.gep(
                            context,
                            location,
                            metadata_ptr,
                            &[GepIndex::Const(0), GepIndex::Const(1)],
                            get_metadata_llvm_type(context),
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

                        // Load data_ptr from metadata (field index 2)
                        let data_ptr_ptr = block.gep(
                            context,
                            location,
                            metadata_ptr,
                            &[GepIndex::Const(0), GepIndex::Const(2)],
                            get_metadata_llvm_type(context),
                        )?;
                        let data_ptr = block.load(
                            context,
                            location,
                            data_ptr_ptr,
                            llvm::r#type::pointer(context, 0),
                        )?;

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
                                    data_ptr,
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

                    // Load data_ptr from metadata (field index 2) and free it
                    let data_ptr_ptr = block.gep(
                        context,
                        location,
                        metadata_ptr,
                        &[GepIndex::Const(0), GepIndex::Const(2)],
                        get_metadata_llvm_type(context),
                    )?;
                    let data_ptr = block.load(
                        context,
                        location,
                        data_ptr_ptr,
                        llvm::r#type::pointer(context, 0),
                    )?;

                    // Free the data allocation
                    block.append_operation(ReallocBindingsMeta::free(context, data_ptr, location)?);

                    // Free the metadata struct
                    block.append_operation(ReallocBindingsMeta::free(
                        context,
                        metadata_ptr,
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

/// Metadata struct definition for arrays (RC, maxlen, data_ptr)
/// Note: capacity stays in the array struct, not here!
#[repr(C)]
pub struct ArrayMetadata {
    pub refcount: u32,
    pub max_len: u32,
    pub data_ptr: *mut u8,
}

/// Returns the metadata struct layout size.
pub fn calc_metadata_size() -> usize {
    std::mem::size_of::<ArrayMetadata>()
}

/// Get the LLVM struct type for ArrayMetadata: { refcount: i32, max_len: i32, data_ptr: ptr }
///
/// Field indices:
/// - 0: refcount (u32)
/// - 1: max_len (u32)
/// - 2: data_ptr (*mut u8)
pub fn get_metadata_llvm_type(context: &Context) -> Type<'_> {
    llvm::r#type::r#struct(
        context,
        &[
            IntegerType::new(context, 32).into(), // refcount: u32
            IntegerType::new(context, 32).into(), // max_len: u32
            llvm::r#type::pointer(context, 0),    // data_ptr: *mut u8
        ],
        false,
    )
}

#[cfg(test)]
mod test {
    use crate::{
        utils::testing::{get_compiled_program, run_program},
        values::Value,
    };
    use pretty_assertions_sorted::assert_eq;

    #[test]
    fn test_array_snapshot_deep_clone() {
        let program = get_compiled_program("test_data_artifacts/programs/types/nested_arrays");
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
