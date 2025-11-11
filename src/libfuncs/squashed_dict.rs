use super::LibfuncHelper;
use crate::{
    error::{Error, Result},
    metadata::{
        debug_utils::DebugUtils, dup_overrides::DupOverridesMeta,
        realloc_bindings::ReallocBindingsMeta, runtime_bindings::RuntimeBindingsMeta,
        MetadataStorage,
    },
    native_panic,
    types::array::calc_data_prefix_offset,
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        lib_func::SignatureAndTypeConcreteLibfunc,
        squashed_felt252_dict::SquashedFelt252DictConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    helpers::{ArithBlockExt, BuiltinBlockExt, GepIndex, LlvmBlockExt},
    ir::{r#type::IntegerType, Block, Location, Value},
    Context,
};
use std::alloc::Layout;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &SquashedFelt252DictConcreteLibfunc,
) -> Result<()> {
    match selector {
        SquashedFelt252DictConcreteLibfunc::IntoEntries(info) => {
            build_into_entries(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Get the layout of the tuple (felt252, T, T)
fn get_inner_type_layout<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<Layout> {
    let array_ty = registry.get_type(&info.signature.branch_signatures[0].vars[0].ty)?;
    let CoreTypeConcrete::Array(info) = array_ty else {
        native_panic!("Received wrong type");
    };
    let (_, elem_layout) = registry.build_type_with_layout(context, helper, metadata, &info.ty)?;

    Ok(elem_layout)
}

fn build_entries_array<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<Value<'ctx, 'this>> {
    metadata.get_or_insert_with(|| ReallocBindingsMeta::new(context, helper));
    // Register dup and drop implentations
    // register_dup_and_drop(context, registry, helper, metadata, info); // TODO: Investigate this
    let inner_type_layout = get_inner_type_layout(context, registry, helper, metadata, info)?;
    let data_prefix_size = calc_data_prefix_offset(inner_type_layout);
    let elem_stride = entry.const_int(
        context,
        location,
        inner_type_layout.pad_to_align().size(),
        64,
    )?;

    // Get the lenght of the dictionary and the layout of the elements
    let dict_len = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?
        .dict_len(context, helper, entry, entry.arg(0)?, location)?;

    // Build the array
    let ptr_ty = llvm::r#type::pointer(context, 0);
    let len_ty = IntegerType::new(context, 32).into();

    let nullptr = entry.append_op_result(llvm::zero(ptr_ty, location))?;
    let k0 = entry.const_int_from_type(context, location, 0, len_ty)?;

    let value = entry.append_op_result(llvm::undef(
        llvm::r#type::r#struct(context, &[ptr_ty, len_ty, len_ty, len_ty], false),
        location,
    ))?;
    let allocated_capacity =
        entry.trunci(dict_len, IntegerType::new(context, 32).into(), location)?; // TODO: Check if this is the correct capacity
    let end_offset = entry.trunci(dict_len, IntegerType::new(context, 32).into(), location)?; // TODO: Check if this is the correct end_offset. Is it the number of bytes or the quantity of elements

    // let start_offset = TODO: What value should go here?
    let arr = entry.insert_values(
        context,
        location,
        value,
        &[nullptr, k0, end_offset, allocated_capacity],
    )?;

    // Alloc space for elements of the array
    let nullptr = entry.append_op_result(llvm::zero(ptr_ty, location))?; // TODO: Maybe I should use the one from above
    let data_prefix_size_value = entry.const_int(context, location, data_prefix_size, 64)?;
    let realloc_len = entry.muli(elem_stride, dict_len, location)?;
    let realloc_len = entry.addi(realloc_len, data_prefix_size_value, location)?;
    let array_ptr = entry.append_op_result(ReallocBindingsMeta::realloc(
        context,
        nullptr,
        realloc_len,
        location,
    )?)?;
    // Store the reference counter
    let k1 = entry.const_int_from_type(context, location, 1, len_ty)?;
    entry.store(context, location, array_ptr, k1)?;
    // Store the max length
    let max_len_ptr = entry.gep(
        context,
        location,
        array_ptr,
        &[GepIndex::Const(size_of::<u32>() as i32)],
        IntegerType::new(context, 8).into(),
    )?;
    let max_length = entry.trunci(dict_len, IntegerType::new(context, 32).into(), location)?;
    entry.store(context, location, max_len_ptr, max_length)?; // TODO: Check if this is the correct value to store here

    // Move to get the pointer in the position of the data
    let array_ptr = entry.gep(
        context,
        location,
        array_ptr,
        &[GepIndex::Const(data_prefix_size as i32)],
        IntegerType::new(context, 8).into(),
    )?;
    // Alloc space in the null_ptr so it can store the ptr to the array
    let k8 = entry.const_int(context, location, 8, 64)?;
    let array_ptr_ptr = entry.append_op_result(ReallocBindingsMeta::realloc(
        context, nullptr, k8, location,
    )?)?;
    // Store the pointer to the data in the pointer of the array
    entry.store(context, location, array_ptr_ptr, array_ptr)?;
    // Insert the pointer to the data pointer inside the array
    let arr = entry.insert_value(context, location, arr, array_ptr_ptr, 0)?;

    // Set max length
    // let max_len_ptr = entry.gep(
    //     context,
    //     location,
    //     array_ptr,
    //     &[GepIndex::Const(
    //         -((data_prefix_size - size_of::<u32>()) as i32),
    //     )],
    //     IntegerType::new(context, 8).into(),
    // )?;
    // let max_length = entry.trunci(dict_len, IntegerType::new(context, 32).into(), location)?;
    // entry.store(context, location, max_len_ptr, max_length)?;
    // Set reference counter to 1
    // let refcount_ptr = entry.gep(
    //     context,
    //     location,
    //     array_ptr,
    //     &[GepIndex::Const(-(data_prefix_size as i32))],
    //     IntegerType::new(context, 8).into(),
    // )?;
    // let k1 = entry.const_int(context, location, 1, 32)?;
    // entry.store(context, location, refcount_ptr, k1)?;

    Ok(arr)
}

pub fn build_into_entries<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    // Build the tuples array
    let entries_array =
        build_entries_array(context, registry, entry, location, helper, metadata, info)?;

    // Get the ptr to the data and pass it to the runtime function
    let ptr_ty = llvm::r#type::pointer(context, 0);
    let data_ptr_ptr = entry.extract_value(context, location, entries_array, ptr_ty, 0)?;
    let data_ptr = entry.load(context, location, data_ptr_ptr, ptr_ty)?;

    // let len_ty = IntegerType::new(context, 32).into();
    // let start_off = entry.extract_value(context, location, entries_array, len_ty, 1)?;
    // let end_off = entry.extract_value(context, location, entries_array, len_ty, 2)?;
    // let capacity = entry.extract_value(context, location, entries_array, len_ty, 3)?;
    // metadata
    //     .get_mut::<DebugUtils>()
    //     .unwrap()
    //     .print_pointer(context, helper, entry, ptr, location)?;
    // metadata
    //     .get_mut::<DebugUtils>()
    //     .unwrap()
    //     .print_i32(context, helper, entry, start_off, location)?;
    // metadata
    //     .get_mut::<DebugUtils>()
    //     .unwrap()
    //     .print_i32(context, helper, entry, end_off, location)?;
    // metadata
    //     .get_mut::<DebugUtils>()
    //     .unwrap()
    //     .print_i32(context, helper, entry, capacity, location)?;

    // let ptr_ty = llvm::r#type::pointer(context, 0);
    // let len_ty = IntegerType::new(context, 32).into();

    // let nullptr = entry.append_op_result(llvm::zero(ptr_ty, location))?;
    // let k0 = entry.const_int_from_type(context, location, 0, len_ty)?;

    // let value = entry.append_op_result(llvm::undef(
    //     llvm::r#type::r#struct(context, &[ptr_ty, len_ty, len_ty, len_ty], false),
    //     location,
    // ))?;
    // let value = entry.insert_values(context, location, value, &[nullptr, k0, k0, k0])?;

    helper.br(entry, 0, &[entries_array], location)
}
