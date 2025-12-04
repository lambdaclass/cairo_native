use super::LibfuncHelper;
use crate::{
    debug::type_to_name,
    error::{Error, Result},
    metadata::{
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
    dialect::{llvm, scf},
    helpers::{ArithBlockExt, BuiltinBlockExt, GepIndex, LlvmBlockExt},
    ir::{
        operation::OperationBuilder, r#type::IntegerType, Block, BlockLike, Location, Region, Type,
        Value, ValueLike,
    },
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

/// Get the inner types of the tuple
fn get_inner_types<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<(Type<'ctx>, Type<'ctx>)> {
    let array_ty = registry.get_type(&info.signature.branch_signatures[0].vars[0].ty)?;
    let CoreTypeConcrete::Array(info) = array_ty else {
        native_panic!("Expected Array");
    };
    let struct_ty = registry.get_type(&info.ty)?;
    let CoreTypeConcrete::Struct(info) = struct_ty else {
        native_panic!("Expected Struct");
    };
    let felt_ty_id = &info.members[0];
    let generic_ty_id = &info.members[1];

    let temp = registry.get_type(generic_ty_id)?;
    println!("El generic es: {}", type_to_name(registry, temp));

    let felt_ty = registry.build_type(context, helper, metadata, felt_ty_id)?;
    let generic_ty = registry.build_type(context, helper, metadata, generic_ty_id)?;

    Ok((felt_ty, generic_ty))
}

/// Fills the Array with tuples of the form (felt252, T, T).
#[allow(clippy::too_many_arguments)]
fn append_entry_tuples<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
    arr: Value<'ctx, 'this>,
    dict_len: Value<'ctx, 'this>,
    elem_stride: Value<'ctx, 'this>,
) -> Result<Value<'ctx, 'this>> {
    // Region of the 'for loop' which holds the logic to append a tuple in the array
    let region = {
        let region = Region::new();
        let block = region.append_block(Block::new(&[
            (IntegerType::new(context, 64).into(), location),
            (arr.r#type(), location),
        ]));
        let loop_arr = block.arg(1)?;
        // Get the array and move the pointer to the new slot where the tuple will be inserted (end_offset * element_size)
        let array_ptr_ptr = block.extract_value(
            context,
            location,
            loop_arr,
            llvm::r#type::pointer(context, 0),
            0,
        )?;
        let ptr_ty = llvm::r#type::pointer(context, 0);
        let array_ptr = block.load(context, location, array_ptr_ptr, ptr_ty)?;
        let end_offset = block.extract_value(
            context,
            location,
            loop_arr,
            IntegerType::new(context, 32).into(),
            2,
        )?;
        let end_offset = block.extui(end_offset, IntegerType::new(context, 64).into(), location)?;
        let target_offset = block.muli(end_offset, elem_stride, location)?;
        let target_ptr = block.gep(
            context,
            location,
            array_ptr,
            &[GepIndex::Value(target_offset)],
            IntegerType::new(context, 8).into(),
        )?;

        // Build tuple and store it in the empty slot
        let (felt_ty, generic_ty) = get_inner_types(context, registry, helper, metadata, info)?;
        let tuple = block.append_op_result(llvm::undef(
            llvm::r#type::r#struct(context, &[felt_ty, generic_ty, generic_ty], false),
            location,
        ))?;
        block.store(context, location, target_ptr, tuple)?;

        // Update end offset
        let k1 = block.const_int_from_type(
            context,
            location,
            1,
            IntegerType::new(context, 64).into(),
        )?;
        let end_offset = block.addi(end_offset, k1, location)?;
        let end_offset =
            block.trunci(end_offset, IntegerType::new(context, 32).into(), location)?;
        let new_arr = block.insert_value(context, location, loop_arr, end_offset, 2)?;

        // Return the updated array so the next iteration can use it
        block.append_operation(scf::r#yield(&[new_arr], location));
        region
    };

    let k0 = entry.const_int(context, location, 0, 64)?;
    let k1 = entry.const_int(context, location, 1, 64)?;

    Ok(entry.append_op_result(
        OperationBuilder::new("scf.for", location)
            .add_operands(&[k0, dict_len, k1])
            .add_operands(&[arr])
            .add_results(&[arr.r#type()])
            .add_regions([region])
            .build()?,
    )?)
}

/// Builds an Array<(felt252, T, T)>.
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
    // register_dup_and_drop(
    //     context,
    //     registry,
    //     helper,
    //     metadata,
    //     WithSelf::new(&info.ty, info),
    // ); // TODO: Investigate this

    let inner_type_layout = get_inner_type_layout(context, registry, helper, metadata, info)?;
    let data_prefix_size = calc_data_prefix_offset(inner_type_layout);
    let elem_stride = entry.const_int(
        context,
        location,
        inner_type_layout.pad_to_align().size(),
        64,
    )?;

    // Get the lenght of the dictionary
    let dict_len = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?
        .dict_len(context, helper, entry, entry.arg(0)?, location)?;

    // Build the array structure
    let ptr_ty = llvm::r#type::pointer(context, 0);
    let len_ty = IntegerType::new(context, 32).into();
    let nullptr = entry.append_op_result(llvm::zero(ptr_ty, location))?;
    let value = entry.append_op_result(llvm::undef(
        llvm::r#type::r#struct(context, &[ptr_ty, len_ty, len_ty, len_ty], false),
        location,
    ))?;
    let allocated_capacity =
        entry.trunci(dict_len, IntegerType::new(context, 32).into(), location)?;
    let end_offset = entry.const_int_from_type(context, location, 0, len_ty)?;
    let start_offset = entry.const_int_from_type(context, location, 0, len_ty)?;
    let arr = entry.insert_values(
        context,
        location,
        value,
        &[nullptr, start_offset, end_offset, allocated_capacity],
    )?;

    // Alloc space for elements of the array. We need space to hold the prefix (reference counter and max lenght)
    // and the N tuples, where N is the number of elements in the dict.
    let nullptr = entry.append_op_result(llvm::zero(ptr_ty, location))?;
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
    // After the reference counter we need to store the max length
    let max_len_ptr = entry.gep(
        context,
        location,
        array_ptr,
        &[GepIndex::Const(size_of::<u32>() as i32)],
        IntegerType::new(context, 8).into(),
    )?;
    let max_length = entry.trunci(dict_len, IntegerType::new(context, 32).into(), location)?;
    entry.store(context, location, max_len_ptr, max_length)?;

    // Move to get the pointer in the position of the data
    let array_ptr = entry.gep(
        context,
        location,
        array_ptr,
        &[GepIndex::Const(data_prefix_size as i32)],
        IntegerType::new(context, 8).into(),
    )?;
    // Alloc space in the null_ptr so it can store the ptr to the array
    let nullptr = entry.append_op_result(llvm::zero(ptr_ty, location))?;
    let k8 = entry.const_int(context, location, 8, 64)?;
    let array_ptr_ptr = entry.append_op_result(ReallocBindingsMeta::realloc(
        context, nullptr, k8, location,
    )?)?;
    // Store the pointer to the data in the pointer of the array
    entry.store(context, location, array_ptr_ptr, array_ptr)?;
    // Insert the pointer to the data pointer inside the array
    let arr = entry.insert_value(context, location, arr, array_ptr_ptr, 0)?;

    // Append a tuple for each entry in the dictionary
    let arr = append_entry_tuples(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        info,
        arr,
        dict_len,
        elem_stride,
    )?;

    Ok(arr)
}

/// Generate MLIR operations for the `squashed_felt252_dict_entries` libfunc.
///
/// Receives a `SquashedFelt252Dict<T>` and returns an `Array<(felt252, T, T)>`. This
/// array will have a tuple for each element on the dictionary. The first item represents
/// the key of the element in the dictionary, it is followed by the first value and last
/// value of that same element. Then (felt252, T, T) = (key, first_value, last_value).
///
/// # Caveats
///
/// In the tuple, the value that represents the first value will hold the value 0.
///
/// # Signature
///
/// ```cairo
/// extern fn squashed_felt252_dict_entries<T>(
///    dict: SquashedFelt252Dict<T>,
/// ) -> Array<(felt252, T, T)> nopanic;
/// ```
pub fn build_into_entries<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    // Build the entries array (Array<(felt252, T, T)>)
    let entries_array =
        build_entries_array(context, registry, entry, location, helper, metadata, info)?;

    // Get the ptr to the data of the array and pass it to the runtime function
    let ptr_ty = llvm::r#type::pointer(context, 0);
    let data_ptr_ptr = entry.extract_value(context, location, entries_array, ptr_ty, 0)?;
    let data_ptr = entry.load(context, location, data_ptr_ptr, ptr_ty)?;
    let dict_ptr = entry.arg(0)?;

    // Call runtime function that pushes the elements and keys into the tuples of the array
    let tuple_layout = get_inner_type_layout(context, registry, helper, metadata, info)?;
    let tuple_stride = entry.const_int_from_type(
        context,
        location,
        tuple_layout.pad_to_align().size(),
        IntegerType::new(context, 32).into(),
    )?;
    metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?
        .dict_into_entries(
            context,
            helper,
            entry,
            dict_ptr,
            data_ptr,
            tuple_stride,
            location,
        )?;

    helper.br(entry, 0, &[entries_array], location)
}

#[cfg(test)]
mod test {
    use crate::{jit_struct, load_cairo, utils::testing::run_program, Value};

    #[test]
    fn test_into_entries() {
        let program = load_cairo! {
            use core::dict::{Felt252Dict, Felt252DictEntryTrait, SquashedFelt252DictTrait};

            fn into_entries_u8_values() -> Array<(felt252, u8, u8)> {
                let mut dict: Felt252Dict<u8> = Default::default();
                dict.insert(0, 0);
                dict.insert(1, 1);
                dict.insert(2, 2);
                dict.squash().into_entries()
            }

            fn into_entries_u32_values() -> Array<(felt252, u32, u32)> {
                let mut dict: Felt252Dict<u32> = Default::default();
                dict.insert(0, 0);
                dict.insert(1, 1);
                dict.insert(2, 2);
                dict.squash().into_entries()
            }

            fn into_entries_u128_values() -> Array<(felt252, u128, u128)> {
                let mut dict: Felt252Dict<u128> = Default::default();
                dict.insert(0, 0);
                dict.insert(1, 1);
                dict.insert(2, 2);
                dict.squash().into_entries()
            }

            fn into_entries_felt252_values() -> Array<(felt252, felt252, felt252)> {
                let mut dict: Felt252Dict<felt252> = Default::default();
                dict.insert(0, 0);
                dict.insert(1, 1);
                dict.insert(2, 2);
                dict.squash().into_entries()
            }
        };

        let result = run_program(&program, "into_entries_u8_values", &[]).return_value;
        if let Value::Array(arr) = result {
            assert_eq!(
                arr[0],
                jit_struct!(Value::Felt252(0.into()), Value::Uint8(0), Value::Uint8(0))
            );
            assert_eq!(
                arr[1],
                jit_struct!(Value::Felt252(1.into()), Value::Uint8(0), Value::Uint8(1))
            );
            assert_eq!(
                arr[2],
                jit_struct!(Value::Felt252(2.into()), Value::Uint8(0), Value::Uint8(2))
            );
        }

        let result = run_program(&program, "into_entries_u32_values", &[]).return_value;
        if let Value::Array(arr) = result {
            assert_eq!(
                arr[0],
                jit_struct!(Value::Felt252(0.into()), Value::Uint32(0), Value::Uint32(0))
            );
            assert_eq!(
                arr[1],
                jit_struct!(Value::Felt252(1.into()), Value::Uint32(0), Value::Uint32(1))
            );
            assert_eq!(
                arr[2],
                jit_struct!(Value::Felt252(2.into()), Value::Uint32(0), Value::Uint32(2))
            );
        }

        let result = run_program(&program, "into_entries_u128_values", &[]).return_value;
        if let Value::Array(arr) = result {
            assert_eq!(
                arr[0],
                jit_struct!(
                    Value::Felt252(0.into()),
                    Value::Uint128(0),
                    Value::Uint128(0)
                )
            );
            assert_eq!(
                arr[1],
                jit_struct!(
                    Value::Felt252(1.into()),
                    Value::Uint128(0),
                    Value::Uint128(1)
                )
            );
            assert_eq!(
                arr[2],
                jit_struct!(
                    Value::Felt252(2.into()),
                    Value::Uint128(0),
                    Value::Uint128(2)
                )
            );
        }

        let result = run_program(&program, "into_entries_felt252_values", &[]).return_value;
        if let Value::Array(arr) = result {
            assert_eq!(
                arr[0],
                jit_struct!(
                    Value::Felt252(0.into()),
                    Value::Felt252(0.into()),
                    Value::Felt252(0.into())
                )
            );
            assert_eq!(
                arr[1],
                jit_struct!(
                    Value::Felt252(1.into()),
                    Value::Felt252(0.into()),
                    Value::Felt252(1.into())
                )
            );
            assert_eq!(
                arr[2],
                jit_struct!(
                    Value::Felt252(2.into()),
                    Value::Felt252(0.into()),
                    Value::Felt252(2.into())
                )
            );
        }
    }
}
