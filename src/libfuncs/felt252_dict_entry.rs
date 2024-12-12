//! # `Felt` dictionary entry libfuncs

use super::LibfuncHelper;
use crate::{
    error::{Error, Result},
    metadata::{
        drop_overrides::DropOverridesMeta, dup_overrides::DupOverridesMeta,
        realloc_bindings::ReallocBindingsMeta, runtime_bindings::RuntimeBindingsMeta,
        MetadataStorage,
    },
    types::TypeBuilder,
    utils::{BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        felt252_dict::Felt252DictEntryConcreteLibfunc,
        lib_func::SignatureAndTypeConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{cf, llvm, ods, scf},
    ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, Location, Region},
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Felt252DictEntryConcreteLibfunc,
) -> Result<()> {
    match selector {
        Felt252DictEntryConcreteLibfunc::Get(info) => {
            build_get(context, registry, entry, location, helper, metadata, info)
        }
        Felt252DictEntryConcreteLibfunc::Finalize(info) => {
            build_finalize(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// The felt252_dict_entry_get libfunc receives the dictionary and the key and
/// returns the associated dict entry, along with a clone of it's value.
///
/// The dict entry also contains a pointer to the dictionary.
///
/// If the key doesn't yet exist, it is created and the type's zero-value is returned.
///
/// # Cairo Signature
///
/// ```cairo
/// fn felt252_dict_entry_get<T>(dict: Felt252Dict<T>, key: felt252) -> (Felt252DictEntry<T>, T) nopanic;
/// ```
pub fn build_get<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let (key_ty, key_layout) = registry.build_type_with_layout(
        context,
        helper,
        registry,
        metadata,
        &info.param_signatures()[1].ty,
    )?;
    let entry_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;
    let value_ty = registry.build_type(context, helper, registry, metadata, &info.ty)?;

    let dict_ptr = entry.arg(0)?;
    let entry_key = entry.arg(1)?;

    let entry_key_ptr =
        helper
            .init_block()
            .alloca1(context, location, key_ty, key_layout.align())?;
    entry.store(context, location, entry_key_ptr, entry_key)?;

    // Runtime's dict_get returnes a pointer to the key value, which is a
    // pointer itself. Effectively, we have a double pointer to the value,
    // avoiding allocating an element from inside the runtime.
    let entry_value_ptr_ptr = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?
        .dict_get(context, helper, entry, dict_ptr, entry_key_ptr, location)?;
    let entry_value_ptr = entry.load(
        context,
        location,
        entry_value_ptr_ptr,
        llvm::r#type::pointer(context, 0),
    )?;

    let null_ptr =
        entry.append_op_result(llvm::zero(llvm::r#type::pointer(context, 0), location))?;

    // If the entry_value_ptr is null, then the entry is vacant
    let is_vacant = entry.append_op_result(
        ods::llvm::icmp(
            context,
            IntegerType::new(context, 1).into(),
            entry_value_ptr,
            null_ptr,
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
            location,
        )
        .into(),
    )?;

    let value = entry.append_op_result(scf::r#if(
        is_vacant,
        &[value_ty],
        {
            // If the entry is vacant, then create the default value
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let value = registry
                .get_type(&info.branch_signatures()[0].vars[1].ty)?
                .build_default(
                    context,
                    registry,
                    &block,
                    location,
                    helper,
                    metadata,
                    &info.branch_signatures()[0].vars[1].ty,
                )?;
            block.append_operation(scf::r#yield(&[value], location));

            region
        },
        {
            // If the entry is occupied, then load the previous value from the pointer
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let value = block.load(context, location, entry_value_ptr, value_ty)?;

            block.append_operation(scf::r#yield(&[value], location));

            region
        },
        location,
    ))?;

    // Build the dict entry struct: `crate::types::felt252_dict_entry`
    let dict_entry = entry.append_op_result(llvm::undef(entry_ty, location))?;
    let dict_entry = entry.insert_values(
        context,
        location,
        dict_entry,
        &[dict_ptr, entry_value_ptr_ptr],
    )?;

    entry.append_operation(helper.br(0, &[dict_entry, value], location));
    Ok(())
}

/// The felt252_dict_entry_finalize libfunc receives the dict entry and a new value,
/// inserts the new value in the entry, and returns the full dictionary.
///
/// # Cairo Signature
///
/// ```cairo
/// fn felt252_dict_entry_finalize<T>(dict_entry: Felt252DictEntry<T>, new_value: T) -> Felt252Dict<T> nopanic;
/// ```
pub fn build_finalize<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }

    let (value_ty, value_layout) = registry.build_type_with_layout(
        context,
        helper,
        registry,
        metadata,
        &info.signature.param_signatures[1].ty,
    )?;

    // Get the dict entry struct: `crate::types::felt252_dict_entry`
    let dict_entry = entry.arg(0)?;
    let new_entry_value = entry.arg(1)?;

    let dict_ptr = entry.extract_value(
        context,
        location,
        dict_entry,
        llvm::r#type::pointer(context, 0),
        0,
    )?;
    let entry_value_ptr_ptr = entry.extract_value(
        context,
        location,
        dict_entry,
        llvm::r#type::pointer(context, 0),
        1,
    )?;
    let entry_value_ptr = entry.load(
        context,
        location,
        entry_value_ptr_ptr,
        llvm::r#type::pointer(context, 0),
    )?;

    let null_ptr =
        entry.append_op_result(llvm::zero(llvm::r#type::pointer(context, 0), location))?;

    // If the entry_value_ptr is null, then the entry is vacant
    let is_vacant = entry.append_op_result(
        ods::llvm::icmp(
            context,
            IntegerType::new(context, 1).into(),
            entry_value_ptr,
            null_ptr,
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
            location,
        )
        .into(),
    )?;

    let entry_value_ptr = entry.append_op_result(scf::r#if(
        is_vacant,
        &[llvm::r#type::pointer(context, 0)],
        {
            // If the entry is vacant, then alloc space for the new value
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let value_len = block.const_int(context, location, value_layout.size(), 64)?;
            let entry_value_ptr = block.append_op_result(ReallocBindingsMeta::realloc(
                context, null_ptr, value_len, location,
            )?)?;
            block.store(context, location, entry_value_ptr_ptr, entry_value_ptr)?;
            block.append_operation(scf::r#yield(&[entry_value_ptr], location));

            region
        },
        {
            // If the entry is occupied, do nothing
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            block.append_operation(scf::r#yield(&[entry_value_ptr], location));

            region
        },
        location,
    ))?;

    // Write the new value to the entry pointer
    block_final.store(context, location, entry_value_ptr, new_entry_value)?;

    block_final.append_operation(helper.br(0, &[dict_ptr], location));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::{jit_dict, load_cairo, run_program_assert_output};

    #[test]
    fn run_dict_insert() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test() -> u32 {
                let mut dict: Felt252Dict<u32> = Default::default();
                dict.insert(2, 1_u32);
                dict.get(2)
            }
        );

        run_program_assert_output(&program, "run_test", &[], 1u32.into());
    }

    #[test]
    fn run_dict_insert_big() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test() -> u64 {
                let mut dict: Felt252Dict<u64> = Default::default();
                dict.insert(200000000, 4_u64);
                dict.get(200000000)
            }
        );

        run_program_assert_output(&program, "run_test", &[], 4u64.into());
    }

    #[test]
    fn run_dict_insert_ret_dict() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test() -> Felt252Dict<u32> {
                let mut dict: Felt252Dict<u32> = Default::default();
                dict.insert(2, 1_u32);
                dict
            }
        );

        run_program_assert_output(
            &program,
            "run_test",
            &[],
            jit_dict!(
                2 => 1u32
            ),
        );
    }

    #[test]
    fn run_dict_insert_multiple() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test() -> u32 {
                let mut dict: Felt252Dict<u32> = Default::default();
                dict.insert(2, 1_u32);
                dict.insert(3, 1_u32);
                dict.insert(4, 1_u32);
                dict.insert(5, 1_u32);
                dict.insert(6, 1_u32);
                dict.insert(7, 1_u32);
                dict.insert(8, 1_u32);
                dict.insert(9, 1_u32);
                dict.insert(10, 1_u32);
                dict.insert(11, 1_u32);
                dict.insert(12, 1_u32);
                dict.insert(13, 1_u32);
                dict.insert(14, 1_u32);
                dict.insert(15, 1_u32);
                dict.insert(16, 1_u32);
                dict.insert(17, 1_u32);
                dict.insert(18, 1345432_u32);
                dict.get(18)
            }
        );

        run_program_assert_output(&program, "run_test", &[], 1345432_u32.into());
    }
}
