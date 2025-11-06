//! # `Felt` dictionary libfuncs

use super::LibfuncHelper;
use crate::{
    error::{Error, Result},
    metadata::{debug_utils::DebugUtils, runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureAndTypeConcreteLibfunc,
        squashed_felt252_dict::SquashedFelt252DictConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{llvm, scf},
    helpers::{ArithBlockExt, BuiltinBlockExt, LlvmBlockExt},
    ir::{r#type::IntegerType, Block, BlockLike, Location, Region},
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
    selector: &SquashedFelt252DictConcreteLibfunc,
) -> Result<()> {
    match selector {
        SquashedFelt252DictConcreteLibfunc::IntoEntries(info) => {
            build_into_entries(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_into_entries<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let dict_len = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?
        .dict_len(context, helper, entry, entry.arg(0)?, location)?;

    let k0 = entry.const_int(context, location, 0, 64)?; // TODO: Check if we can use less bits
    let k1 = entry.const_int(context, location, 1, 64)?; // TODO: Check if we can use less bits
    entry.append_operation(scf::r#for(
        k0,
        dict_len,
        k1,
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[(
                IntegerType::new(context, 64).into(), // TODO: Is this the i from the for loop???
                location,
            )]));
            // Get the pointers
            let initial_value_ptr = metadata
                .get_mut::<RuntimeBindingsMeta>()
                .ok_or(Error::MissingMetadata)?
                .dict_get_all(context, helper, entry, entry.arg(0)?, location)?; // TODO: Should I use the entry or the block

            // TODO: The type must not be hardcoded like it it now. It can be take from the info i guess
            let initial_value = block.load(
                context,
                location,
                initial_value_ptr,
                IntegerType::new(context, 32).into(),
            )?; // TODO: Should I use the entry or the block

            metadata.get_mut::<DebugUtils>().unwrap().print_i32(
                context,
                helper,
                entry,
                initial_value,
                location,
            )?; // TODO: Should I use the entry or the block

            region
        },
        location,
    ));

    let entries_arr_ty = llvm::r#type::array(IntegerType::new(context, 32).into(), 0);
    let temp_arr = entry.append_op_result(llvm::undef(entries_arr_ty, location))?;
    helper.br(entry, 0, &[temp_arr], location)
}
