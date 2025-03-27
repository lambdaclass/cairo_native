use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        types::InfoOnlyConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{func, llvm, ods}, helpers::{ArithBlockExt, BuiltinBlockExt}, ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, BlockLike, Location, Module, Region, Type}, Context
};

use crate::{error::Result, metadata::{drop_overrides::DropOverridesMeta, dup_overrides::DupOverridesMeta, realloc_bindings::ReallocBindingsMeta, MetadataStorage}};

use super::WithSelf;

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: WithSelf<InfoOnlyConcreteType>,
) -> Result<Type<'ctx>> {
    let location = Location::unknown(context);
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, module));
    }

    DupOverridesMeta::register_with(context, module, registry, metadata, info.self_ty(), |_| {
        let region = Region::new();
        let block =
            region.append_block(Block::new(&[(llvm::r#type::pointer(context, 0), location)]));

        let null_ptr =
            block.append_op_result(llvm::zero(llvm::r#type::pointer(context, 0), location))?;
        let k32 = block.const_int(context, location, 32, 64)?;
        let new_ptr = block.append_op_result(ReallocBindingsMeta::realloc(
            context, null_ptr, k32, location,
        )?)?;

        block.append_operation(
            ods::llvm::intr_memcpy_inline(
                context,
                new_ptr,
                block.arg(0)?,
                IntegerAttribute::new(IntegerType::new(context, 64).into(), 32),
                IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
                location,
            )
            .into(),
        );

        block.append_operation(func::r#return(&[block.arg(0)?, new_ptr], location));
        Ok(Some(region))
    })?;
    DropOverridesMeta::register_with(context, module, registry, metadata, info.self_ty(), |_| {
        let region = Region::new();
        let block =
            region.append_block(Block::new(&[(llvm::r#type::pointer(context, 0), location)]));

        block.append_operation(ReallocBindingsMeta::free(context, block.arg(0)?, location)?);

        block.append_operation(func::r#return(&[], location));
        Ok(Some(region))
    })?;

    // A ptr to a heap (realloc) allocated [u32; 8]
    Ok(llvm::r#type::pointer(context, 0))
}
