//! # Builtin costs type
//!
//! A ptr to a list of u64, this list will not change at runtime in size and thus we only really need to store the pointer,
//! it can be allocated on the stack on rust side and passed.

use super::WithSelf;
use crate::{error::Result, metadata::MetadataStorage, utils::BlockExt};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        types::InfoOnlyConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{llvm, ods},
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        Attribute, Block, Location, Module, Region, Type,
    },
    Context,
};

struct BuiltinCostsMeta;

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    _info: WithSelf<InfoOnlyConcreteType>,
) -> Result<Type<'ctx>> {
    // A ptr to a list of u64

    if metadata.get::<BuiltinCostsMeta>().is_none() {
        let region = Region::new();
        let location = Location::unknown(context);
        let block = region.append_block(Block::new(&[]));
        let value = block.append_op_result(
            ods::llvm::mlir_zero(context, llvm::r#type::pointer(context, 0), location).into(),
        )?;
        block.append_op_result(melior::dialect::llvm::r#return(Some(value), location))?;

        ods::llvm::mlir_global(
            context,
            region,
            TypeAttribute::new(llvm::r#type::pointer(context, 0)),
            StringAttribute::new(context, "builtin_costs"),
            Attribute::parse(context, "#llvm.linkage<external>").unwrap(),
            location,
        );
        metadata.insert(BuiltinCostsMeta);
    }

    Ok(llvm::r#type::pointer(context, 0))
}
