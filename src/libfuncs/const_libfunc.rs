//! # Const libfuncs

use super::LibfuncHelper;
use crate::{error::libfuncs::Result, metadata::MetadataStorage, utils::ProgramRegistryExt};
use cairo_lang_sierra::{
    extensions::{
        const_type::{
            ConstAsBoxConcreteLibfunc, ConstAsImmediateConcreteLibfunc, ConstConcreteLibfunc,
        },
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::arith,
    ir::{Attribute, Block, Location},
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
    selector: &ConstConcreteLibfunc,
) -> Result<()> {
    match selector {
        ConstConcreteLibfunc::AsBox(info) => {
            build_const_as_box(context, registry, entry, location, helper, metadata, info)
        }
        ConstConcreteLibfunc::AsImmediate(info) => {
            build_const_as_immediate(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `const_as_box` libfunc.
pub fn build_const_as_box<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _entry: &'this Block<'ctx>,
    _location: Location<'ctx>,
    _helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &ConstAsBoxConcreteLibfunc,
) -> Result<()> {
    todo!()
}

/// Generate MLIR operations for the `const_as_immediate` libfunc.
pub fn build_const_as_immediate<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &ConstAsImmediateConcreteLibfunc,
) -> Result<()> {
    let value_ty = registry.get_type(&info.const_type)?;

    let const_type = match &value_ty {
        CoreTypeConcrete::Const(inner) => inner,
        _ => unreachable!(),
    };
    // const_type.inner_data Should be one of the following:
    // - A single value, if the inner type is a simple numeric type (e.g., `felt252`, `u32`,
    //   etc.).
    // - A list of const types, if the inner type is a struct. The type of each const type must be
    //   the same as the corresponding struct member type.
    // - A selector (a single value) followed by a const type, if the inner type is an enum. The
    //   type of the const type must be the same as the corresponding enum variant type.

    let value_type =
        registry.build_type(context, helper, registry, metadata, &const_type.inner_ty)?;

    // it seems it's only used for simple data types.
    let result = entry
        .append_operation(arith::constant(
            context,
            Attribute::parse(
                context,
                &format!("{} : {}", const_type.inner_data[0], value_type),
            )
            .unwrap(),
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[result], location));
    Ok(())
}
