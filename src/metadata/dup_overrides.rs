//! # Duplication logic overrides
//!
//! By default, values are copied (aka. `memcpy`'d), but some cases (like arrays, boxes, nullables,
//! dictionaries and some structs and enums) need a clone implementation instad. This metadata is
//! a register of types that require a clone implementation as well as the logic to register and
//! invoke those implementations.
//!
//! ## Clone implementations
//!
//! The clone logic is implemented as a function for each type that requires it. It has to be a
//! function to allow self-referencing types. If we inlined the clone implementations,
//! self-referencing types would generate infinite code thus overflowing the stack when generating
//! code.
//!
//! The generated functions are not public (they are internal) and follow this naming convention:
//!
//! ```text
//! dup${type id}
//! ```
//!
//! where `{type id}` is the numeric value of the `ConcreteTypeId`.

use super::MetadataStorage;
use crate::{
    error::{Error, Result},
    types::TypeBuilder,
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{cf, func, llvm},
    helpers::{BuiltinBlockExt, LlvmBlockExt},
    ir::{
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        r#type::FunctionType,
        Attribute, Block, BlockLike, Identifier, Location, Module, Region, Value,
    },
    Context,
};
use std::collections::HashSet;

#[derive(Debug, Default)]
pub struct DupOverridesMeta {
    overriden_types: HashSet<ConcreteTypeId>,
}

impl DupOverridesMeta {
    /// Register a dup override using a closure.
    ///
    /// This function does several things:
    ///   - Registers `DupOverrideMeta` if it wasn't already present.
    ///   - If the type id was already registered it returns and does nothing.
    ///   - Registers the type (without it being actually registered yet).
    ///   - Calls the closure, which returns an `Option<Region>`.
    ///   - If the closure returns a region, generates the function implementation.
    ///   - If the closure returns `None`, it removes the registry entry for the type.
    ///
    /// The type need to be registered before calling the closure, otherwise self-referencing types
    /// would cause stack overflow when registering themselves.
    ///
    /// The callback serves two purposes:
    ///   - To generate the dup implementation, if necessary.
    ///   - To check if we need to generate the implementation (for example, in structs and enums).
    pub(crate) fn register_with<'ctx>(
        context: &'ctx Context,
        module: &Module<'ctx>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        metadata: &mut MetadataStorage,
        id: &ConcreteTypeId,
        f: impl FnOnce(&mut MetadataStorage) -> Result<Option<Region<'ctx>>>,
    ) -> Result<()> {
        {
            let dup_override_meta = metadata.get_or_insert_with(Self::default);
            if dup_override_meta.overriden_types.contains(id) {
                return Ok(());
            }

            dup_override_meta.overriden_types.insert(id.clone());
        }

        match f(metadata)? {
            Some(region) => {
                let location = Location::unknown(context);

                let ty = registry.build_type(context, module, metadata, id)?;
                let ptr_ty = llvm::r#type::pointer(context, 0);

                let entry_block = region.first_block().unwrap();
                let pre_entry_block =
                    region.insert_block_before(entry_block, Block::new(&[(ptr_ty, location)]));
                pre_entry_block.append_operation(cf::br(
                    &entry_block,
                    &[pre_entry_block.load(context, location, pre_entry_block.arg(0)?, ty)?],
                    location,
                ));

                module.body().append_operation(func::func(
                    context,
                    StringAttribute::new(context, &format!("dup${}", id.id)),
                    TypeAttribute::new(FunctionType::new(context, &[ptr_ty], &[ty, ty]).into()),
                    region,
                    &[
                        (
                            Identifier::new(context, "sym_visibility"),
                            StringAttribute::new(context, "public").into(),
                        ),
                        (
                            Identifier::new(context, "llvm.CConv"),
                            Attribute::parse(context, "#llvm.cconv<fastcc>")
                                .ok_or(Error::ParseAttributeError)?,
                        ),
                        (
                            Identifier::new(context, "llvm.linkage"),
                            Attribute::parse(context, "#llvm.linkage<private>")
                                .ok_or(Error::ParseAttributeError)?,
                        ),
                    ],
                    Location::unknown(context),
                ));
            }
            None => {
                // The following getter should always return a value, but the if statement is kept
                // just in case the meta has been removed (which it shouldn't).
                if let Some(dup_override_meta) = metadata.get_mut::<Self>() {
                    dup_override_meta.overriden_types.remove(id);
                }
            }
        }

        Ok(())
    }

    /// Returns whether a type has a registered dup implementation.
    pub(crate) fn is_overriden(metadata: &mut MetadataStorage, id: &ConcreteTypeId) -> bool {
        metadata
            .get_or_insert_with(Self::default)
            .overriden_types
            .contains(id)
    }

    /// Generates code to invoke a dup implementation for a type, or just returns the same value
    /// twice if no implementation was registered.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn invoke_override<'ctx, 'this>(
        context: &'ctx Context,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        module: &Module<'ctx>,
        block: &'this Block<'ctx>,
        location: Location<'ctx>,
        metadata: &mut MetadataStorage,
        id: &ConcreteTypeId,
        value: Value<'ctx, 'this>,
    ) -> Result<(Value<'ctx, 'this>, Value<'ctx, 'this>)> {
        Ok(if Self::is_overriden(metadata, id) {
            let ty = registry.build_type(context, module, metadata, id)?;
            let sierra_ty = registry.get_type(id)?;

            let value = {
                let value_ptr =
                    block.alloca1(context, location, ty, sierra_ty.layout(registry)?.align())?;
                block.store(context, location, value_ptr, value)?;
                value_ptr
            };

            let result = block.append_operation(func::call(
                context,
                FlatSymbolRefAttribute::new(context, &format!("dup${}", id.id)),
                &[value],
                &[ty, ty],
                location,
            ));

            let original = result.result(0)?.into();
            let copy = result.result(1)?.into();

            (original, copy)
        } else {
            (value, value)
        })
    }
}
