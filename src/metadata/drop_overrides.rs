//! # Dropping logic overrides
//!
//! By default, values are discarded, but some cases (like arrays, boxes, nullables, dictionaries
//! and some structs and enums) need a drop implementation instad. This metadata is a register of
//! types that require a drop implementation as well as the logic to register and invoke those
//! implementations.
//!
//! ## Drop implementations
//!
//! The drop logic is implemented as a function for each type that requires it. It has to be a
//! function to allow self-referencing types. If we inlined the drop implementations,
//! self-referencing types would generate infinite code thus overflowing the stack when generating
//! code.
//!
//! The generated functions are not public (they are internal) and follow this naming convention:
//!
//! ```text
//! drop${type id}
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
pub struct DropOverridesMeta {
    overriden_types: HashSet<ConcreteTypeId>,
}

impl DropOverridesMeta {
    /// Register a drop override using a closure.
    ///
    /// This function does several things:
    ///   - Registers `DropOverrideMeta` if it wasn't already present.
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
    ///   - To generate the drop implementation, if necessary.
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
            let drop_override_meta = metadata.get_or_insert_with(Self::default);
            if drop_override_meta.overriden_types.contains(id) {
                return Ok(());
            }

            drop_override_meta.overriden_types.insert(id.clone());
        }

        match f(metadata)? {
            Some(region) => {
                let location = Location::unknown(context);

                let ty = registry.build_type(context, module, metadata, id)?;
                let ptr_ty = llvm::r#type::pointer(context, 0);

                let sierra_ty = registry.get_type(id)?;
                let is_memory_allocated = sierra_ty.is_memory_allocated(registry)?;

                let signature_ty = if is_memory_allocated { ptr_ty } else { ty };

                // For memory allocated types, the generated function receives
                // a pointer as argument. However, the user provided callback
                // generates a region that receives a concrete value as
                // argument. To workaround this, we insert a block at the start
                // of the region that dereferences the pointer, and jumps to the
                // user provided implementation.
                if is_memory_allocated {
                    let entry_block = region.first_block().unwrap();
                    let pre_entry_block =
                        region.insert_block_before(entry_block, Block::new(&[(ptr_ty, location)]));
                    pre_entry_block.append_operation(cf::br(
                        &entry_block,
                        &[pre_entry_block.load(context, location, pre_entry_block.arg(0)?, ty)?],
                        location,
                    ));
                }

                module.body().append_operation(func::func(
                    context,
                    StringAttribute::new(context, &format!("drop${}", id.id)),
                    TypeAttribute::new(FunctionType::new(context, &[signature_ty], &[]).into()),
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
                if let Some(drop_override_meta) = metadata.get_mut::<Self>() {
                    drop_override_meta.overriden_types.remove(id);
                }
            }
        }

        Ok(())
    }

    /// Returns whether a type has a registered drop implementation.
    pub(crate) fn is_overriden(metadata: &mut MetadataStorage, id: &ConcreteTypeId) -> bool {
        metadata
            .get_or_insert_with(Self::default)
            .overriden_types
            .contains(id)
    }

    /// Generates code to invoke a drop implementation for a type, or does nothing if no
    /// implementation was registered.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn invoke_override<'ctx, 'this>(
        context: &'ctx Context,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        module: &Module<'ctx>,
        init_block: &'this Block<'ctx>,
        block: &'this Block<'ctx>,
        location: Location<'ctx>,
        metadata: &mut MetadataStorage,
        id: &ConcreteTypeId,
        value: Value<'ctx, 'this>,
    ) -> Result<()> {
        if Self::is_overriden(metadata, id) {
            let ty = registry.build_type(context, module, metadata, id)?;
            let sierra_ty = registry.get_type(id)?;
            let is_memory_allocated = sierra_ty.is_memory_allocated(registry)?;

            // From memory allocated types, the drop function receives a pointer
            // as argument, so we need to alloc the given value onto the stack
            // and pass a pointer to it instead.
            let value = if is_memory_allocated {
                // The init_block is guaranteed to not be executed multiple
                // times on tail-recursive functions.
                let value_ptr = init_block.alloca1(
                    context,
                    location,
                    ty,
                    sierra_ty.layout(registry)?.align(),
                )?;
                block.store(context, location, value_ptr, value)?;
                value_ptr
            } else {
                value
            };

            block.append_operation(func::call(
                context,
                FlatSymbolRefAttribute::new(context, &format!("drop${}", id.id)),
                &[value],
                &[],
                location,
            ));
        }

        Ok(())
    }
}
