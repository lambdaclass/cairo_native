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
//! function to allow self-referencing types. If we inlined the drop implementations,
//! self-referencing types would generate infinite code thus overflowing the stack when generating
//! code.
//!
//! The generated functions are not public (they are internal) and follow this naming convention:
//!
//!     dup${type id}
//!
//! where `{type id}` is the numeric value of the `ConcreteTypeId`.

use super::MetadataStorage;
use crate::{error::Result, utils::ProgramRegistryExt};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::func,
    ir::{
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        r#type::FunctionType,
        Block, Location, Module, Region, Value, ValueLike,
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
                let ty = registry.build_type(context, module, registry, metadata, id)?;
                module.body().append_operation(func::func(
                    context,
                    StringAttribute::new(context, &format!("dup${}", id.id)),
                    TypeAttribute::new(FunctionType::new(context, &[ty], &[ty, ty]).into()),
                    region,
                    &[],
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

    /// Returns whether a type has a registered clone implementation.
    pub(crate) fn is_overriden(&self, id: &ConcreteTypeId) -> bool {
        self.overriden_types.contains(id)
    }

    /// Generates code to invoke a clone implementation for a type, or just returns the same value
    /// twice if no implementation was registered.
    pub(crate) fn invoke_override<'ctx, 'this>(
        &self,
        context: &'ctx Context,
        block: &'this Block<'ctx>,
        location: Location<'ctx>,
        id: &ConcreteTypeId,
        value: Value<'ctx, 'this>,
    ) -> Result<(Value<'ctx, 'this>, Value<'ctx, 'this>)> {
        Ok(if self.overriden_types.contains(id) {
            let res = block.append_operation(func::call(
                context,
                FlatSymbolRefAttribute::new(context, &format!("dup${}", id.id)),
                &[value],
                &[value.r#type(), value.r#type()],
                location,
            ));

            (res.result(0)?.into(), res.result(1)?.into())
        } else {
            (value, value)
        })
    }
}
