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
        Block, Location, Module, Region, Value,
    },
    Context,
};
use std::collections::HashSet;

#[derive(Debug, Default)]
pub struct DropOverridesMeta {
    overriden_types: HashSet<ConcreteTypeId>,
}

impl DropOverridesMeta {
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
                let ty = registry.build_type(context, module, registry, metadata, id)?;
                module.body().append_operation(func::func(
                    context,
                    StringAttribute::new(context, &format!("drop${}", id.id)),
                    TypeAttribute::new(FunctionType::new(context, &[ty], &[]).into()),
                    region,
                    &[],
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

    /// Returns whether a type has a registered clone implementation.
    pub(crate) fn is_overriden(&self, id: &ConcreteTypeId) -> bool {
        self.overriden_types.contains(id)
    }

    pub(crate) fn invoke_override<'ctx, 'this>(
        &self,
        context: &'ctx Context,
        block: &'this Block<'ctx>,
        location: Location<'ctx>,
        id: &ConcreteTypeId,
        value: Value<'ctx, 'this>,
    ) -> Result<()> {
        if self.overriden_types.contains(id) {
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
