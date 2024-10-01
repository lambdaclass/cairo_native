use super::MetadataStorage;
use crate::{error::Result, libfuncs::LibfuncHelper, types::WithSelf};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Block, Location, Value},
    Context,
};
use std::{collections::HashMap, sync::Arc};

type CloneFn<P> = for<'ctx, 'this> fn(
    &'ctx Context,
    &ProgramRegistry<CoreType, CoreLibfunc>,
    &'this Block<'ctx>,
    Location<'ctx>,
    &LibfuncHelper<'ctx, 'this>,
    &mut MetadataStorage,
    WithSelf<P>,
    Value<'ctx, 'this>,
) -> Result<(&'this Block<'ctx>, Value<'ctx, 'this>)>;

type CloneFnWrapper = Arc<
    dyn for<'ctx, 'this> Fn(
        &'ctx Context,
        &ProgramRegistry<CoreType, CoreLibfunc>,
        &'this Block<'ctx>,
        Location<'ctx>,
        &LibfuncHelper<'ctx, 'this>,
        &mut MetadataStorage,
        Value<'this, 'ctx>,
    ) -> Result<(&'this Block<'ctx>, Value<'ctx, 'this>)>,
>;

#[derive(Default)]
pub struct SnapshotClonesMeta {
    mappings: HashMap<ConcreteTypeId, CloneFnWrapper>,
    partials: Vec<ConcreteTypeId>,
}

impl SnapshotClonesMeta {
    pub(crate) fn register_with<P>(
        metadata: &mut MetadataStorage,
        id: ConcreteTypeId,
        f: impl FnOnce(&mut MetadataStorage) -> Result<Option<(CloneFn<P>, P)>>,
    ) -> Result<()>
    where
        P: 'static,
    {
        {
            let snapshot_clones_meta = metadata.get_or_insert_with(Self::default);
            if snapshot_clones_meta.is_registered(&id) {
                return Ok(());
            }

            snapshot_clones_meta.partials.push(id);
        }

        let result = f(metadata)?;
        {
            // This unwrap is unreachble because the meta was created just before if it wasn't
            // already present.
            let snapshot_clones_meta = metadata.get_mut::<Self>().unwrap();

            let self_id = snapshot_clones_meta.partials.pop().unwrap();
            if let Some((clone_fn, params)) = result {
                snapshot_clones_meta.mappings.insert(
                    self_id.clone(),
                    Arc::new(
                        move |context, registry, entry, location, helper, metadata, value| {
                            clone_fn(
                                context,
                                registry,
                                entry,
                                location,
                                helper,
                                metadata,
                                WithSelf::new(&self_id, &params),
                                value,
                            )
                        },
                    ),
                );
            }
        }

        Ok(())
    }

    pub(crate) fn register_dup(&mut self, id: ConcreteTypeId, from_id: &ConcreteTypeId) {
        assert!(
            !self.is_registered(&id),
            "attempting to register a clone impl that is already registered",
        );

        if let Some(clone_fn) = self.mappings.get(from_id) {
            self.mappings.insert(id, clone_fn.clone());
        }
    }

    pub(crate) fn is_registered(&self, id: &ConcreteTypeId) -> bool {
        self.mappings.contains_key(id) || self.partials.contains(id)
    }

    pub(crate) fn wrap_invoke(&self, id: &ConcreteTypeId) -> Option<CloneFnWrapper> {
        self.mappings.get(id).cloned()
    }
}
