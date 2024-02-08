use super::MetadataStorage;
use crate::{error::libfuncs, libfuncs::LibfuncHelper, types::WithSelf};
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

pub type CloneFn<P> = for<'ctx, 'this> fn(
    &'ctx Context,
    &ProgramRegistry<CoreType, CoreLibfunc>,
    &'this Block<'ctx>,
    Location<'ctx>,
    &LibfuncHelper<'ctx, 'this>,
    &mut MetadataStorage,
    WithSelf<P>,
    Value<'ctx, 'this>,
) -> libfuncs::Result<Value<'ctx, 'this>>;

type CloneFnWrapper = Arc<
    dyn for<'ctx, 'this> Fn(
        &'ctx Context,
        &ProgramRegistry<CoreType, CoreLibfunc>,
        &'this Block<'ctx>,
        Location<'ctx>,
        &LibfuncHelper<'ctx, 'this>,
        &mut MetadataStorage,
        Value<'this, 'ctx>,
    ) -> libfuncs::Result<Value<'ctx, 'this>>,
>;

#[derive(Default)]
pub struct SnapshotClonesMeta {
    mappings: HashMap<ConcreteTypeId, CloneFnWrapper>,
}

impl SnapshotClonesMeta {
    pub fn register<P>(&mut self, id: ConcreteTypeId, handler: CloneFn<P>, params: P)
    where
        P: 'static,
    {
        let self_ty = id.clone();
        self.mappings.insert(
            id,
            Arc::new(
                move |context, registry, entry, location, helper, metadata, value| {
                    handler(
                        context,
                        registry,
                        entry,
                        location,
                        helper,
                        metadata,
                        WithSelf::new(&self_ty, &params),
                        value,
                    )
                },
            ),
        );
    }

    pub fn wrap_invoke(&self, id: &ConcreteTypeId) -> Option<CloneFnWrapper> {
        self.mappings.get(id).cloned()
    }
}
