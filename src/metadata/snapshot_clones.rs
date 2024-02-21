//! # Snapshot clones database
//!
//! The `snapshot_take` libfunc doesn't know which types implement `Copy` and which need to be
//! `Clone`d. This metadata provides `Clone` implementations for those types that need it. The
//! absence of a `Clone` implementation in this database means that the type should be `Copy`able.

use super::MetadataStorage;
use crate::{error::builders, libfuncs::LibfuncHelper, types::WithSelf};
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

/// The signature for a clone MLIR builder.
pub type CloneFn<P> = for<'ctx, 'this> fn(
    &'ctx Context,
    &ProgramRegistry<CoreType, CoreLibfunc>,
    &'this Block<'ctx>,
    Location<'ctx>,
    &LibfuncHelper<'ctx, 'this>,
    &mut MetadataStorage,
    WithSelf<P>,
    Value<'ctx, 'this>,
) -> builders::Result<Value<'ctx, 'this>>;

type CloneFnWrapper = Arc<
    dyn for<'ctx, 'this> Fn(
        &'ctx Context,
        &ProgramRegistry<CoreType, CoreLibfunc>,
        &'this Block<'ctx>,
        Location<'ctx>,
        &LibfuncHelper<'ctx, 'this>,
        &mut MetadataStorage,
        Value<'this, 'ctx>,
    ) -> builders::Result<Value<'ctx, 'this>>,
>;

/// The snapshot clones metadata.
#[derive(Default)]
pub struct SnapshotClonesMeta {
    mappings: HashMap<ConcreteTypeId, CloneFnWrapper>,
}

impl SnapshotClonesMeta {
    /// Register a clone implementation builder for a given type.
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

    /// Return the clone implementation builder if present.
    pub fn wrap_invoke(&self, id: &ConcreteTypeId) -> Option<CloneFnWrapper> {
        self.mappings.get(id).cloned()
    }
}
