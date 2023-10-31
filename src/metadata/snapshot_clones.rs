use super::MetadataStorage;
use crate::{
    error::{libfuncs, CoreTypeBuilderError},
    libfuncs::{LibfuncBuilder, LibfuncHelper},
    types::{TypeBuilder, WithSelf},
};
use cairo_lang_sierra::{
    extensions::{GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Block, Location, Value},
    Context,
};
use std::{collections::HashMap, sync::Arc};

pub type CloneFn<TType, TLibfunc, P> = for<'ctx, 'this> fn(
    &'ctx Context,
    &ProgramRegistry<TType, TLibfunc>,
    &'this Block<'ctx>,
    Location<'ctx>,
    &LibfuncHelper<'ctx, 'this>,
    &mut MetadataStorage,
    WithSelf<P>,
) -> libfuncs::Result<Value<'ctx, 'this>>;

type CloneFnWrapper<TType, TLibfunc> = Arc<
    dyn for<'ctx, 'this> Fn(
        &'ctx Context,
        &ProgramRegistry<TType, TLibfunc>,
        &'this Block<'ctx>,
        Location<'ctx>,
        &LibfuncHelper<'ctx, 'this>,
        &mut MetadataStorage,
    ) -> libfuncs::Result<Value<'ctx, 'this>>,
>;

// #[derive(Debug)]
pub struct SnapshotClonesMeta<TType, TLibfunc>
where
    TType: 'static + GenericType,
    TLibfunc: 'static + GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete:
        LibfuncBuilder<TType, TLibfunc, Error = libfuncs::Error>,
{
    mappings: HashMap<ConcreteTypeId, CloneFnWrapper<TType, TLibfunc>>,
}

impl<TType, TLibfunc> SnapshotClonesMeta<TType, TLibfunc>
where
    TType: 'static + GenericType,
    TLibfunc: 'static + GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete:
        LibfuncBuilder<TType, TLibfunc, Error = libfuncs::Error>,
{
    pub fn register<P>(
        &mut self,
        id: ConcreteTypeId,
        handler: CloneFn<TType, TLibfunc, P>,
        params: P,
    ) where
        P: 'static,
    {
        let self_ty = id.clone();
        self.mappings.insert(
            id,
            Arc::new(
                move |context, registry, entry, location, helper, metadata| {
                    handler(
                        context,
                        registry,
                        entry,
                        location,
                        helper,
                        metadata,
                        WithSelf::new(&self_ty, &params),
                    )
                },
            ),
        );
    }

    pub fn wrap_invoke(&self, id: &ConcreteTypeId) -> Option<CloneFnWrapper<TType, TLibfunc>> {
        self.mappings.get(id).cloned()
    }
}

impl<TType, TLibfunc> Default for SnapshotClonesMeta<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete:
        LibfuncBuilder<TType, TLibfunc, Error = libfuncs::Error>,
{
    fn default() -> Self {
        Self {
            mappings: Default::default(),
        }
    }
}
