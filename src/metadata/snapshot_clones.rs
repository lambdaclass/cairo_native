//use super::MetadataStorage;
use super::MetadataStorage;
//use crate::{error::Result, libfuncs::LibfuncHelper, types::WithSelf};
use crate::{error::Result, libfuncs::LibfuncHelper, types::WithSelf};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::core::{CoreLibfunc, CoreType},
    extensions::core::{CoreLibfunc, CoreType},
//    ids::ConcreteTypeId,
    ids::ConcreteTypeId,
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    ir::{Block, Location, Value},
    ir::{Block, Location, Value},
//    Context,
    Context,
//};
};
//use std::{collections::HashMap, sync::Arc};
use std::{collections::HashMap, sync::Arc};
//

//pub type CloneFn<P> = for<'ctx, 'this> fn(
pub type CloneFn<P> = for<'ctx, 'this> fn(
//    &'ctx Context,
    &'ctx Context,
//    &ProgramRegistry<CoreType, CoreLibfunc>,
    &ProgramRegistry<CoreType, CoreLibfunc>,
//    &'this Block<'ctx>,
    &'this Block<'ctx>,
//    Location<'ctx>,
    Location<'ctx>,
//    &LibfuncHelper<'ctx, 'this>,
    &LibfuncHelper<'ctx, 'this>,
//    &mut MetadataStorage,
    &mut MetadataStorage,
//    WithSelf<P>,
    WithSelf<P>,
//    Value<'ctx, 'this>,
    Value<'ctx, 'this>,
//) -> Result<(&'this Block<'ctx>, Value<'ctx, 'this>)>;
) -> Result<(&'this Block<'ctx>, Value<'ctx, 'this>)>;
//

//type CloneFnWrapper = Arc<
type CloneFnWrapper = Arc<
//    dyn for<'ctx, 'this> Fn(
    dyn for<'ctx, 'this> Fn(
//        &'ctx Context,
        &'ctx Context,
//        &ProgramRegistry<CoreType, CoreLibfunc>,
        &ProgramRegistry<CoreType, CoreLibfunc>,
//        &'this Block<'ctx>,
        &'this Block<'ctx>,
//        Location<'ctx>,
        Location<'ctx>,
//        &LibfuncHelper<'ctx, 'this>,
        &LibfuncHelper<'ctx, 'this>,
//        &mut MetadataStorage,
        &mut MetadataStorage,
//        Value<'this, 'ctx>,
        Value<'this, 'ctx>,
//    ) -> Result<(&'this Block<'ctx>, Value<'ctx, 'this>)>,
    ) -> Result<(&'this Block<'ctx>, Value<'ctx, 'this>)>,
//>;
>;
//

//#[derive(Default)]
#[derive(Default)]
//pub struct SnapshotClonesMeta {
pub struct SnapshotClonesMeta {
//    mappings: HashMap<ConcreteTypeId, CloneFnWrapper>,
    mappings: HashMap<ConcreteTypeId, CloneFnWrapper>,
//}
}
//

//impl SnapshotClonesMeta {
impl SnapshotClonesMeta {
//    pub fn register<P>(&mut self, id: ConcreteTypeId, handler: CloneFn<P>, params: P)
    pub fn register<P>(&mut self, id: ConcreteTypeId, handler: CloneFn<P>, params: P)
//    where
    where
//        P: 'static,
        P: 'static,
//    {
    {
//        let self_ty = id.clone();
        let self_ty = id.clone();
//        self.mappings.insert(
        self.mappings.insert(
//            id,
            id,
//            Arc::new(
            Arc::new(
//                move |context, registry, entry, location, helper, metadata, value| {
                move |context, registry, entry, location, helper, metadata, value| {
//                    handler(
                    handler(
//                        context,
                        context,
//                        registry,
                        registry,
//                        entry,
                        entry,
//                        location,
                        location,
//                        helper,
                        helper,
//                        metadata,
                        metadata,
//                        WithSelf::new(&self_ty, &params),
                        WithSelf::new(&self_ty, &params),
//                        value,
                        value,
//                    )
                    )
//                },
                },
//            ),
            ),
//        );
        );
//    }
    }
//

//    pub fn wrap_invoke(&self, id: &ConcreteTypeId) -> Option<CloneFnWrapper> {
    pub fn wrap_invoke(&self, id: &ConcreteTypeId) -> Option<CloneFnWrapper> {
//        self.mappings.get(id).cloned()
        self.mappings.get(id).cloned()
//    }
    }
//}
}
