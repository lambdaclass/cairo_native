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
pub struct DupOverrideMeta {
    overriden_types: HashSet<ConcreteTypeId>,
}

impl DupOverrideMeta {
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

// use super::MetadataStorage;
// use crate::{error::Result, libfuncs::LibfuncHelper, types::WithSelf};
// use cairo_lang_sierra::{
//     extensions::core::{CoreLibfunc, CoreType},
//     ids::ConcreteTypeId,
//     program_registry::ProgramRegistry,
// };
// use melior::{
//     ir::{Block, Location, Value},
//     Context,
// };
// use std::{collections::HashMap, sync::Arc};
//
// type DupFn<P> = for<'ctx, 'this> fn(
//     &'ctx Context,
//     &ProgramRegistry<CoreType, CoreLibfunc>,
//     &'this Block<'ctx>,
//     Location<'ctx>,
//     &LibfuncHelper<'ctx, 'this>,
//     &mut MetadataStorage,
//     WithSelf<P>,
//     Value<'ctx, 'this>,
// ) -> Result<(&'this Block<'ctx>, Value<'ctx, 'this>)>;
//
// type DupFnWrapper = Arc<
//     dyn for<'ctx, 'this> Fn(
//         &'ctx Context,
//         &ProgramRegistry<CoreType, CoreLibfunc>,
//         &'this Block<'ctx>,
//         Location<'ctx>,
//         &LibfuncHelper<'ctx, 'this>,
//         &mut MetadataStorage,
//         Value<'this, 'ctx>,
//     ) -> Result<(&'this Block<'ctx>, Value<'ctx, 'this>)>,
// >;

// #[derive(Default)]
// pub struct DupOverrideMeta {
//     mappings: HashMap<ConcreteTypeId, DupFnWrapper>,
//     partials: Vec<ConcreteTypeId>,
// }

// impl DupOverrideMeta {
//     pub(crate) fn register_with<P>(
//         metadata: &mut MetadataStorage,
//         id: ConcreteTypeId,
//         f: impl FnOnce(&mut MetadataStorage) -> Result<Option<(DupFn<P>, P)>>,
//     ) -> Result<()>
//     where
//         P: 'static,
//     {
//         {
//             let dup_override_meta = metadata.get_or_insert_with(Self::default);
//             if dup_override_meta.is_registered(&id) {
//                 return Ok(());
//             }

//             dup_override_meta.partials.push(id);
//         }

//         let result = f(metadata)?;
//         {
//             // This unwrap is unreachble because the meta was created just before if it wasn't
//             // already present.
//             let dup_override_meta = metadata.get_mut::<Self>().unwrap();

//             let self_id = dup_override_meta.partials.pop().unwrap();
//             if let Some((clone_fn, params)) = result {
//                 dup_override_meta.mappings.insert(
//                     self_id.clone(),
//                     Arc::new(
//                         move |context, registry, entry, location, helper, metadata, value| {
//                             clone_fn(
//                                 context,
//                                 registry,
//                                 entry,
//                                 location,
//                                 helper,
//                                 metadata,
//                                 WithSelf::new(&self_id, &params),
//                                 value,
//                             )
//                         },
//                     ),
//                 );
//             }
//         }

//         Ok(())
//     }

//     pub(crate) fn register_dup(&mut self, id: ConcreteTypeId, from_id: &ConcreteTypeId) {
//         assert!(
//             !self.is_registered(&id),
//             "attempting to register a clone impl that is already registered",
//         );

//         if let Some(clone_fn) = self.mappings.get(from_id) {
//             self.mappings.insert(id, clone_fn.clone());
//         }
//     }

//     pub(crate) fn is_registered(&self, id: &ConcreteTypeId) -> bool {
//         self.mappings.contains_key(id) || self.partials.contains(id)
//     }

//     pub(crate) fn wrap_invoke(&self, id: &ConcreteTypeId) -> Option<DupFnWrapper> {
//         self.mappings.get(id).cloned()
//     }
// }
