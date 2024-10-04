use crate::{error::Result, metadata::MetadataStorage, types::TypeBuilder};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Module, Type},
    Context,
};
use std::alloc::Layout;

pub trait ProgramRegistryExt {
    fn build_type<'ctx>(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        metadata: &mut MetadataStorage,
        id: &ConcreteTypeId,
    ) -> Result<Type<'ctx>>;

    fn build_type_with_layout<'ctx>(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        metadata: &mut MetadataStorage,
        id: &ConcreteTypeId,
    ) -> Result<(Type<'ctx>, Layout)>;
}

impl ProgramRegistryExt for ProgramRegistry<CoreType, CoreLibfunc> {
    fn build_type<'ctx>(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        metadata: &mut MetadataStorage,
        id: &ConcreteTypeId,
    ) -> Result<Type<'ctx>> {
        registry
            .get_type(id)?
            .build(context, module, registry, metadata, id)
    }

    fn build_type_with_layout<'ctx>(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        metadata: &mut MetadataStorage,
        id: &ConcreteTypeId,
    ) -> Result<(Type<'ctx>, Layout)> {
        let concrete_type = registry.get_type(id)?;

        Ok((
            concrete_type.build(context, module, registry, metadata, id)?,
            concrete_type.layout(registry)?,
        ))
    }
}
