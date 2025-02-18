use super::{drop_overrides::DropOverridesMeta, dup_overrides::DupOverridesMeta, MetadataStorage};
use crate::{
    error::{Error, Result},
    utils::{BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        Attribute, Block, BlockLike, Identifier, Location, Module, Region,
    },
    Context,
};
use std::collections::{hash_map::Entry, HashMap};

#[derive(Clone, Debug, Default)]
pub struct Felt252DictOverrides {
    dup_overrides: HashMap<ConcreteTypeId, String>,
    drop_overrides: HashMap<ConcreteTypeId, String>,
}

impl Felt252DictOverrides {
    pub fn get_dup_fn(&self, type_id: &ConcreteTypeId) -> Option<&str> {
        self.dup_overrides.get(type_id).map(String::as_str)
    }

    pub fn get_drop_fn(&self, type_id: &ConcreteTypeId) -> Option<&str> {
        self.drop_overrides.get(type_id).map(String::as_str)
    }

    pub fn build_dup_fn<'ctx>(
        &mut self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        metadata: &mut MetadataStorage,
        type_id: &ConcreteTypeId,
    ) -> Result<Option<FlatSymbolRefAttribute<'ctx>>> {
        let location = Location::unknown(context);

        let inner_ty = registry.build_type(context, module, metadata, type_id)?;
        Ok(match metadata.get::<DupOverridesMeta>() {
            Some(dup_overrides_meta) if dup_overrides_meta.is_overriden(type_id) => {
                let dup_fn_symbol = format!("dup${}$item", type_id.id);
                let flat_symbol_ref = FlatSymbolRefAttribute::new(context, &dup_fn_symbol);

                if let Entry::Vacant(entry) = self.dup_overrides.entry(type_id.clone()) {
                    let dup_fn_symbol = entry.insert(dup_fn_symbol);

                    let region = Region::new();
                    let entry = region.append_block(Block::new(&[
                        (llvm::r#type::pointer(context, 0), location),
                        (llvm::r#type::pointer(context, 0), location),
                    ]));

                    let source_ptr = entry.arg(0)?;
                    let target_ptr = entry.arg(1)?;

                    let value = entry.load(context, location, source_ptr, inner_ty)?;
                    let values = dup_overrides_meta
                        .invoke_override(context, &entry, location, type_id, value)?;
                    entry.store(context, location, source_ptr, values.0)?;
                    entry.store(context, location, target_ptr, values.1)?;

                    entry.append_operation(llvm::r#return(None, location));

                    module.body().append_operation(llvm::func(
                        context,
                        StringAttribute::new(context, dup_fn_symbol),
                        TypeAttribute::new(llvm::r#type::function(
                            llvm::r#type::void(context),
                            &[
                                llvm::r#type::pointer(context, 0),
                                llvm::r#type::pointer(context, 0),
                            ],
                            false,
                        )),
                        region,
                        &[
                            (
                                Identifier::new(context, "sym_visibility"),
                                StringAttribute::new(context, "public").into(),
                            ),
                            (
                                Identifier::new(context, "linkage"),
                                Attribute::parse(context, "#llvm.linkage<private>")
                                    .ok_or(Error::ParseAttributeError)?,
                            ),
                        ],
                        location,
                    ));
                }

                Some(flat_symbol_ref)
            }
            _ => None,
        })
    }

    pub fn build_drop_fn<'ctx>(
        &mut self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        metadata: &mut MetadataStorage,
        type_id: &ConcreteTypeId,
    ) -> Result<Option<FlatSymbolRefAttribute<'ctx>>> {
        let location = Location::unknown(context);

        let inner_ty = registry.build_type(context, module, metadata, type_id)?;
        Ok(match metadata.get::<DropOverridesMeta>() {
            Some(drop_overrides_meta) if drop_overrides_meta.is_overriden(type_id) => {
                let drop_fn_symbol = format!("drop${}$item", type_id.id);
                let flat_symbol_ref = FlatSymbolRefAttribute::new(context, &drop_fn_symbol);

                if let Entry::Vacant(entry) = self.drop_overrides.entry(type_id.clone()) {
                    let drop_fn_symbol = entry.insert(drop_fn_symbol);

                    let region = Region::new();
                    let entry = region
                        .append_block(Block::new(&[(llvm::r#type::pointer(context, 0), location)]));

                    let value = entry.load(context, location, entry.arg(0)?, inner_ty)?;
                    drop_overrides_meta
                        .invoke_override(context, &entry, location, type_id, value)?;

                    entry.append_operation(llvm::r#return(None, location));

                    module.body().append_operation(llvm::func(
                        context,
                        StringAttribute::new(context, drop_fn_symbol),
                        TypeAttribute::new(llvm::r#type::function(
                            llvm::r#type::void(context),
                            &[llvm::r#type::pointer(context, 0)],
                            false,
                        )),
                        region,
                        &[
                            (
                                Identifier::new(context, "sym_visibility"),
                                StringAttribute::new(context, "public").into(),
                            ),
                            (
                                Identifier::new(context, "llvm.linkage"),
                                Attribute::parse(context, "#llvm.linkage<private>")
                                    .ok_or(Error::ParseAttributeError)?,
                            ),
                        ],
                        location,
                    ));
                }

                Some(flat_symbol_ref)
            }
            _ => None,
        })
    }
}
