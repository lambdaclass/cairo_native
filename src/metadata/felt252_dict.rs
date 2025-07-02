use super::{drop_overrides::DropOverridesMeta, MetadataStorage};
use crate::{
    error::{Error, Result},
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    helpers::{BuiltinBlockExt, LlvmBlockExt},
    ir::{
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        Attribute, Block, BlockLike, Identifier, Location, Module, Region,
    },
    Context,
};
use std::collections::{hash_map::Entry, HashMap};

#[derive(Clone, Debug, Default)]
pub struct Felt252DictOverrides {
    drop_overrides: HashMap<ConcreteTypeId, String>,
}

impl Felt252DictOverrides {
    pub fn get_drop_fn(&self, type_id: &ConcreteTypeId) -> Option<&str> {
        self.drop_overrides.get(type_id).map(String::as_str)
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
