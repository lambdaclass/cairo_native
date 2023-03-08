use cairo_lang_sierra::{ids::GenericTypeId, program::GenericArg};
use melior::ir::Type;
use tracing::debug;

use crate::compiler::{Compiler, Storage};

impl<'ctx> Compiler<'ctx> {
    pub fn process_types(&'ctx self, storage: &mut Storage<'ctx>) {
        for type_decl in &self.program.type_declarations {
            let id = type_decl.id.id;
            let name = type_decl.long_id.generic_id.0.as_str();
            debug!(name, "processing type decl");

            match name {
                "felt" => {
                    let ty = self.felt_type();
                    storage.types.insert(id, ty);
                }
                "NonZero" => {
                    let gen_arg = type_decl
                        .long_id
                        .generic_args
                        .get(0)
                        .expect("should have 1 generic arg");
                    let gen_arg_ty = match gen_arg {
                        GenericArg::Type(gen_arg_typeid) => storage
                            .types
                            .get(&gen_arg_typeid.id)
                            .expect("type should exist"),
                        _ => todo!(),
                    };
                    storage.types.insert(id, *gen_arg_ty);
                }
                _ => debug!(?type_decl, "unhandled type"),
            }
        }

        debug!(types = ?storage.types, "processed")
    }
}
