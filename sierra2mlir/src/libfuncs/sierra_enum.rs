use cairo_lang_sierra::program::{GenericArg, LibfuncDeclaration};
use color_eyre::Result;
use melior_next::ir::Value;

use crate::compiler::fn_attributes::FnAttributes;
use crate::utility::get_type_id;
use crate::{
    compiler::{Compiler, Storage},
    sierra_type::SierraType,
};

use super::lib_func_def::SierraLibFunc;

impl<'ctx> Compiler<'ctx> {
    pub fn create_libfunc_enum_init(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        let enum_arg_type = storage
            .types
            .get(&get_type_id(&func_decl.long_id.generic_args[0]))
            .cloned()
            .expect("type to exist");

        let enum_tag = match &func_decl.long_id.generic_args[1] {
            GenericArg::Value(tag) => tag.to_string(),
            _ => unreachable!(),
        };

        let (enum_mlir_type, tag_type, variant_types) =
            if let SierraType::Enum { ty, tag_type, variants_types, .. } = &enum_arg_type {
                (*ty, tag_type, variants_types)
            } else {
                return Err(color_eyre::eyre::eyre!(
                    "found non enum type passed to enum_init: {:?}",
                    enum_arg_type
                ));
            };

        let enum_tag_idx: usize = enum_tag.parse().unwrap();
        let variant_sierra_type = &variant_types[enum_tag_idx];

        let block = self.new_block(&[variant_sierra_type.get_type()]);

        let enum_variant_value = block.argument(0)?;

        // allocate the enum, generic form
        let enum_alloca_op = self.op_llvm_alloca(&block, enum_mlir_type, 1)?;
        let enum_ptr: Value = enum_alloca_op.result(0)?.into();

        let tag_op = self.op_const(&block, &enum_tag, *tag_type);
        let tag_op_value = tag_op.result(0)?;

        let tag_ptr_op = self.op_llvm_gep(&block, &[0, 0], enum_ptr, enum_mlir_type)?;
        let tag_ptr = tag_ptr_op.result(0)?;

        // store the tag
        self.op_llvm_store(&block, tag_op_value.into(), tag_ptr.into())?;

        // get the enum variant type for GEP.
        let variant_enum_type =
            self.llvm_struct_type(&[self.u16_type(), variant_sierra_type.get_type()], false);

        let variant_ptr_op = self.op_llvm_gep(&block, &[0, 1], enum_ptr, variant_enum_type)?;
        let variant_ptr = variant_ptr_op.result(0)?;

        self.op_llvm_store(&block, enum_variant_value.into(), variant_ptr.into())?;

        let enum_value_op = self.op_llvm_load(&block, enum_ptr, enum_mlir_type)?;

        self.op_return(&block, &[enum_value_op.result(0)?.into()]);

        storage.libfuncs.insert(
            id.clone(),
            SierraLibFunc::create_function_all_args(
                vec![variant_sierra_type.clone()],
                vec![enum_arg_type.clone()],
            ),
        );

        self.create_function(
            &id,
            vec![block],
            &[enum_mlir_type],
            FnAttributes::libfunc(false, true),
        )
    }
}
