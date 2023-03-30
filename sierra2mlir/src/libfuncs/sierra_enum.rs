use cairo_lang_sierra::program::{GenericArg, LibfuncDeclaration};
use color_eyre::Result;
use melior_next::ir::{Block, BlockRef, Region, Value};

use crate::{
    compiler::{Compiler, FunctionDef, SierraType, Storage},
    utility::create_fn_signature,
};

impl<'ctx> Compiler<'ctx> {
    pub fn create_libfunc_enum_init(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = Self::normalize_func_name(func_decl.id.debug_name.as_ref().unwrap().as_str())
            .to_string();

        let enum_arg_type = match &func_decl.long_id.generic_args[0] {
            GenericArg::Type(type_id) => {
                storage.types.get(&type_id.id.to_string()).cloned().expect("type to exist")
            }
            _ => unreachable!(),
        };

        let enum_tag = match &func_decl.long_id.generic_args[1] {
            GenericArg::Value(tag) => tag.to_string(),
            _ => unreachable!(),
        };

        if let SierraType::Enum {
            ty,
            tag_type,
            storage_bytes_len: _,
            storage_type,
            variants_types,
        } = &enum_arg_type
        {
            let enum_tag_idx: usize = enum_tag.parse().unwrap();
            let variant_sierra_type = &variants_types[enum_tag_idx];

            let region = Region::new();
            let block = region
                .append_block(Block::new(&[variant_sierra_type.get_type_location(&self.context)]));

            let enum_variant_value = block.argument(0)?;

            let enum_alloca_op = self.op_llvm_struct_alloca(&block, &[*tag_type, *storage_type])?;
            let enum_ptr: Value = enum_alloca_op.result(0)?.into();

            let tag_op = self.op_const(&block, &enum_tag, *tag_type);
            let tag_op_value = tag_op.result(0)?;

            let tag_ptr_op = self.op_llvm_gep(&block, 0, enum_ptr, *ty)?;
            let tag_ptr = tag_ptr_op.result(0)?;

            self.op_llvm_store(&block, tag_op_value.into(), tag_ptr.into())?;

            let variant_ptr_op = self.op_llvm_gep(&block, 1, enum_ptr, *ty)?;
            let variant_ptr = variant_ptr_op.result(0)?;

            self.op_llvm_store(&block, enum_variant_value.into(), variant_ptr.into())?;

            let enum_value_op = self.op_llvm_load(&block, enum_ptr, *ty)?;

            self.op_return(&block, &[enum_value_op.result(0)?.into()]);

            let function_type = create_fn_signature(&[variant_sierra_type.get_type()], &[*ty]);

            let func = self.op_func(&id, &function_type, vec![region], false, false)?;

            storage.libfuncs.insert(
                id,
                FunctionDef {
                    args: vec![variant_sierra_type.clone()],
                    return_types: vec![enum_arg_type],
                },
            );

            parent_block.append_operation(func);

            Ok(())
        } else {
            Err(color_eyre::eyre::eyre!(
                "found non enum type passed to enum_init: {:?}",
                enum_arg_type
            ))
        }
    }
}
