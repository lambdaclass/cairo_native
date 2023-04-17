use cairo_lang_sierra::program::{GenericArg, LibfuncDeclaration};
use color_eyre::Result;
use melior_next::ir::{operation, Block, BlockRef, Location, Region, Value};

use crate::{
    compiler::{CmpOp, Compiler, FnAttributes, SierraType, Storage},
    utility::create_fn_signature,
};

use super::lib_func_def::SierraLibFunc;

impl<'ctx> Compiler<'ctx> {
    pub fn create_libfunc_enum_init(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<()> {
        let id = func_decl.id.debug_name.as_ref().unwrap().to_string();

        let (is_panic, enum_arg_type) = match &func_decl.long_id.generic_args[0] {
            GenericArg::Type(type_id) => (
                type_id.debug_name.as_deref().unwrap().starts_with("core::PanicResult"),
                storage.types.get(&type_id.id.to_string()).cloned().expect("type to exist"),
            ),
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
            let tag_op_value = tag_op.result(0)?.into();

            // only panic if the enum is a PanicResult, and the tag is Error.
            let block = if is_panic {
                let continue_block = region.append_block(Block::new(&[]));
                let trap_block = region.append_block(Block::new(&[]));

                let const_0_op = self.op_u16_const(&block, "0");
                let const_0 = const_0_op.result(0)?.into();

                let cond_op = self.op_cmp(&block, CmpOp::Equal, tag_op_value, const_0);

                self.op_cond_br(
                    &block,
                    cond_op.result(0)?.into(),
                    &continue_block,
                    &trap_block,
                    &[],
                    &[],
                );

                trap_block.append_operation(
                    operation::Builder::new(
                        "llvm.call_intrinsic",
                        Location::unknown(&self.context),
                    )
                    .add_attributes(&[self.named_attribute("intrin", "\"llvm.trap\"").unwrap()])
                    .build(),
                );
                self.op_unreachable(&trap_block);
                continue_block
            } else {
                block
            };

            let tag_ptr_op = self.op_llvm_gep(&block, 0, enum_ptr, *ty)?;
            let tag_ptr = tag_ptr_op.result(0)?;

            self.op_llvm_store(&block, tag_op_value, tag_ptr.into())?;

            let variant_ptr_op = self.op_llvm_gep(&block, 1, enum_ptr, *ty)?;
            let variant_ptr = variant_ptr_op.result(0)?;

            self.op_llvm_store(&block, enum_variant_value.into(), variant_ptr.into())?;

            let enum_value_op = self.op_llvm_load(&block, enum_ptr, *ty)?;

            self.op_return(&block, &[enum_value_op.result(0)?.into()]);

            let function_type = create_fn_signature(&[variant_sierra_type.get_type()], &[*ty]);

            let func = self.op_func(
                &id,
                &function_type,
                vec![region],
                FnAttributes::libfunc(false, true),
            )?;

            storage.libfuncs.insert(
                id,
                SierraLibFunc::create_function_all_args(
                    vec![variant_sierra_type.clone()],
                    vec![enum_arg_type],
                ),
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
