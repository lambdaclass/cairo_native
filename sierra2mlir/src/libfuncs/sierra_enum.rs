use std::{cell::RefCell, cmp::Ordering, rc::Rc};

use cairo_lang_sierra::program::{GenericArg, LibfuncDeclaration};
use color_eyre::Result;
use itertools::Itertools;
use melior_next::ir::{Block, BlockRef, Location, Region, Type, TypeLike, Value};
use tracing::debug;

use crate::{
    compiler::{CmpOp, Compiler, FunctionDef, SierraType, Storage},
    statements::create_fn_signature,
};

impl<'ctx> Compiler<'ctx> {
    pub fn create_libfunc_enum_init(
        &'ctx self,
        func_decl: &LibfuncDeclaration,
        parent_block: BlockRef<'ctx>,
        storage: Rc<RefCell<Storage<'ctx>>>,
    ) -> Result<()> {
        let id = Self::normalize_func_name(func_decl.id.debug_name.as_ref().unwrap().as_str())
            .to_string();

        let enum_arg_type = match &func_decl.long_id.generic_args[0] {
            GenericArg::UserType(_) => todo!(),
            GenericArg::Type(type_id) => {
                let storage = RefCell::borrow(&*storage);
                let ty =
                    storage.types.get(&type_id.id.to_string()).cloned().expect("type to exist");

                ty
            }
            GenericArg::Value(_) => unreachable!(),
            GenericArg::UserFunc(_) => unreachable!(),
            GenericArg::Libfunc(_) => unreachable!(),
        };

        let enum_tag = match &func_decl.long_id.generic_args[1] {
            GenericArg::UserType(_) => unreachable!(),
            GenericArg::Type(_) => unreachable!(),
            GenericArg::Value(tag) => tag.to_string(),
            GenericArg::UserFunc(_) => unreachable!(),
            GenericArg::Libfunc(_) => unreachable!(),
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
            let arg_sierra_type = &variants_types[enum_tag_idx];

            let region = Region::new();
            let block = region
                .append_block(Block::new(&[arg_sierra_type.get_type_location(&self.context)]));

            let enum_alloca_op = self.op_llvm_struct(&block, &[*tag_type, *storage_type]);
            let enum_value: Value = enum_alloca_op.result(0)?.into();

            let tag_op = self.op_const(&block, &enum_tag, *tag_type);
            let tag_op_value = tag_op.result(0)?;

            self.op_llvm_insertvalue(&block, 0, enum_value, tag_op_value.into(), *ty)?;

            // TODO: store the provided value, bitcasting to the array.

            self.op_return(&block, &[enum_value]);

            // get the variant type for this init, it will be the type of the argument.

            let function_type = create_fn_signature(&[arg_sierra_type.get_type()], &[*ty]);

            let func = self.op_func(&id, &function_type, vec![region], false, false)?;

            {
                let mut storage = storage.borrow_mut();
                storage
                    .functions
                    .insert(id, FunctionDef { args: vec![], return_types: vec![enum_arg_type] });
            }

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
