use color_eyre::Result;
use melior_next::ir::{Block, OperationRef, Region, Value};

use crate::compiler::fn_attributes::FnAttributes;
use crate::{
    compiler::{Compiler, Storage},
    sierra_type::SierraType,
    utility::create_fn_signature,
};

impl<'ctx> Compiler<'ctx> {
    fn create_array_len_impl(
        &'ctx self,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let (len_type, element_type) = if let SierraType::Array { ty: _, len_type, element_type } =
            array_type
        {
            (*len_type, element_type)
        } else {
            panic!("create_array_len_impl should have been passed an Array SierraType, but was instead passed {:?}", array_type)
        };

        let func_name = format!("array_len_impl<{}>", element_type.get_type());

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }

        let region = Region::new();

        let block = region.append_block(Block::new(&[array_type.get_type_location(&self.context)]));

        let array_mlir_type = array_type.get_type();
        let array_value = block.argument(0)?.into();

        let array_len_op = self.op_llvm_extractvalue(&block, 0, array_value, len_type)?;
        let array_len = array_len_op.result(0)?.into();

        self.op_return(&block, &[array_len]);

        let function_type = create_fn_signature(&[array_mlir_type], &[len_type]);

        let func = self.op_func(
            &func_name,
            &function_type,
            vec![region],
            FnAttributes::libfunc(false, true),
        )?;

        storage.helperfuncs.insert(func_name.clone());

        self.module.body().append_operation(func);

        Ok(func_name)
    }

    fn create_array_capacity_impl(
        &'ctx self,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let (len_type, element_type) = if let SierraType::Array { ty: _, len_type, element_type } =
            array_type
        {
            (*len_type, element_type)
        } else {
            panic!("create_array_capacity_impl should have been passed an Array SierraType, but was instead passed {:?}", array_type)
        };

        let func_name = format!("array_capacity_impl<{}>", element_type.get_type());

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }

        let region = Region::new();

        let block = region.append_block(Block::new(&[array_type.get_type_location(&self.context)]));

        let array_mlir_type = array_type.get_type();
        let array_value = block.argument(0)?.into();

        let array_capacity_op = self.op_llvm_extractvalue(&block, 1, array_value, len_type)?;
        let array_capacity = array_capacity_op.result(0)?.into();

        self.op_return(&block, &[array_capacity]);

        let function_type = create_fn_signature(&[array_mlir_type], &[len_type]);

        let func = self.op_func(
            &func_name,
            &function_type,
            vec![region],
            FnAttributes::libfunc(false, true),
        )?;

        storage.helperfuncs.insert(func_name.clone());

        self.module.body().append_operation(func);

        Ok(func_name)
    }
}

impl<'ctx> Compiler<'ctx> {
    pub fn call_array_len_impl<'block>(
        &'ctx self,
        block: &'block Block,
        array: Value,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        let len_type = if let SierraType::Array { ty: _, len_type, element_type: _ } = array_type {
            *len_type
        } else {
            panic!("call_array_len_impl should have been passed an array type, but instead was passed {:?}", array_type)
        };
        let func_name = self.create_array_len_impl(array_type, storage)?;
        self.op_llvm_call(block, &func_name, &[array], &[len_type])
    }

    pub fn call_array_capacity_impl<'block>(
        &'ctx self,
        block: &'block Block,
        array: Value,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        let len_type = if let SierraType::Array { ty: _, len_type, element_type: _ } = array_type {
            *len_type
        } else {
            panic!("call_array_capacity_impl should have been passed an array type, but instead was passed {:?}", array_type)
        };
        let func_name = self.create_array_capacity_impl(array_type, storage)?;
        self.op_llvm_call(block, &func_name, &[array], &[len_type])
    }
}
