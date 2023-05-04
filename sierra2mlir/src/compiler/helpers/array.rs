use color_eyre::Result;
use melior_next::ir::{Block, OperationRef, Value};

use crate::compiler::fn_attributes::FnAttributes;
use crate::{
    compiler::{Compiler, Storage},
    sierra_type::SierraType,
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

        let block = self.new_block(&[array_type.get_type()]);

        let array_value = block.argument(0)?.into();

        let array_len_op = self.op_llvm_extractvalue(&block, 0, array_value, len_type)?;
        let array_len = array_len_op.result(0)?.into();

        self.op_return(&block, &[array_len]);

        storage.helperfuncs.insert(func_name.clone());

        self.create_function(
            &func_name,
            vec![block],
            &[len_type],
            FnAttributes::libfunc(false, true),
        )?;

        Ok(func_name)
    }

    fn create_array_set_len_impl(
        &'ctx self,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let (len_type, element_type) = if let SierraType::Array { len_type, element_type, .. } =
            array_type
        {
            (*len_type, element_type)
        } else {
            panic!("create_array_set_len_impl should have been passed an Array SierraType, but was instead passed {:?}", array_type)
        };

        let func_name = format!("array_set_len_impl<{}>", element_type.get_type());

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }

        let block = self.new_block(&[array_type.get_type(), len_type]);
        let array_value = block.argument(0)?.into();
        let new_len = block.argument(1)?.into();

        let updated_array_op =
            self.op_llvm_insertvalue(&block, 0, array_value, new_len, array_type.get_type())?;
        let updated_array = updated_array_op.result(0)?.into();

        self.op_return(&block, &[updated_array]);

        storage.helperfuncs.insert(func_name.clone());

        self.create_function(
            &func_name,
            vec![block],
            &[array_type.get_type()],
            FnAttributes::libfunc(false, true),
        )?;

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

        let block = self.new_block(&[array_type.get_type()]);

        let array_value = block.argument(0)?.into();

        let array_capacity_op = self.op_llvm_extractvalue(&block, 1, array_value, len_type)?;
        let array_capacity = array_capacity_op.result(0)?.into();

        self.op_return(&block, &[array_capacity]);

        storage.helperfuncs.insert(func_name.clone());

        self.create_function(
            &func_name,
            vec![block],
            &[len_type],
            FnAttributes::libfunc(false, true),
        )?;

        Ok(func_name)
    }

    fn create_array_set_capacity_impl(
        &'ctx self,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let (len_type, element_type) = if let SierraType::Array { len_type, element_type, .. } =
            array_type
        {
            (*len_type, element_type)
        } else {
            panic!("create_array_set_capacity_impl should have been passed an Array SierraType, but was instead passed {:?}", array_type)
        };

        let func_name = format!("array_set_capacity_impl<{}>", element_type.get_type());

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }

        let block = self.new_block(&[array_type.get_type(), len_type]);
        let array_value = block.argument(0)?.into();
        let new_capacity = block.argument(1)?.into();

        let updated_array_op =
            self.op_llvm_insertvalue(&block, 1, array_value, new_capacity, array_type.get_type())?;
        let updated_array = updated_array_op.result(0)?.into();

        self.op_return(&block, &[updated_array]);

        storage.helperfuncs.insert(func_name.clone());

        self.create_function(
            &func_name,
            vec![block],
            &[array_type.get_type()],
            FnAttributes::libfunc(false, true),
        )?;

        Ok(func_name)
    }

    fn create_array_get_unchecked(
        &'ctx self,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let (len_type, element_type) = if let SierraType::Array { ty: _, len_type, element_type } =
            array_type
        {
            (*len_type, element_type.get_type())
        } else {
            panic!("create_array_get_unchecked should have been passed an Array SierraType, but was instead passed {:?}", array_type)
        };

        let func_name = format!("array_get_unchecked<{}>", element_type);

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }
        storage.helperfuncs.insert(func_name.clone());

        let block = self.new_block(&[array_type.get_type(), len_type]);

        let array_value = block.argument(0)?.into();
        let index_value = block.argument(1)?.into();

        let array_data_op =
            self.op_llvm_extractvalue(&block, 2, array_value, self.llvm_ptr_type())?;
        let array_data = array_data_op.result(0)?.into();
        let array_element_ptr_op =
            self.op_llvm_gep_dynamic(&block, &[index_value], array_data, element_type)?;
        let array_element_ptr = array_element_ptr_op.result(0)?.into();
        let array_element_op = self.op_llvm_load(&block, array_element_ptr, element_type)?;
        let array_element = array_element_op.result(0)?.into();

        self.op_return(&block, &[array_element]);

        self.create_function(
            &func_name,
            vec![block],
            &[element_type],
            FnAttributes::libfunc(false, true),
        )?;

        Ok(func_name)
    }

    fn create_array_set_unchecked(
        &'ctx self,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let (len_type, element_type) = if let SierraType::Array { ty: _, len_type, element_type } =
            array_type
        {
            (*len_type, element_type.get_type())
        } else {
            panic!("create_array_set_unchecked should have been passed an Array SierraType, but was instead passed {:?}", array_type)
        };

        let func_name = format!("array_set_unchecked<{}>", element_type);

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }
        storage.helperfuncs.insert(func_name.clone());

        let block = self.new_block(&[array_type.get_type(), len_type, element_type]);

        let array_value = block.argument(0)?.into();
        let index_value = block.argument(1)?.into();
        let element_value = block.argument(2)?.into();

        let array_data_op =
            self.op_llvm_extractvalue(&block, 2, array_value, self.llvm_ptr_type())?;
        let array_data = array_data_op.result(0)?.into();
        let array_element_ptr_op =
            self.op_llvm_gep_dynamic(&block, &[index_value], array_data, element_type)?;
        let array_element_ptr = array_element_ptr_op.result(0)?.into();
        self.op_llvm_store(&block, element_value, array_element_ptr)?;

        self.op_return(&block, &[]);

        self.create_function(&func_name, vec![block], &[], FnAttributes::libfunc(false, true))?;

        Ok(func_name)
    }

    fn create_array_get_data_ptr(
        &'ctx self,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let element_type = if let SierraType::Array { element_type, .. } = array_type {
            element_type
        } else {
            panic!("create_array_get_data_ptr should have been passed an Array SierraType, but was instead passed {:?}", array_type)
        };

        let func_name = format!("array_get_data_ptr<{}>", element_type.get_type());

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }

        let block = self.new_block(&[array_type.get_type()]);
        let array_value = block.argument(0)?.into();

        let data_ptr_op =
            self.op_llvm_extractvalue(&block, 2, array_value, self.llvm_ptr_type())?;
        let data_ptr = data_ptr_op.result(0)?.into();

        self.op_return(&block, &[data_ptr]);

        storage.helperfuncs.insert(func_name.clone());

        self.create_function(
            &func_name,
            vec![block],
            &[self.llvm_ptr_type()],
            FnAttributes::libfunc(false, true),
        )?;

        Ok(func_name)
    }

    fn create_array_set_data_ptr(
        &'ctx self,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let element_type = if let SierraType::Array { element_type, .. } = array_type {
            element_type
        } else {
            panic!("create_array_set_data_ptr should have been passed an Array SierraType, but was instead passed {:?}", array_type)
        };

        let func_name = format!("array_set_data_ptr<{}>", element_type.get_type());

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }

        let block = self.new_block(&[array_type.get_type(), self.llvm_ptr_type()]);
        let array_value = block.argument(0)?.into();
        let new_ptr = block.argument(1)?.into();

        let updated_array_op =
            self.op_llvm_insertvalue(&block, 2, array_value, new_ptr, array_type.get_type())?;
        let updated_array = updated_array_op.result(0)?.into();

        self.op_return(&block, &[updated_array]);

        storage.helperfuncs.insert(func_name.clone());

        self.create_function(
            &func_name,
            vec![block],
            &[array_type.get_type()],
            FnAttributes::libfunc(false, true),
        )?;

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
        self.op_func_call(block, &func_name, &[array], &[len_type])
    }

    pub fn call_array_set_len_impl<'block>(
        &'ctx self,
        block: &'block Block,
        array: Value,
        new_len: Value,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        let func_name = self.create_array_set_len_impl(array_type, storage)?;
        self.op_func_call(block, &func_name, &[array, new_len], &[array_type.get_type()])
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
        self.op_func_call(block, &func_name, &[array], &[len_type])
    }

    pub fn call_array_set_capacity_impl<'block>(
        &'ctx self,
        block: &'block Block,
        array: Value,
        new_capacity: Value,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        let func_name = self.create_array_set_capacity_impl(array_type, storage)?;
        self.op_func_call(block, &func_name, &[array, new_capacity], &[array_type.get_type()])
    }

    pub fn call_array_get_unchecked<'block>(
        &'ctx self,
        block: &'block Block,
        array: Value,
        index: Value,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        let element_type = if let SierraType::Array { element_type, .. } = array_type {
            element_type.get_type()
        } else {
            panic!("call_array_get_unchecked should have been passed an array type, but instead was passed {:?}", array_type)
        };
        let func_name = self.create_array_get_unchecked(array_type, storage)?;
        self.op_func_call(block, &func_name, &[array, index], &[element_type])
    }

    pub fn call_array_set_unchecked<'block>(
        &'ctx self,
        block: &'block Block,
        array: Value,
        index: Value,
        value: Value,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        let func_name = self.create_array_set_unchecked(array_type, storage)?;
        self.op_func_call(block, &func_name, &[array, index, value], &[])
    }

    pub fn call_array_get_data_ptr<'block>(
        &'ctx self,
        block: &'block Block,
        array: Value,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        let func_name = self.create_array_get_data_ptr(array_type, storage)?;
        self.op_func_call(block, &func_name, &[array], &[self.llvm_ptr_type()])
    }

    pub fn call_array_set_data_ptr<'block>(
        &'ctx self,
        block: &'block Block,
        array: Value,
        new_ptr: Value,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        let func_name = self.create_array_set_data_ptr(array_type, storage)?;
        self.op_func_call(block, &func_name, &[array, new_ptr], &[array_type.get_type()])
    }
}
