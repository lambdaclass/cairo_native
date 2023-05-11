use color_eyre::Result;
use melior_next::ir::{Block, OperationRef, Value};

use crate::compiler::fn_attributes::FnAttributes;
use crate::{
    compiler::{Compiler, Storage},
    sierra_type::SierraType,
};

impl<'ctx> Compiler<'ctx> {
    fn create_dict_len_impl(
        &'ctx self,
        dict_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let (len_type, element_type) = if let SierraType::Dictionary {
            ty: _,
            len_type,
            element_type,
        } = dict_type
        {
            (*len_type, element_type)
        } else {
            panic!("create_dict_len_impl should have been passed an Dictionary SierraType, but was instead passed {:?}", dict_type)
        };

        let func_name = format!("dict_len_impl<{}>", element_type.get_type());

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }

        let block = self.new_block(&[dict_type.get_type()]);

        let dict_value = block.argument(0)?.into();

        let dict_len_op = self.op_llvm_extractvalue(&block, 0, dict_value, len_type)?;
        let dict_len = dict_len_op.result(0)?.into();

        self.op_return(&block, &[dict_len]);

        storage.helperfuncs.insert(func_name.clone());

        self.create_function(
            &func_name,
            vec![block],
            &[len_type],
            FnAttributes::libfunc(false, true),
        )?;

        Ok(func_name)
    }

    fn create_dict_set_len_impl(
        &'ctx self,
        dict_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let (len_type, element_type) = if let SierraType::Dictionary {
            len_type,
            element_type,
            ..
        } = dict_type
        {
            (*len_type, element_type)
        } else {
            panic!("create_dict_set_len_impl should have been passed an Dictionary SierraType, but was instead passed {:?}", dict_type)
        };

        let func_name = format!("dict_set_len_impl<{}>", element_type.get_type());

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }

        let block = self.new_block(&[dict_type.get_type(), len_type]);
        let dict_value = block.argument(0)?.into();
        let new_len = block.argument(1)?.into();

        let updated_dict_op =
            self.op_llvm_insertvalue(&block, 0, dict_value, new_len, dict_type.get_type())?;
        let updated_dict = updated_dict_op.result(0)?.into();

        self.op_return(&block, &[updated_dict]);

        storage.helperfuncs.insert(func_name.clone());

        self.create_function(
            &func_name,
            vec![block],
            &[dict_type.get_type()],
            FnAttributes::libfunc(false, true),
        )?;

        Ok(func_name)
    }

    /// key is always a felt
    #[allow(unused)]
    fn create_dict_get_unchecked(
        &'ctx self,
        dict_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        todo!();
    }

    #[allow(unused)]
    fn create_dict_set_unchecked(
        &'ctx self,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        todo!();
    }

    fn create_dict_get_data_ptr(
        &'ctx self,
        dict_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let element_type = if let SierraType::Dictionary { element_type, .. } = dict_type {
            element_type
        } else {
            panic!("create_dict_get_data_ptr should have been passed an Dictionary SierraType, but was instead passed {:?}", dict_type)
        };

        let func_name = format!("dict_get_data_ptr<{}>", element_type.get_type());

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }

        let block = self.new_block(&[dict_type.get_type()]);
        let dict_value = block.argument(0)?.into();

        let data_ptr_op = self.op_llvm_extractvalue(&block, 1, dict_value, self.llvm_ptr_type())?;
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

    fn create_dict_set_data_ptr(
        &'ctx self,
        dict_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let element_type = if let SierraType::Dictionary { element_type, .. } = dict_type {
            element_type
        } else {
            panic!("create_dict_set_data_ptr should have been passed an Dictionary SierraType, but was instead passed {:?}", dict_type)
        };

        let func_name = format!("dict_set_data_ptr<{}>", element_type.get_type());

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }

        let block = self.new_block(&[dict_type.get_type(), self.llvm_ptr_type()]);
        let dict_value = block.argument(0)?.into();
        let new_ptr = block.argument(1)?.into();

        let updated_array_op =
            self.op_llvm_insertvalue(&block, 1, dict_value, new_ptr, dict_type.get_type())?;
        let updated_dict = updated_array_op.result(0)?.into();

        self.op_return(&block, &[updated_dict]);

        storage.helperfuncs.insert(func_name.clone());

        self.create_function(
            &func_name,
            vec![block],
            &[dict_type.get_type()],
            FnAttributes::libfunc(false, true),
        )?;

        Ok(func_name)
    }
}

impl<'ctx> Compiler<'ctx> {
    pub fn call_dict_len_impl<'block>(
        &'ctx self,
        block: &'block Block,
        dict: Value,
        dict_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        let len_type = if let SierraType::Dictionary { ty: _, len_type, element_type: _ } =
            dict_type
        {
            *len_type
        } else {
            panic!("call_dict_len_impl should have been passed an array type, but instead was passed {:?}", dict_type)
        };
        let func_name = self.create_dict_len_impl(dict_type, storage)?;
        self.op_func_call(block, &func_name, &[dict], &[len_type])
    }

    pub fn call_dict_set_len_impl<'block>(
        &'ctx self,
        block: &'block Block,
        dict: Value,
        new_len: Value,
        dict_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        let func_name = self.create_dict_set_len_impl(dict_type, storage)?;
        self.op_func_call(block, &func_name, &[dict, new_len], &[dict_type.get_type()])
    }

    pub fn call_dict_get_data_ptr<'block>(
        &'ctx self,
        block: &'block Block,
        dict: Value,
        dict_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        let func_name = self.create_dict_get_data_ptr(dict_type, storage)?;
        self.op_func_call(block, &func_name, &[dict], &[self.llvm_ptr_type()])
    }

    pub fn call_dict_set_data_ptr<'block>(
        &'ctx self,
        block: &'block Block,
        dict: Value,
        new_ptr: Value,
        dict_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        let func_name = self.create_dict_set_data_ptr(dict_type, storage)?;
        self.op_func_call(block, &func_name, &[dict, new_ptr], &[dict_type.get_type()])
    }
}
