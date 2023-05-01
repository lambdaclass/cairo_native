use color_eyre::Result;
use melior_next::ir::{Block, OperationRef, Value, ValueLike};

use crate::compiler::fn_attributes::FnAttributes;
use crate::compiler::mlir_ops::CmpOp;
use crate::{
    compiler::{Compiler, Storage},
    sierra_type::SierraType,
};

// macro_rules! array_helper {
//     ($name:pat, $distinction:expr, $implementation:block, +$arg) => {
//         pub fn create_$name(
//             &'ctx self,
//             array_type: &SierraType,
//             storage: &mut Storage<'ctx>,
//         ) -> Result<String> {
//             let (ty, len_type, element_type) = if let SierraType::Array { ty, len_type, element_type } = array_type {
//                 (*ty, *len_type, element_type)
//             } else {
//                 panic!("create_$name should have been passed an Array SierraType, but was instead passed {:?}", array_type)
//             }

//             let func_name = format!("$name<{}>", $distinction);

//             if storage.helperfuncs.contains(&func_name) {
//                 return Ok(func_name);
//             }

//             $implementation

//             storage.helperfuncs.insert(func_name.clone());

//             Ok(func_name)
//         }

//         pub fn call_$name<'block>(
//             &'ctx self,
//             block: &'block Block,
//             array: Value,
//             array_type: &SierraType,
//             storage: &mut Storage<'ctx>,
//         )
//     };
// }

impl<'ctx> Compiler<'ctx> {
    fn create_array_len_impl(
        &'ctx self,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let len_type = if let SierraType::Array { len_type, .. } = array_type {
            *len_type
        } else {
            panic!("create_array_len_impl should have been passed an Array SierraType, but was instead passed {:?}", array_type)
        };

        // Since arrays use opaque pointers, we only need one version of this function per length type
        let func_name = format!("array_len_impl<{}>", len_type);

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
        let len_type = if let SierraType::Array { len_type, .. } = array_type {
            *len_type
        } else {
            panic!("create_array_set_len_impl should have been passed an Array SierraType, but was instead passed {:?}", array_type)
        };

        let func_name = format!("array_set_len_impl<{}>", len_type);

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
        let len_type = if let SierraType::Array { len_type, .. } = array_type {
            *len_type
        } else {
            panic!("create_array_capacity_impl should have been passed an Array SierraType, but was instead passed {:?}", array_type)
        };

        let func_name = format!("array_capacity_impl<{}>", len_type);

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
        let len_type = if let SierraType::Array { len_type, .. } = array_type {
            *len_type
        } else {
            panic!("create_array_set_capacity_impl should have been passed an Array SierraType, but was instead passed {:?}", array_type)
        };

        let func_name = format!("array_set_capacity_impl<{}>", len_type);

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

    pub fn create_array_get_unchecked(
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

    pub fn create_array_is_empty(
        &'ctx self,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let len_type = if let SierraType::Array { len_type, .. } = array_type {
            *len_type
        } else {
            panic!("create_array_is_empty should have been passed an Array SierraType, but was instead passed {:?}", array_type)
        };

        // Since arrays use opaque pointers, we only need one version of this function per length type
        let func_name = format!("array_is_empty<{}>", len_type);

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }

        let block = self.new_block(&[array_type.get_type()]);
        let array = block.argument(0)?.into();
        let len_op = self.call_array_len_impl(&block, array, &array_type, storage)?;
        let len: Value = len_op.result(0)?.into();
        let zero_op = self.op_const(&block, "0", len.r#type());
        let zero = zero_op.result(0)?.into();
        let len_is_zero_op = self.op_cmp(&block, CmpOp::Equal, len, zero);
        let len_is_zero = len_is_zero_op.result(0)?.into();
        self.op_return(&block, &[len_is_zero]);

        storage.helperfuncs.insert(func_name.clone());

        self.create_function(
            &func_name,
            vec![block],
            &[len_type],
            FnAttributes::libfunc(false, true),
        )?;

        Ok(func_name)
    }

    pub fn create_array_pop_unchecked(
        &'ctx self,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let (len_type, element_type) = if let SierraType::Array { len_type, element_type, .. } =
            array_type
        {
            (*len_type, element_type)
        } else {
            panic!("create_array_pop_unchecked should have been passed an Array SierraType, but was instead passed {:?}", array_type)
        };

        // Since arrays use opaque pointers, we only need one version of this function per length type
        let func_name = format!("array_is_empty<{}>", len_type);

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }

        let block = self.new_block(&[array_type.get_type()]);
        let array = block.argument(0)?.into();

        // get the element to return
        let zero_op = self.op_const(&block, "0", len_type);
        let zero = zero_op.result(0)?.into();
        let first_element_op =
            self.call_array_get_unchecked(&block, array, zero, array_type, storage)?;
        let first_element = first_element_op.result(0)?.into();

        // decrement the length
        let len_op = self.call_array_len_impl(&block, array, array_type, storage)?;
        let len = len_op.result(0)?.into();
        let one_op = self.op_const(&block, "1", len_type);
        let one = one_op.result(0)?.into();
        let new_len_op = self.op_sub(&block, len, one);
        let new_len = new_len_op.result(0)?.into();
        self.call_array_set_len_impl(&block, array, new_len, array_type, storage)?;

        // get the new length in bytes
        let new_length_zext_op = self.op_zext(&block, new_len, self.u64_type());
        let new_length_zext = new_length_zext_op.result(0)?.into();

        let element_size_bytes = (element_type.get_width() + 7) / 8;
        let const_element_size_bytes_op =
            self.op_u64_const(&block, &element_size_bytes.to_string());
        let const_element_size_bytes = const_element_size_bytes_op.result(0)?.into();

        let new_length_bytes_op = self.op_mul(&block, new_length_zext, const_element_size_bytes);
        let new_length_bytes = new_length_bytes_op.result(0)?.into();

        // move the second element onwards to the start of the allocated area
        let base_ptr_op = self.call_array_get_data_ptr(&block, array, array_type, storage)?;
        let base_ptr = base_ptr_op.result(0)?.into();
        let second_element_ptr_op =
            self.op_llvm_gep_dynamic(&block, &[one], base_ptr, element_type.get_type())?;
        let second_element_ptr = second_element_ptr_op.result(0)?.into();
        self.call_memmove(&block, base_ptr, second_element_ptr, new_length_bytes, storage)?;

        self.op_return(&block, &[first_element]);

        storage.helperfuncs.insert(func_name.clone());

        self.create_function(
            &func_name,
            vec![block],
            &[element_type.get_type()],
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

    pub fn call_array_is_empty<'block>(
        &'ctx self,
        block: &'block Block,
        array: Value,
        array_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        let func_name = self.create_array_is_empty(array_type, storage)?;
        self.op_func_call(block, &func_name, &[array], &[self.bool_type()])
    }
}
