use color_eyre::Result;
use melior_next::ir::{Block, OperationRef, TypeLike, Value};

use crate::compiler::fn_attributes::FnAttributes;
use crate::compiler::mlir_ops::CmpOp;
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

    fn create_dict_capacity_impl(
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
            panic!("create_dict_capacity_impl should have been passed an Dictionary SierraType, but was instead passed {:?}", dict_type)
        };

        let func_name = format!("dict_capacity_impl<{}>", element_type.get_type());

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }

        let block = self.new_block(&[dict_type.get_type()]);

        let dict_value = block.argument(0)?.into();

        let dict_len_op = self.op_llvm_extractvalue(&block, 1, dict_value, len_type)?;
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

    fn create_dict_set_capacity_impl(
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
            panic!("create_dict_set_capacity_impl should have been passed an Dictionary SierraType, but was instead passed {:?}", dict_type)
        };

        let func_name = format!("dict_set_capacity_impl<{}>", element_type.get_type());

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }

        let block = self.new_block(&[dict_type.get_type(), len_type]);
        let dict_value = block.argument(0)?.into();
        let new_capacity = block.argument(1)?.into();

        let updated_dict_op =
            self.op_llvm_insertvalue(&block, 1, dict_value, new_capacity, dict_type.get_type())?;
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
    /// returns dict, pointer to the entry
    fn create_dict_get_unchecked(
        &'ctx self,
        dict_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let element_type = if let SierraType::Dictionary { element_type, .. } = dict_type {
            element_type.get_type()
        } else {
            panic!("create_dict_get_unchecked should have been passed an Array SierraType, but was instead passed {:?}", dict_type)
        };

        /*
        To search for a given key x, the cells of T are examined, beginning with the cell at index h(x) (where h is the hash function)
        and continuing to the adjacent cells h(x) + 1, h(x) + 2, ..., until finding either an empty cell
        or a cell whose stored key is x. If a cell containing the key is found,
        the search returns the value from that cell.
        Otherwise, if an empty cell is found, the key cannot be in the table,
        because it would have been placed in that cell in preference to any later cell that has not yet been searched.
        In this case, the search returns as its result that the key is not present in the dictionary.
        https://courses.cs.washington.edu/courses/cse326/04su/lectures/hashEx.pdf
        https://en.wikipedia.org/wiki/Linear_probing
        */

        // (key (always felt), value (T), is_used (bool))
        let entry_type =
            self.llvm_struct_type(&[self.felt_type(), element_type, self.bool_type()], false);

        let func_name = format!("dict_get_unchecked<{}>", element_type);

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }
        storage.helperfuncs.insert(func_name.clone());

        let entry_block = self.new_block(&[dict_type.get_type(), self.felt_type()]);

        let dict_value = entry_block.argument(0)?.into();
        let dict_key = entry_block.argument(1)?.into();

        // get the current length
        let len_op = self.call_dict_len_impl(&entry_block, dict_value, dict_type, storage)?;
        let dict_len: Value = len_op.result(0)?.into();

        // get the capacity
        let dict_capacity_op =
            self.call_dict_capacity_impl(&entry_block, dict_value, dict_type, storage)?;
        let dict_capacity = dict_capacity_op.result(0)?.into();

        // check if we need to resize, done here since get in sierra always finds a slot, whether it existed
        // before or not
        let has_enough_cap_op =
            self.op_cmp(&entry_block, CmpOp::UnsignedLessThan, dict_len, dict_capacity);
        let has_enough: Value = has_enough_cap_op.result(0)?.into();

        let resize_block = self.new_block(&[]);
        let continue_block = self.new_block(&[dict_type.get_type()]);

        self.op_cond_br(
            &entry_block,
            has_enough,
            &continue_block,
            &resize_block,
            &[dict_value],
            &[],
        );

        // resize block

        // calculate the new size
        let const_2_op = self.op_u32_const(&resize_block, "2");
        let new_size_op = self.op_mul(&resize_block, dict_capacity, const_2_op.result(0)?.into());
        let new_capacity: Value = new_size_op.result(0)?.into();

        let null_ptr_op = self.op_llvm_nullptr(&resize_block);

        let dict_slot_size_bytes = ((self.felt_type().get_width().unwrap()
            + element_type.get_width().unwrap()
            + self.bool_type().get_width().unwrap())
            + 7)
            / 8;

        let slot_size_bytes_op =
            self.op_u32_const(&resize_block, &dict_slot_size_bytes.to_string());
        let slot_size_bytes = slot_size_bytes_op.result(0)?.into();

        let alloc_size_op = self.op_mul(&resize_block, slot_size_bytes, new_capacity);
        let alloc_size = alloc_size_op.result(0)?.into();
        let alloc_size_op = self.op_zext(&resize_block, alloc_size, self.u64_type());
        let alloc_size = alloc_size_op.result(0)?.into();

        let alloc_ptr_op =
            self.call_realloc(&resize_block, null_ptr_op.result(0)?.into(), alloc_size, storage)?;

        // TODO: rehash here

        // todo: free old ptr

        self.op_br(&resize_block, &continue_block, &[todo!()]);

        // continue block
        let dict_value = continue_block.argument(0)?.into();

        // get the hash
        let dict_key_ptr_op = self.op_llvm_alloca(&continue_block, self.felt_type(), 1)?;
        let dict_key_ptr = dict_key_ptr_op.result(0)?.into();
        self.op_llvm_store(&continue_block, dict_key, dict_key_ptr)?;

        let dict_capacity_zext_op = self.op_zext(&continue_block, dict_capacity, self.u64_type());
        let dict_capacity = dict_capacity_zext_op.result(0)?.into();

        let hash_op = self.call_hash_i256(&continue_block, dict_key_ptr, storage)?;
        // u64
        let hash: Value = hash_op.result(0)?.into();

        // hash mod capacity
        let index_op = self.op_rem(&continue_block, hash, dict_capacity);
        let index_value: Value = index_op.result(0)?.into();

        let dict_data_op =
            self.op_llvm_extractvalue(&continue_block, 2, dict_value, self.llvm_ptr_type())?;
        let dict_data = dict_data_op.result(0)?.into();

        let dict_entry_ptr_op =
            self.op_llvm_gep_dynamic(&entry_block, &[index_value], dict_data, entry_type)?;
        let dict_entry_ptr = dict_entry_ptr_op.result(0)?.into();

        // TODO: check if the key equals, do linear probing if not.
        // right now it returns the first without checking.

        // todo: return the dict too
        self.op_return(&continue_block, &[dict_value, dict_entry_ptr]);

        self.create_function(
            &func_name,
            vec![entry_block, resize_block, continue_block],
            &[self.llvm_ptr_type()],
            FnAttributes::libfunc(false, true),
        )?;

        Ok(func_name)
    }

    fn create_dict_resize(
        &'ctx self,
        dict_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let element_type = if let SierraType::Dictionary { element_type, .. } = dict_type {
            element_type.get_type()
        } else {
            panic!("create_dict_resize should have been passed an Array SierraType, but was instead passed {:?}", dict_type)
        };

        // (key (always felt), value (T), is_used (bool))
        let entry_type =
            self.llvm_struct_type(&[self.felt_type(), element_type, self.bool_type()], false);

        let func_name = format!("dict_get_resize<{}>", element_type);

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }
        storage.helperfuncs.insert(func_name.clone());

        let entry_block = self.new_block(&[dict_type.get_type()]);

        let dict_value = entry_block.argument(0)?.into();

        todo!();

        self.op_return(&entry_block, &[dict_value]);

        self.create_function(
            &func_name,
            vec![entry_block],
            &[self.llvm_ptr_type()],
            FnAttributes::libfunc(false, true),
        )?;

        Ok(func_name)
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

        let data_ptr_op = self.op_llvm_extractvalue(&block, 2, dict_value, self.llvm_ptr_type())?;
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
            self.op_llvm_insertvalue(&block, 2, dict_value, new_ptr, dict_type.get_type())?;
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

    /// returns the entry possibly null pointer
    fn create_dict_get_entry_ptr(
        &'ctx self,
        dict_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let element_type = if let SierraType::Dictionary { element_type, .. } = dict_type {
            element_type
        } else {
            panic!("create_dict_get_data_ptr should have been passed an Dictionary SierraType, but was instead passed {:?}", dict_type)
        };

        let func_name = format!("dict_get_entry_ptr<{}>", element_type.get_type());

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }

        let block = self.new_block(&[dict_type.get_type()]);
        let dict_value = block.argument(0)?.into();

        let data_ptr_op = self.op_llvm_extractvalue(&block, 3, dict_value, self.llvm_ptr_type())?;
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

    fn create_dict_set_entry_ptr(
        &'ctx self,
        dict_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let element_type = if let SierraType::Dictionary { element_type, .. } = dict_type {
            element_type
        } else {
            panic!("create_dict_set_entry_ptr should have been passed an Dictionary SierraType, but was instead passed {:?}", dict_type)
        };

        let func_name = format!("dict_set_entry_ptr<{}>", element_type.get_type());

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }

        let block = self.new_block(&[dict_type.get_type(), self.llvm_ptr_type()]);
        let dict_value = block.argument(0)?.into();
        let new_ptr = block.argument(1)?.into();

        let updated_array_op =
            self.op_llvm_insertvalue(&block, 3, dict_value, new_ptr, dict_type.get_type())?;
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

    pub fn call_dict_capacity_impl<'block>(
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
        let func_name = self.create_dict_capacity_impl(dict_type, storage)?;
        self.op_func_call(block, &func_name, &[dict], &[len_type])
    }

    pub fn call_dict_set_capacity_impl<'block>(
        &'ctx self,
        block: &'block Block,
        dict: Value,
        new_len: Value,
        dict_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        let func_name = self.create_dict_set_capacity_impl(dict_type, storage)?;
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

    /// returns a pointer to the entry with the key.
    pub fn call_dict_get_unchecked<'block>(
        &'ctx self,
        block: &'block Block,
        dict: Value,
        dict_type: &SierraType,
        key: Value, // felt
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        let func_name = self.create_dict_get_unchecked(dict_type, storage)?;
        self.op_func_call(block, &func_name, &[dict, key], &[self.llvm_ptr_type()])
    }

    /// returns a possibly null pointer to the currently selected entry in the dict.
    pub fn call_dict_get_entry_ptr<'block>(
        &'ctx self,
        block: &'block Block,
        dict: Value,
        dict_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        let func_name = self.create_dict_get_entry_ptr(dict_type, storage)?;
        self.op_func_call(block, &func_name, &[dict], &[self.llvm_ptr_type()])
    }

    /// sets the possibly null pointer to the currently selected entry in the dict.
    pub fn call_dict_set_entry_ptr<'block>(
        &'ctx self,
        block: &'block Block,
        dict: Value,
        new_ptr: Value,
        dict_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        let func_name = self.create_dict_set_entry_ptr(dict_type, storage)?;
        self.op_func_call(block, &func_name, &[dict, new_ptr], &[dict_type.get_type()])
    }
}
