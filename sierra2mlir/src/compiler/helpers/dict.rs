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

        let op =
            self.op_llvm_insertvalue(&block, 1, dict_value, new_capacity, dict_type.get_type())?;
        let updated_dict = op.result(0)?.into();

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

        // const 0
        let op = self.op_u32_const(&entry_block, "0");
        let const_0 = op.result(0)?.into();

        // get the current length
        let op = self.call_dict_len_impl(&entry_block, dict_value, dict_type, storage)?;
        let dict_len: Value = op.result(0)?.into();

        // get the capacity
        let op = self.call_dict_capacity_impl(&entry_block, dict_value, dict_type, storage)?;
        let dict_capacity = op.result(0)?.into();

        // check if we need to resize, done here since get in sierra always finds a slot, whether it existed
        // before or not
        self.call_dprintf(
            &entry_block,
            "has_enough cap: %d / %d\n",
            &[dict_len, dict_capacity],
            storage,
        )?;
        let op = self.op_cmp(&entry_block, CmpOp::UnsignedLessThan, dict_len, dict_capacity);
        let has_enough: Value = op.result(0)?.into();

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

        /*
           [resize block]
           - calculate new size
           - create new dict struct
           [rehash loop check block]
           - check current index is less than previous dict capacity
               [rehash loop body block]
               - get slot at index
               - check if slot is used
                   - calculate new hash based on slot key
                   - copy data to new dict
           [rehash loop end block]
           - free old dict
        */

        // useful constants
        let op = self.op_u32_const(&resize_block, "1");
        let const_1 = op.result(0)?.into();
        let op = self.op_u32_const(&resize_block, "2");
        let const_2 = op.result(0)?.into();

        // calculate the new size
        let op = self.op_mul(&resize_block, dict_capacity, const_2);
        let new_capacity: Value = op.result(0)?.into();

        let op = self.op_llvm_nullptr(&resize_block);
        let null_ptr = op.result(0)?.into();

        // calculate the new dict size
        let dict_slot_size_bytes = ((self.felt_type().get_width().unwrap()
            + element_type.get_width().unwrap()
            + self.bool_type().get_width().unwrap())
            + 7)
            / 8;

        let op = self.op_u32_const(&resize_block, &dict_slot_size_bytes.to_string());
        let slot_size_bytes = op.result(0)?.into();

        let op = self.op_mul(&resize_block, slot_size_bytes, new_capacity);
        let alloc_size = op.result(0)?.into();
        let op = self.op_zext(&resize_block, alloc_size, self.u64_type());
        let alloc_size = op.result(0)?.into();

        // allocate the new dict data
        let op = self.call_realloc(&resize_block, null_ptr, alloc_size, storage)?;
        let alloc_ptr = op.result(0)?.into();
        let op = self.call_memset(&resize_block, alloc_ptr, const_0, alloc_size, storage)?;
        let alloc_ptr = op.result(0)?.into();

        let old_dict_value = dict_value;
        let op = self.op_llvm_undef(&resize_block, dict_type.get_type());
        let new_dict_value = op.result(0)?.into();

        // setup the new dict struct
        let op = self.call_dict_set_capacity_impl(
            &resize_block,
            new_dict_value,
            new_capacity,
            dict_type,
            storage,
        )?;
        let new_dict_value = op.result(0)?.into();
        let op = self.call_dict_set_len_impl(
            &resize_block,
            new_dict_value,
            const_0,
            dict_type,
            storage,
        )?;
        let new_dict_value = op.result(0)?.into();
        let op = self.call_dict_set_data_ptr(
            &resize_block,
            new_dict_value,
            alloc_ptr,
            dict_type,
            storage,
        )?;
        let new_dict_value = op.result(0)?.into();

        // better names
        let old_dict_capacity = dict_capacity;

        // index, new_dict
        let loop_check_block = self.new_block(&[self.u32_type(), dict_type.get_type()]);
        let loop_body_block = self.new_block(&[self.u32_type(), dict_type.get_type()]);
        let loop_end_block = self.new_block(&[dict_type.get_type()]);

        self.op_br(&resize_block, &loop_check_block, &[const_0, new_dict_value]);

        // loop check block
        {
            let index = loop_check_block.argument(0)?.into();
            let new_dict_value = loop_check_block.argument(1)?.into();

            self.call_dprintf(
                &loop_check_block,
                "resizing, index: %d / %d -> (%d / %d)\n",
                &[index, old_dict_capacity, index, new_capacity],
                storage,
            )?;

            let op =
                self.op_cmp(&loop_check_block, CmpOp::UnsignedLessThan, index, old_dict_capacity);
            let is_less = op.result(0)?.into();

            self.op_cond_br(
                &loop_check_block,
                is_less,
                &loop_body_block,
                &loop_end_block,
                &[index, new_dict_value],
                &[new_dict_value],
            );
        }

        let loop_body_is_used_block = self.new_block(&[]);

        // body block
        {
            // get slot, check if used, rehash (using get_unchecked on new dict), copy data
            let index = loop_body_block.argument(0)?.into();
            let new_dict_value: Value = loop_body_block.argument(1)?.into();

            let op = self.op_add(&loop_body_block, index, const_1);
            let new_index = op.result(0)?.into();

            let op =
                self.call_dict_get_data_ptr(&loop_body_block, old_dict_value, dict_type, storage)?;
            let old_data_ptr = op.result(0)?.into();

            let op =
                self.op_llvm_gep_dynamic(&loop_body_block, &[index], old_data_ptr, entry_type)?;
            let old_entry_ptr = op.result(0)?.into();

            // first only load is_used and check if we need to copy it over
            // is used
            let op = self.op_llvm_gep(&loop_body_block, &[0, 2], old_entry_ptr, entry_type)?;
            let is_used_ptr: Value = op.result(0)?.into();
            let op = self.op_llvm_load(&loop_body_block, is_used_ptr, self.bool_type())?;
            let is_used: Value = op.result(0)?.into();

            self.op_cond_br(
                &loop_body_block,
                is_used,
                &loop_body_is_used_block,
                &loop_check_block,
                &[],
                &[new_index, new_dict_value],
            );

            // is used block, we can use previous block values

            // entry key
            let op =
                self.op_llvm_gep(&loop_body_is_used_block, &[0, 0], old_entry_ptr, entry_type)?;
            let entry_key_ptr = op.result(0)?.into();
            let op =
                self.op_llvm_load(&loop_body_is_used_block, entry_key_ptr, self.felt_type())?;
            let entry_key: Value = op.result(0)?.into();

            // entry value
            let op =
                self.op_llvm_gep(&loop_body_is_used_block, &[0, 1], old_entry_ptr, entry_type)?;
            let entry_value_ptr: Value = op.result(0)?.into();
            let op = self.op_llvm_load(&loop_body_is_used_block, entry_value_ptr, element_type)?;
            let entry_value: Value = op.result(0)?.into();

            // get the entry ptr on the new dict
            let op = self.call_dict_get_unchecked(
                &loop_body_is_used_block,
                new_dict_value,
                dict_type,
                entry_key,
                storage,
            )?;

            let new_dict_value = op.result(0)?.into();
            let new_entry_ptr = op.result(1)?.into();

            let op =
                self.op_llvm_gep(&loop_body_is_used_block, &[0, 0], new_entry_ptr, entry_type)?;
            let entry_key_ptr: Value = op.result(0)?.into();
            self.op_llvm_store(&loop_body_is_used_block, entry_key, entry_key_ptr)?;

            let op =
                self.op_llvm_gep(&loop_body_is_used_block, &[0, 1], new_entry_ptr, entry_type)?;
            let entry_value_ptr: Value = op.result(0)?.into();
            self.op_llvm_store(&loop_body_is_used_block, entry_value, entry_value_ptr)?;

            let op =
                self.op_llvm_gep(&loop_body_is_used_block, &[0, 2], new_entry_ptr, entry_type)?;
            let is_used_ptr: Value = op.result(0)?.into();
            self.op_llvm_store(&loop_body_is_used_block, is_used, is_used_ptr)?;

            self.op_br(&loop_body_is_used_block, &loop_check_block, &[new_index, new_dict_value]);
        }

        // loop end block
        {
            let new_dict_value = loop_end_block.argument(0)?.into();

            // free old dict ptr
            let op =
                self.call_dict_get_data_ptr(&loop_end_block, old_dict_value, dict_type, storage)?;
            let old_data_ptr = op.result(0)?.into();
            self.call_free(&loop_end_block, old_data_ptr, storage)?;

            self.op_br(&loop_end_block, &continue_block, &[new_dict_value]);
        }

        // hash result, current index
        let check_slot_block = self.new_block(&[self.u64_type()]);
        // entry type, bool = if key was found = if false, inc length
        let final_block = self.new_block(&[self.llvm_ptr_type(), self.bool_type()]);
        let final_block_inc_length = self.new_block(&[]);
        let final_block_return = self.new_block(&[dict_type.get_type()]);

        // continue block
        {
            let dict_value = continue_block.argument(0)?.into();

            // get the hash
            let op = self.op_llvm_alloca(&continue_block, self.felt_type(), 1)?;
            let dict_key_ptr = op.result(0)?.into();
            self.op_llvm_store(&continue_block, dict_key, dict_key_ptr)?;

            let op =
                self.call_dict_capacity_impl(&continue_block, dict_value, dict_type, storage)?;
            let dict_capacity = op.result(0)?.into();
            let op = self.op_zext(&continue_block, dict_capacity, self.u64_type());
            let dict_capacity = op.result(0)?.into();

            let op =
                self.op_llvm_extractvalue(&continue_block, 2, dict_value, self.llvm_ptr_type())?;
            let dict_data = op.result(0)?.into();

            let op = self.call_hash_i256(&continue_block, dict_key_ptr, storage)?;
            // u64
            let hash: Value = op.result(0)?.into();
            self.call_dprintf(&continue_block, "hash: %llu\n", &[hash], storage)?;

            let op = self.op_u64_const(&continue_block, "0");
            let const_0 = op.result(0)?.into();
            let op = self.op_u64_const(&continue_block, "1");
            let const_1: Value = op.result(0)?.into();

            self.op_br(&continue_block, &check_slot_block, &[const_0]);

            let index = check_slot_block.argument(0)?.into();
            let op = self.op_add(&check_slot_block, hash, index);
            let current_hash = op.result(0)?.into();

            // hash mod capacity
            let op = self.op_rem(&check_slot_block, current_hash, dict_capacity);
            let index_value: Value = op.result(0)?.into();
            self.call_dprintf(
                &check_slot_block,
                "hash index_value: %llu\n",
                &[index_value],
                storage,
            )?;

            let op =
                self.op_llvm_gep_dynamic(&check_slot_block, &[index_value], dict_data, entry_type)?;
            let entry_ptr = op.result(0)?.into();

            // entry key
            let op = self.op_llvm_gep(&check_slot_block, &[0, 0], entry_ptr, entry_type)?;
            let entry_key_ptr = op.result(0)?.into();
            let op = self.op_llvm_load(&check_slot_block, entry_key_ptr, self.felt_type())?;
            let entry_key: Value = op.result(0)?.into();

            // entry is_used
            let op = self.op_llvm_gep(&check_slot_block, &[0, 2], entry_ptr, entry_type)?;
            let is_used_ptr = op.result(0)?.into();
            let op = self.op_llvm_load(&check_slot_block, is_used_ptr, self.bool_type())?;
            let is_used: Value = op.result(0)?.into();

            let op = self.op_const(&check_slot_block, "0", self.bool_type());
            let const_false: Value = op.result(0)?.into();
            let op = self.op_cmp(&check_slot_block, CmpOp::Equal, is_used, const_false);
            let is_not_used = op.result(0)?.into();

            let op = self.op_cmp(&check_slot_block, CmpOp::Equal, dict_key, entry_key);
            let is_key_equal = op.result(0)?.into();

            let op = self.op_or(&check_slot_block, is_not_used, is_key_equal, self.bool_type());
            let is_found = op.result(0)?.into();

            let op = self.op_add(&check_slot_block, index, const_1);
            let next_index = op.result(0)?.into();

            self.op_cond_br(
                &check_slot_block,
                is_found,
                &final_block,
                &check_slot_block,
                &[entry_ptr, is_key_equal],
                &[next_index],
            );

            let dict_entry_ptr = final_block.argument(0)?.into();
            let was_key_found: Value = final_block.argument(1)?.into();

            self.op_cond_br(
                &final_block,
                was_key_found,
                &final_block_return,
                &final_block_inc_length,
                &[dict_value],
                &[],
            );

            // increase length in case key was not found (insert)

            let op =
                self.call_dict_len_impl(&final_block_inc_length, dict_value, dict_type, storage)?;
            let dict_len = op.result(0)?.into();
            let op = self.op_u32_const(&final_block_inc_length, "1");
            let const_1: Value = op.result(0)?.into();
            let op = self.op_add(&final_block_inc_length, dict_len, const_1);
            let new_dict_len = op.result(0)?.into();

            let op = self.call_dict_set_len_impl(
                &final_block_inc_length,
                dict_value,
                new_dict_len,
                dict_type,
                storage,
            )?;
            let dict_value = op.result(0)?.into();

            self.op_br(&final_block_inc_length, &final_block_return, &[dict_value]);

            let dict_value = final_block_return.argument(0)?.into();
            self.op_return(&final_block_return, &[dict_value, dict_entry_ptr]);
        }

        self.create_function(
            &func_name,
            vec![
                entry_block,
                resize_block,
                loop_check_block,
                loop_body_block,
                loop_body_is_used_block,
                loop_end_block,
                check_slot_block,
                final_block,
                final_block_inc_length,
                final_block_return,
                continue_block,
            ],
            &[dict_type.get_type(), self.llvm_ptr_type()],
            FnAttributes {
                inline: false, // too big
                local: true,
                public: false,
                norecurse: false, // we recurse
                nounwind: true,
                emit_c_interface: false,
            },
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
        self.op_func_call(
            block,
            &func_name,
            &[dict, key],
            &[dict_type.get_type(), self.llvm_ptr_type()],
        )
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
