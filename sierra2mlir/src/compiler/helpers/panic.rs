use color_eyre::Result;
use melior_next::ir::{Block, OperationRef, Region, Value};

use crate::compiler::fn_attributes::FnAttributes;
use crate::compiler::mlir_ops::CmpOp;
use crate::{
    compiler::{Compiler, Storage},
    sierra_type::SierraType,
    utility::create_fn_signature,
};

impl<'ctx> Compiler<'ctx> {
    fn create_print_panic_message(
        &'ctx self,
        panic_type: &SierraType,
        storage: &mut Storage<'ctx>,
    ) -> Result<String> {
        let func_name = format!("print_panic_message<{}>", panic_type.get_type());

        if storage.helperfuncs.contains(&func_name) {
            return Ok(func_name);
        }
        storage.helperfuncs.insert(func_name.clone());

        let err_message_type = match &panic_type {
            SierraType::Enum {
                ty: _,
                tag_type: _,
                storage_bytes_len: _,
                storage_type: _,
                variants_types,
            } => &variants_types[1],
            _ => panic!("create_print_panic_message passed non enum type {:?}", panic_type),
        };

        let (panic_array_len_type, panic_array_element_type) = match &err_message_type {
            SierraType::Array { ty: _, len_type, element_type } => {
                (*len_type, element_type.get_type())
            }
            _ => panic!("panic variant type should be an array. Found {:?}", err_message_type),
        };

        let region = Region::new();

        // Block structure we'll create
        // block                        block is the entry point of the function, where "Program panicked" is printed
        // v
        // outer_loop_block-----------  the outer loop iterates through the panic's array
        // v inner_loop_start_block  ^  the inner loop iterates through the bytes of each element
        // v v                       ^
        // v print_if_nonzero_block  ^  print_if_nonzero_block extracts the 8bit segment, and jumps to print_block if it's nonzero
        // v v v            ^        ^
        // v v print_block  ^        ^  print_block calls dprintf
        // v v v            ^        ^
        // v loop_end_block-^--------^  loop_end_block returns to print_if_nonzero_block if there is more of this element left to print,
        // v                            otherwise it returns to outer_loop_block to move onto the next array element or finish
        // done_block

        let block = region.append_block(self.new_block(&[panic_type.get_type()]));
        let panic_value = block.argument(0)?.into();

        // In the panic block, extract the error message. This should be an Array<felt252>
        // In order to do this, the enum's data needs to be stored on the stack
        self.call_dprintf(&block, "Program panicked\n", &[], storage)?;
        let err_message_data_op =
            self.call_enum_get_data_as_variant_type(&block, panic_value, panic_type, 1, storage)?;
        // An Array<felt252>, containing one character per byte
        let err_message_array = err_message_data_op.result(0)?.into();
        let len_op =
            self.call_array_len_impl(&block, err_message_array, err_message_type, storage)?;
        let len = len_op.result(0)?.into();
        let zero_op = self.op_const(&block, "0", panic_array_len_type);
        let zero = zero_op.result(0)?.into();
        let one_op = self.op_const(&block, "1", panic_array_len_type);
        let one = one_op.result(0)?.into();

        // Create a loop to loop through the elements of the array to be printed
        let outer_loop_block = region.append_block(self.new_block(&[panic_array_len_type]));
        let done_block = region.append_block(Block::new(&[]));
        self.op_br(&block, &outer_loop_block, &[zero]);
        // At the start of the outer block, check if the loop index is equal to len
        // If it is, jump to the done block, otherwise the inner block
        let loop_index = outer_loop_block.argument(0)?.into();
        let index_eq_len_op = self.op_cmp(&outer_loop_block, CmpOp::Equal, loop_index, len);
        let index_eq_len = index_eq_len_op.result(0)?.into();
        let inner_loop_start_block = region.append_block(Block::new(&[]));
        self.op_cond_br(
            &outer_loop_block,
            index_eq_len,
            &done_block,
            &inner_loop_start_block,
            &[],
            &[],
        );

        // In the inner loop start block, get the element at the given index, and pass it to the print if nonzero block
        let err_message_element_op = self.call_array_get_unchecked(
            &inner_loop_start_block,
            err_message_array,
            loop_index,
            err_message_type,
            storage,
        )?;
        let err_message_element = err_message_element_op.result(0)?.into();
        let increment_op = self.op_add(&inner_loop_start_block, loop_index, one);
        let incremented_loop_index = increment_op.result(0)?.into();

        // In the print if nonzero block, we're going to print the topmost 8bits of the value, then shift it left by 8 bits, and repeat until it is 0
        let print_if_nonzero_block =
            region.append_block(self.new_block(&[panic_array_element_type]));
        self.op_br(&inner_loop_start_block, &print_if_nonzero_block, &[err_message_element]);
        let print_arg = print_if_nonzero_block.argument(0)?.into();
        // check the current value against zero
        // the loop only breaks once we've shifted all non-zero bits out of the value
        let zero_op = self.op_felt_const(&print_if_nonzero_block, "0");
        let zero = zero_op.result(0)?.into();
        let is_zero_op = self.op_cmp(&print_if_nonzero_block, CmpOp::Equal, print_arg, zero);
        let is_zero = is_zero_op.result(0)?.into();
        // shifting right by 248 extracts the current topmost 8 bits
        let shift_amount_op = self.op_felt_const(&print_if_nonzero_block, "248");
        let shift_op =
            self.op_shru(&print_if_nonzero_block, print_arg, shift_amount_op.result(0)?.into());
        let shift = shift_op.result(0)?.into();
        let bits_to_print_op = self.op_trunc(&print_if_nonzero_block, shift, self.u8_type());
        let bits_to_print = bits_to_print_op.result(0)?.into();
        // shifting left by 8 sets up the next loop iteration
        let eight_op = self.op_felt_const(&print_if_nonzero_block, "8");
        let eight = eight_op.result(0)?.into();
        let next_print_arg_op = self.op_shl(&print_if_nonzero_block, print_arg, eight);
        let next_print_arg = next_print_arg_op.result(0)?.into();

        let print_block = region.append_block(Block::new(&[]));
        let loop_end_block = region.append_block(Block::new(&[]));
        // check specifically the current 8 bits against zero
        let bits_to_print_zero_cmp_op =
            self.op_cmp(&print_if_nonzero_block, CmpOp::Equal, shift, zero);
        let bits_to_print_zero_cmp = bits_to_print_zero_cmp_op.result(0)?.into();
        self.op_cond_br(
            &print_if_nonzero_block,
            bits_to_print_zero_cmp,
            &loop_end_block,
            &print_block,
            &[],
            &[],
        );
        self.call_dprintf(&print_block, "%c", &[bits_to_print], storage)?;
        self.op_br(&print_block, &loop_end_block, &[]);
        self.op_cond_br(
            &loop_end_block,
            is_zero,
            &outer_loop_block,
            &print_if_nonzero_block,
            &[incremented_loop_index],
            &[next_print_arg],
        );

        self.op_return(&done_block, &[]);

        let function_type = create_fn_signature(&[panic_type.get_type()], &[]);

        let func = self.op_func(
            &func_name,
            &function_type,
            vec![region],
            FnAttributes::libfunc(false, false),
        )?;

        self.module.body().append_operation(func);

        Ok(func_name)
    }
}

impl<'ctx> Compiler<'ctx> {
    pub fn call_print_panic_message<'block>(
        &'ctx self,
        block: &'block Block,
        panic_value: Value,
        panic_type: &SierraType<'ctx>,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        let func_name = self.create_print_panic_message(panic_type, storage)?;
        self.op_llvm_call(block, &func_name, &[panic_value], &[])
    }
}
