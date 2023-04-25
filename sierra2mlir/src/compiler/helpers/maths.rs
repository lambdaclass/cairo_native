use crate::compiler::{fn_attributes::FnAttributes, mlir_ops::CmpOp, Compiler, Storage};
use color_eyre::Result;
use melior_next::ir::{Block, OperationRef, Value};

impl<'ctx> Compiler<'ctx> {
    fn create_felt_mul_impl(&'ctx self, storage: &mut Storage<'ctx>) -> Result<()> {
        if storage.helperfuncs.contains("felt_mul_impl") {
            return Ok(());
        }
        storage.helperfuncs.insert("felt_mul_impl".to_string());

        let block = self.new_block(&[self.felt_type(), self.felt_type()]);

        // Need to first widen arguments so we can accurately calculate the non-modular product before wrapping it back into range
        let wide_type = self.double_felt_type();
        let lhs = block.argument(0)?.into();
        let rhs = block.argument(1)?.into();
        let lhs_wide_op = self.op_zext(&block, lhs, wide_type);
        let rhs_wide_op = self.op_zext(&block, rhs, wide_type);
        let lhs_wide = lhs_wide_op.result(0)?.into();
        let rhs_wide = rhs_wide_op.result(0)?.into();

        // res_wide = lhs_wide * rhs_wide
        let res_wide_op = self.op_mul(&block, lhs_wide, rhs_wide);
        let res_wide = res_wide_op.result(0)?.into();

        //res = res_wide mod PRIME
        let in_range_op = self.op_felt_modulo(&block, res_wide)?;
        let in_range = in_range_op.result(0)?.into();
        let res_op = self.op_trunc(&block, in_range, self.felt_type());
        let res = res_op.result(0)?.into();

        self.op_return(&block, &[res]);

        self.create_function(
            "felt_mul_impl",
            vec![block],
            &[self.felt_type()],
            FnAttributes::libfunc(false, true),
        )
    }

    // The extended euclidean algorithm calculates the greatest common divisor of two integers,
    // as well as the bezout coefficients x and y such that for inputs a and b, ax+by=gcd(a,b)
    // We use this in felt division to find the modular inverse of a given number
    // If a is the number we're trying to find the inverse of, we can do
    // ax+y*PRIME=gcd(a,PRIME)=1 => ax = 1 (mod PRIME)
    // Hence for input a, we return x
    // The input MUST be non-zero
    // See https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
    fn create_egcd_felt_inverse(&'ctx self, storage: &mut Storage<'ctx>) -> Result<()> {
        if storage.helperfuncs.contains("egcd_felt_inverse") {
            return Ok(());
        }
        storage.helperfuncs.insert("egcd_felt_inverse".to_string());

        let start_block = self.new_block(&[self.felt_type()]);
        let loop_block = self.new_block(&[
            self.felt_type(),
            self.felt_type(),
            self.felt_type(),
            self.felt_type(),
        ]);
        let negative_check_block = self.new_block(&[]);
        let negative_block = self.new_block(&[]);
        let positive_block = self.new_block(&[]);
        // Egcd works by calculating a series of remainders, each the remainder of dividing the previous two
        // For the initial setup, r0 = PRIME, r1 = a
        // This order is chosen because if we reverse them, then the first iteration will just swap them
        let prev_remainder_op = self.prime_constant(&start_block);
        let prev_remainder = prev_remainder_op.result(0)?.into();
        let remainder = start_block.argument(0)?.into();
        // Similarly we'll calculate another series which starts 0,1,... and from which we will retrieve the modular inverse of a
        let prev_inverse_op = self.op_felt_const(&start_block, "0");
        let prev_inverse = prev_inverse_op.result(0)?.into();
        let inverse_op = self.op_felt_const(&start_block, "1");
        let inverse = inverse_op.result(0)?.into();
        self.op_br(&start_block, &loop_block, &[prev_remainder, remainder, prev_inverse, inverse]);

        //---Loop body---
        // Arguments are rem_(i-1), rem, inv_(i-1), inv
        let prev_remainder = loop_block.argument(0)?.into();
        let remainder = loop_block.argument(1)?.into();
        let prev_inverse = loop_block.argument(2)?.into();
        let inverse = loop_block.argument(3)?.into();

        // First calculate q = rem_(i-1)/rem_i, rounded down
        let quotient_op = self.op_div(&loop_block, prev_remainder, remainder);
        let quotient = quotient_op.result(0)?.into();

        // Then r_(i+1) = r_(i-1) - q * r_i, and inv_(i+1) = inv_(i-1) - q * inv_i
        let rem_times_quo_op = self.op_mul(&loop_block, remainder, quotient);
        let rem_times_quo = rem_times_quo_op.result(0)?.into();
        let inv_times_quo_op = self.op_mul(&loop_block, inverse, quotient);
        let inv_times_quo = inv_times_quo_op.result(0)?.into();
        let next_remainder_op = self.op_sub(&loop_block, prev_remainder, rem_times_quo);
        let next_remainder = next_remainder_op.result(0)?.into();
        let next_inverse_op = self.op_sub(&loop_block, prev_inverse, inv_times_quo);
        let next_inverse = next_inverse_op.result(0)?.into();

        // If r_(i+1) is 0, then inv_i is the inverse
        let zero_op = self.op_felt_const(&loop_block, "0");
        let zero = zero_op.result(0)?.into();
        let next_remainder_eq_zero_op =
            self.op_cmp(&loop_block, CmpOp::Equal, next_remainder, zero);
        let next_remainder_eq_zero = next_remainder_eq_zero_op.result(0)?.into();
        self.op_cond_br(
            &loop_block,
            next_remainder_eq_zero,
            &negative_check_block,
            &loop_block,
            &[],
            &[remainder, next_remainder, inverse, next_inverse],
        );

        // egcd sometimes returns a negative number for the inverse,
        // in such cases we must simply wrap it around back into [0, PRIME)
        // this suffices because |inv_i| <= divfloor(PRIME,2)
        let zero_op = self.op_felt_const(&negative_check_block, "0");
        let zero = zero_op.result(0)?.into();
        let is_negative_op =
            self.op_cmp(&negative_check_block, CmpOp::SignedLessThan, inverse, zero);
        let is_negative = is_negative_op.result(0)?.into();
        self.op_cond_br(
            &negative_check_block,
            is_negative,
            &negative_block,
            &positive_block,
            &[],
            &[],
        );

        // if the inverse is >= 0, just return it
        self.op_return(&positive_block, &[inverse]);

        // if the inverse is < 0, add PRIME then return it
        let prime_op = self.prime_constant(&negative_block);
        let prime = prime_op.result(0)?.into();
        let wrapped_op = self.op_add(&negative_block, inverse, prime);
        let wrapped = wrapped_op.result(0)?.into();
        self.op_return(&negative_block, &[wrapped]);

        self.create_function(
            "egcd_felt_inverse",
            vec![start_block, loop_block, negative_check_block, positive_block, negative_block],
            &[self.felt_type()],
            FnAttributes::libfunc(false, false),
        )
    }
}

impl<'ctx> Compiler<'ctx> {
    pub fn call_egcd_felt_inverse<'block>(
        &'ctx self,
        block: &'block Block,
        nonzero_input: Value,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        self.create_egcd_felt_inverse(storage)?;
        self.op_func_call(block, "egcd_felt_inverse", &[nonzero_input], &[self.felt_type()])
    }

    pub fn call_felt_mul_impl<'block>(
        &'ctx self,
        block: &'block Block,
        lhs: Value,
        rhs: Value,
        storage: &mut Storage<'ctx>,
    ) -> Result<OperationRef<'block>> {
        self.create_felt_mul_impl(storage)?;
        self.op_func_call(block, "felt_mul_impl", &[lhs, rhs], &[self.felt_type()])
    }
}
