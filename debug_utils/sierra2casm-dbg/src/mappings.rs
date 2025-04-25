use crate::{decode_instruction, Memory, StepId, Trace, ValueId};
use cairo_lang_casm::{
    hints::{CoreHint, CoreHintBase, DeprecatedHint, ExternalHint, Hint, StarknetHint},
    instructions::InstructionBody,
    operand::{CellRef, DerefOrImmediate, Register, ResOperand},
};
use cairo_vm::vm::trace::trace_entry::RelocatedTraceEntry;
use std::{
    collections::{HashMap, HashSet},
    ops::Index,
};

#[derive(Debug)]
pub struct GraphMappings {
    step2value: HashMap<StepId, HashSet<ValueId>>,
    value2step: HashMap<ValueId, HashSet<StepId>>,
}

impl GraphMappings {
    pub fn new(memory: &Memory, trace: &Trace, hints: &HashMap<usize, Vec<Hint>>) -> Self {
        let mut step2value = HashMap::<StepId, HashSet<ValueId>>::new();
        let mut value2step = HashMap::<ValueId, HashSet<StepId>>::new();

        for (step, trace) in trace.iter().enumerate() {
            let mut add_mapping = |value| {
                step2value
                    .entry(StepId(step))
                    .or_default()
                    .insert(ValueId(value));
                value2step
                    .entry(ValueId(value))
                    .or_default()
                    .insert(StepId(step));
            };

            Self::iter_memory_references(memory, trace, &mut add_mapping);
            if let Some(hints) = hints.get(&step) {
                for hint in hints {
                    Self::iter_hint_references(memory, trace, hint, &mut add_mapping);
                }
            }
        }

        Self {
            step2value,
            value2step,
        }
    }

    pub fn step2value(&self) -> &HashMap<StepId, HashSet<ValueId>> {
        &self.step2value
    }

    pub fn value2step(&self) -> &HashMap<ValueId, HashSet<StepId>> {
        &self.value2step
    }

    fn iter_memory_references(
        memory: &Memory,
        trace: &RelocatedTraceEntry,
        mut callback: impl FnMut(usize),
    ) {
        let instr = decode_instruction(memory, trace.pc);

        let mut process_cell_ref = |x: CellRef| {
            let offset = match x.register {
                Register::AP => trace.ap.wrapping_add_signed(x.offset as isize),
                Register::FP => trace.fp.wrapping_add_signed(x.offset as isize),
            };
            callback(offset);
            offset
        };

        match instr.body {
            InstructionBody::AddAp(add_ap_instruction) => match add_ap_instruction.operand {
                ResOperand::Deref(cell_ref) => todo!(),
                ResOperand::DoubleDeref(cell_ref, _) => todo!(),
                ResOperand::Immediate(_) => {}
                ResOperand::BinOp(bin_op_operand) => todo!(),
            },
            InstructionBody::AssertEq(assert_eq_instruction) => {
                process_cell_ref(assert_eq_instruction.a);
                match assert_eq_instruction.b {
                    ResOperand::Deref(cell_ref) => {
                        process_cell_ref(cell_ref);
                    }
                    ResOperand::DoubleDeref(cell_ref, _) => {
                        let offset = process_cell_ref(cell_ref);
                        callback(memory[offset].unwrap().try_into().unwrap());
                    }
                    ResOperand::Immediate(_) => {}
                    ResOperand::BinOp(bin_op_operand) => {
                        process_cell_ref(bin_op_operand.a);
                        match bin_op_operand.b {
                            DerefOrImmediate::Deref(cell_ref) => {
                                process_cell_ref(cell_ref);
                            }
                            DerefOrImmediate::Immediate(_) => {}
                        }
                    }
                }
            }
            InstructionBody::Call(call_instruction) => match call_instruction.target {
                DerefOrImmediate::Deref(cell_ref) => todo!(),
                DerefOrImmediate::Immediate(_) => {}
            },
            InstructionBody::Jnz(jnz_instruction) => {
                process_cell_ref(jnz_instruction.condition);
                match jnz_instruction.jump_offset {
                    DerefOrImmediate::Deref(cell_ref) => todo!(),
                    DerefOrImmediate::Immediate(_) => {}
                }
            }
            InstructionBody::Jump(jump_instruction) => match jump_instruction.target {
                DerefOrImmediate::Deref(cell_ref) => {
                    process_cell_ref(cell_ref);
                }
                DerefOrImmediate::Immediate(_) => {}
            },
            InstructionBody::Ret(_) => {}
        }
    }

    fn iter_hint_references(
        memory: &Memory,
        trace: &RelocatedTraceEntry,
        hint: &Hint,
        mut callback: impl FnMut(usize),
    ) {
        let mut process_cell_ref = |x: CellRef| {
            let offset = match x.register {
                Register::AP => trace.ap.wrapping_add_signed(x.offset as isize),
                Register::FP => trace.fp.wrapping_add_signed(x.offset as isize),
            };
            callback(offset);
            offset
        };

        match hint {
            Hint::Core(core_hint_base) => match core_hint_base {
                CoreHintBase::Core(core_hint) => match core_hint {
                    CoreHint::AllocSegment { dst } => {
                        process_cell_ref(*dst);
                    }
                    CoreHint::TestLessThan { lhs, rhs, dst } => {
                        match lhs {
                            ResOperand::Deref(cell_ref) => {
                                process_cell_ref(*cell_ref);
                            }
                            ResOperand::DoubleDeref(cell_ref, _) => todo!(),
                            ResOperand::Immediate(_) => {}
                            ResOperand::BinOp(bin_op_operand) => {
                                process_cell_ref(bin_op_operand.a);
                                match bin_op_operand.b {
                                    DerefOrImmediate::Deref(cell_ref) => {
                                        process_cell_ref(cell_ref);
                                    }
                                    DerefOrImmediate::Immediate(_) => {}
                                }
                            }
                        }
                        match rhs {
                            ResOperand::Deref(cell_ref) => {
                                process_cell_ref(*cell_ref);
                            }
                            ResOperand::DoubleDeref(cell_ref, _) => todo!(),
                            ResOperand::Immediate(_) => {}
                            ResOperand::BinOp(bin_op_operand) => todo!(),
                        }
                        process_cell_ref(*dst);
                    }
                    CoreHint::TestLessThanOrEqual { lhs, rhs, dst } => {
                        match lhs {
                            ResOperand::Deref(cell_ref) => {
                                process_cell_ref(*cell_ref);
                            }
                            ResOperand::DoubleDeref(cell_ref, _) => todo!(),
                            ResOperand::Immediate(_) => {}
                            ResOperand::BinOp(bin_op_operand) => todo!(),
                        }
                        match rhs {
                            ResOperand::Deref(cell_ref) => {
                                process_cell_ref(*cell_ref);
                            }
                            ResOperand::DoubleDeref(cell_ref, _) => todo!(),
                            ResOperand::Immediate(_) => {}
                            ResOperand::BinOp(bin_op_operand) => todo!(),
                        }
                        process_cell_ref(*dst);
                    }
                    CoreHint::TestLessThanOrEqualAddress { lhs, rhs, dst } => todo!(),
                    CoreHint::WideMul128 {
                        lhs,
                        rhs,
                        high,
                        low,
                    } => {
                        match lhs {
                            ResOperand::Deref(cell_ref) => {
                                process_cell_ref(*cell_ref);
                            }
                            ResOperand::DoubleDeref(cell_ref, _) => todo!(),
                            ResOperand::Immediate(_) => {}
                            ResOperand::BinOp(bin_op_operand) => todo!(),
                        }
                        match rhs {
                            ResOperand::Deref(cell_ref) => {
                                process_cell_ref(*cell_ref);
                            }
                            ResOperand::DoubleDeref(cell_ref, _) => todo!(),
                            ResOperand::Immediate(_) => {}
                            ResOperand::BinOp(bin_op_operand) => todo!(),
                        }
                        process_cell_ref(*high);
                        process_cell_ref(*low);
                    }
                    CoreHint::DivMod {
                        lhs,
                        rhs,
                        quotient,
                        remainder,
                    } => {
                        match lhs {
                            ResOperand::Deref(cell_ref) => {
                                process_cell_ref(*cell_ref);
                            }
                            ResOperand::DoubleDeref(cell_ref, _) => todo!(),
                            ResOperand::Immediate(_) => {}
                            ResOperand::BinOp(bin_op_operand) => todo!(),
                        }
                        match rhs {
                            ResOperand::Deref(cell_ref) => {
                                process_cell_ref(*cell_ref);
                            }
                            ResOperand::DoubleDeref(cell_ref, _) => todo!(),
                            ResOperand::Immediate(_) => {}
                            ResOperand::BinOp(bin_op_operand) => todo!(),
                        }
                        process_cell_ref(*quotient);
                        process_cell_ref(*remainder);
                    }
                    CoreHint::Uint256DivMod {
                        dividend0,
                        dividend1,
                        divisor0,
                        divisor1,
                        quotient0,
                        quotient1,
                        remainder0,
                        remainder1,
                    } => {
                        match dividend0 {
                            ResOperand::Deref(cell_ref) => {
                                process_cell_ref(*cell_ref);
                            }
                            ResOperand::DoubleDeref(cell_ref, _) => todo!(),
                            ResOperand::Immediate(_) => {}
                            ResOperand::BinOp(bin_op_operand) => todo!(),
                        }
                        match dividend1 {
                            ResOperand::Deref(cell_ref) => {
                                process_cell_ref(*cell_ref);
                            }
                            ResOperand::DoubleDeref(cell_ref, _) => todo!(),
                            ResOperand::Immediate(_) => {}
                            ResOperand::BinOp(bin_op_operand) => todo!(),
                        }
                        match divisor0 {
                            ResOperand::Deref(cell_ref) => {
                                process_cell_ref(*cell_ref);
                            }
                            ResOperand::DoubleDeref(cell_ref, _) => todo!(),
                            ResOperand::Immediate(_) => {}
                            ResOperand::BinOp(bin_op_operand) => todo!(),
                        }
                        match divisor1 {
                            ResOperand::Deref(cell_ref) => {
                                process_cell_ref(*cell_ref);
                            }
                            ResOperand::DoubleDeref(cell_ref, _) => todo!(),
                            ResOperand::Immediate(_) => {}
                            ResOperand::BinOp(bin_op_operand) => todo!(),
                        }
                        process_cell_ref(*quotient0);
                        process_cell_ref(*quotient1);
                        process_cell_ref(*remainder0);
                        process_cell_ref(*remainder1);
                    }
                    CoreHint::Uint512DivModByUint256 {
                        dividend0,
                        dividend1,
                        dividend2,
                        dividend3,
                        divisor0,
                        divisor1,
                        quotient0,
                        quotient1,
                        quotient2,
                        quotient3,
                        remainder0,
                        remainder1,
                    } => todo!(),
                    CoreHint::SquareRoot { value, dst } => todo!(),
                    CoreHint::Uint256SquareRoot {
                        value_low,
                        value_high,
                        sqrt0,
                        sqrt1,
                        remainder_low,
                        remainder_high,
                        sqrt_mul_2_minus_remainder_ge_u128,
                    } => todo!(),
                    CoreHint::LinearSplit {
                        value,
                        scalar,
                        max_x,
                        x,
                        y,
                    } => {
                        match value {
                            ResOperand::Deref(cell_ref) => {
                                process_cell_ref(*cell_ref);
                            }
                            ResOperand::DoubleDeref(cell_ref, _) => todo!(),
                            ResOperand::Immediate(_) => {}
                            ResOperand::BinOp(bin_op_operand) => todo!(),
                        }
                        match scalar {
                            ResOperand::Deref(cell_ref) => todo!(),
                            ResOperand::DoubleDeref(cell_ref, _) => todo!(),
                            ResOperand::Immediate(_) => {}
                            ResOperand::BinOp(bin_op_operand) => todo!(),
                        }
                        match max_x {
                            ResOperand::Deref(cell_ref) => todo!(),
                            ResOperand::DoubleDeref(cell_ref, _) => todo!(),
                            ResOperand::Immediate(_) => {}
                            ResOperand::BinOp(bin_op_operand) => todo!(),
                        }
                        process_cell_ref(*x);
                        process_cell_ref(*y);
                    }
                    CoreHint::AllocFelt252Dict { segment_arena_ptr } => todo!(),
                    CoreHint::Felt252DictEntryInit { dict_ptr, key } => todo!(),
                    CoreHint::Felt252DictEntryUpdate { dict_ptr, value } => todo!(),
                    CoreHint::GetSegmentArenaIndex {
                        dict_end_ptr,
                        dict_index,
                    } => todo!(),
                    CoreHint::InitSquashData {
                        dict_accesses,
                        ptr_diff,
                        n_accesses,
                        big_keys,
                        first_key,
                    } => todo!(),
                    CoreHint::GetCurrentAccessIndex { range_check_ptr } => todo!(),
                    CoreHint::ShouldSkipSquashLoop { should_skip_loop } => todo!(),
                    CoreHint::GetCurrentAccessDelta { index_delta_minus1 } => todo!(),
                    CoreHint::ShouldContinueSquashLoop { should_continue } => todo!(),
                    CoreHint::GetNextDictKey { next_key } => todo!(),
                    CoreHint::AssertLeFindSmallArcs {
                        range_check_ptr,
                        a,
                        b,
                    } => todo!(),
                    CoreHint::AssertLeIsFirstArcExcluded {
                        skip_exclude_a_flag,
                    } => todo!(),
                    CoreHint::AssertLeIsSecondArcExcluded {
                        skip_exclude_b_minus_a,
                    } => todo!(),
                    CoreHint::RandomEcPoint { x, y } => todo!(),
                    CoreHint::FieldSqrt { val, sqrt } => todo!(),
                    CoreHint::DebugPrint { start, end } => todo!(),
                    CoreHint::AllocConstantSize { size, dst } => todo!(),
                    CoreHint::U256InvModN {
                        b0,
                        b1,
                        n0,
                        n1,
                        g0_or_no_inv,
                        g1_option,
                        s_or_r0,
                        s_or_r1,
                        t_or_k0,
                        t_or_k1,
                    } => todo!(),
                    CoreHint::EvalCircuit {
                        n_add_mods,
                        add_mod_builtin,
                        n_mul_mods,
                        mul_mod_builtin,
                    } => todo!(),
                },
                CoreHintBase::Deprecated(deprecated_hint) => match deprecated_hint {
                    DeprecatedHint::AssertCurrentAccessIndicesIsEmpty => todo!(),
                    DeprecatedHint::AssertAllAccessesUsed { n_used_accesses } => {
                        todo!()
                    }
                    DeprecatedHint::AssertAllKeysUsed => todo!(),
                    DeprecatedHint::AssertLeAssertThirdArcExcluded => todo!(),
                    DeprecatedHint::AssertLtAssertValidInput { a, b } => todo!(),
                    DeprecatedHint::Felt252DictRead {
                        dict_ptr,
                        key,
                        value_dst,
                    } => todo!(),
                    DeprecatedHint::Felt252DictWrite {
                        dict_ptr,
                        key,
                        value,
                    } => todo!(),
                },
            },
            Hint::Starknet(starknet_hint) => match starknet_hint {
                StarknetHint::SystemCall { system } => match system {
                    ResOperand::Deref(cell_ref) => {
                        process_cell_ref(*cell_ref);
                    }
                    ResOperand::DoubleDeref(cell_ref, _) => todo!(),
                    ResOperand::Immediate(_) => {}
                    ResOperand::BinOp(bin_op_operand) => {
                        process_cell_ref(bin_op_operand.a);
                        match bin_op_operand.b {
                            DerefOrImmediate::Deref(cell_ref) => todo!(),
                            DerefOrImmediate::Immediate(_) => {}
                        }
                    }
                },
                StarknetHint::Cheatcode {
                    selector,
                    input_start,
                    input_end,
                    output_start,
                    output_end,
                } => todo!(),
            },
            Hint::External(external_hint) => match external_hint {
                ExternalHint::AddRelocationRule { src, dst } => todo!(),
                ExternalHint::WriteRunParam { index, dst } => todo!(),
            },
        }
    }
}

impl Index<StepId> for GraphMappings {
    type Output = HashSet<ValueId>;

    fn index(&self, index: StepId) -> &Self::Output {
        self.step2value.get(&index).unwrap()
    }
}

impl Index<ValueId> for GraphMappings {
    type Output = HashSet<StepId>;

    fn index(&self, index: ValueId) -> &Self::Output {
        self.value2step.get(&index).unwrap()
    }
}
