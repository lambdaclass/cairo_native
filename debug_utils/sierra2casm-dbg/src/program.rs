use crate::Memory;
use cairo_lang_casm::{
    instructions::{
        AddApInstruction, AssertEqInstruction, CallInstruction, Instruction, InstructionBody,
        JnzInstruction, JumpInstruction, RetInstruction,
    },
    operand::{BinOpOperand, CellRef, DerefOrImmediate, Operation, ResOperand},
};
use cairo_lang_utils::bigint::BigIntAsHex;
use cairo_vm::{
    types::instruction::{
        ApUpdate, FpUpdate, Instruction as InstructionRepr, Op1Addr, Opcode, PcUpdate, Register,
        Res,
    },
    vm::decoding::decoder,
};
use starknet_types_core::felt::Felt;

/// Source: https://github.com/starkware-libs/cairo/blob/main/crates/cairo-lang-casm/src/assembler.rs
pub fn decode_instruction(memory: &Memory, offset: usize) -> Instruction {
    let instr_repr =
        decoder::decode_instruction(memory[offset].unwrap().try_into().unwrap()).unwrap();

    match instr_repr {
        InstructionRepr {
            off0: -1,
            off1,
            off2,
            dst_register: Register::FP,
            op0_register,
            op1_addr,
            res,
            pc_update: PcUpdate::Regular,
            ap_update: ApUpdate::Add,
            fp_update: FpUpdate::Regular,
            opcode: Opcode::NOp,
        } => Instruction {
            body: InstructionBody::AddAp(AddApInstruction {
                operand: decode_res_operand(ResDescription {
                    off1,
                    off2,
                    imm: memory.get(offset + 1).copied().flatten(),
                    op0_register,
                    op1_addr,
                    res,
                }),
            }),
            inc_ap: false,
            hints: Vec::new(),
        },
        InstructionRepr {
            off0,
            off1,
            off2,
            dst_register,
            op0_register,
            op1_addr,
            res,
            pc_update: PcUpdate::Regular,
            ap_update: ap_update @ (ApUpdate::Add1 | ApUpdate::Regular),
            fp_update: FpUpdate::Regular,
            opcode: Opcode::AssertEq,
        } => Instruction {
            body: InstructionBody::AssertEq(AssertEqInstruction {
                a: CellRef {
                    register: match dst_register {
                        Register::AP => cairo_lang_casm::operand::Register::AP,
                        Register::FP => cairo_lang_casm::operand::Register::FP,
                    },
                    offset: off0 as i16,
                },
                b: decode_res_operand(ResDescription {
                    off1,
                    off2,
                    imm: memory.get(offset + 1).copied().flatten(),
                    op0_register,
                    op1_addr,
                    res,
                }),
            }),
            inc_ap: match ap_update {
                ApUpdate::Regular => false,
                ApUpdate::Add1 => true,
                _ => unreachable!(),
            },
            hints: Vec::new(),
        },
        InstructionRepr {
            off0: 0,
            off1: 1,
            off2,
            dst_register: Register::AP,
            op0_register: Register::AP,
            op1_addr: op1_addr @ (Op1Addr::AP | Op1Addr::FP | Op1Addr::Imm),
            res: Res::Op1,
            pc_update: pc_update @ (PcUpdate::JumpRel | PcUpdate::Jump),
            ap_update: ApUpdate::Add2,
            fp_update: FpUpdate::APPlus2,
            opcode: Opcode::Call,
        } => Instruction {
            body: InstructionBody::Call(CallInstruction {
                target: match op1_addr {
                    Op1Addr::Imm => {
                        assert_eq!(off2, 1);
                        DerefOrImmediate::Immediate(BigIntAsHex {
                            value: memory[offset + 1].unwrap().to_bigint(),
                        })
                    }
                    Op1Addr::AP => DerefOrImmediate::Deref(CellRef {
                        register: cairo_lang_casm::operand::Register::AP,
                        offset: off2 as i16,
                    }),
                    Op1Addr::FP => DerefOrImmediate::Deref(CellRef {
                        register: cairo_lang_casm::operand::Register::FP,
                        offset: off2 as i16,
                    }),
                    _ => unreachable!(),
                },
                relative: match pc_update {
                    PcUpdate::Jump => false,
                    PcUpdate::JumpRel => true,
                    _ => unreachable!(),
                },
            }),
            inc_ap: false,
            hints: Vec::new(),
        },
        InstructionRepr {
            off0: -1,
            off1: -1,
            off2,
            dst_register: Register::FP,
            op0_register: Register::FP,
            op1_addr: op1_addr @ (Op1Addr::AP | Op1Addr::FP | Op1Addr::Imm),
            res: Res::Op1,
            pc_update: pc_update @ (PcUpdate::JumpRel | PcUpdate::Jump),
            ap_update: ap_update @ (ApUpdate::Add1 | ApUpdate::Regular),
            fp_update: FpUpdate::Regular,
            opcode: Opcode::NOp,
        } => Instruction {
            body: InstructionBody::Jump(JumpInstruction {
                target: match op1_addr {
                    Op1Addr::Imm => {
                        assert_eq!(off2, 1);
                        DerefOrImmediate::Immediate(BigIntAsHex {
                            value: memory[offset + 1].unwrap().to_bigint(),
                        })
                    }
                    Op1Addr::AP => DerefOrImmediate::Deref(CellRef {
                        register: cairo_lang_casm::operand::Register::AP,
                        offset: off2 as i16,
                    }),
                    Op1Addr::FP => DerefOrImmediate::Deref(CellRef {
                        register: cairo_lang_casm::operand::Register::FP,
                        offset: off2 as i16,
                    }),
                    _ => unreachable!(),
                },
                relative: match pc_update {
                    PcUpdate::Jump => false,
                    PcUpdate::JumpRel => true,
                    _ => unreachable!(),
                },
            }),
            inc_ap: match ap_update {
                ApUpdate::Regular => false,
                ApUpdate::Add1 => true,
                _ => unreachable!(),
            },
            hints: Vec::new(),
        },
        InstructionRepr {
            off0,
            off1: -1,
            off2,
            dst_register,
            op0_register: Register::FP,
            op1_addr: op1_addr @ (Op1Addr::AP | Op1Addr::FP | Op1Addr::Imm),
            res: Res::Unconstrained,
            pc_update: PcUpdate::Jnz,
            ap_update: ap_update @ (ApUpdate::Add1 | ApUpdate::Regular),
            fp_update: FpUpdate::Regular,
            opcode: Opcode::NOp,
        } => Instruction {
            body: InstructionBody::Jnz(JnzInstruction {
                jump_offset: match op1_addr {
                    Op1Addr::Imm => {
                        assert_eq!(off2, 1);
                        DerefOrImmediate::Immediate(BigIntAsHex {
                            value: memory[offset + 1].unwrap().to_bigint(),
                        })
                    }
                    Op1Addr::AP => DerefOrImmediate::Deref(CellRef {
                        register: cairo_lang_casm::operand::Register::AP,
                        offset: off2 as i16,
                    }),
                    Op1Addr::FP => DerefOrImmediate::Deref(CellRef {
                        register: cairo_lang_casm::operand::Register::FP,
                        offset: off2 as i16,
                    }),
                    _ => unreachable!(),
                },
                condition: CellRef {
                    register: match dst_register {
                        Register::AP => cairo_lang_casm::operand::Register::AP,
                        Register::FP => cairo_lang_casm::operand::Register::FP,
                    },
                    offset: off0 as i16,
                },
            }),
            inc_ap: match ap_update {
                ApUpdate::Regular => false,
                ApUpdate::Add1 => true,
                _ => unreachable!(),
            },
            hints: Vec::new(),
        },
        InstructionRepr {
            off0: -2,
            off1: -1,
            off2: -1,
            dst_register: Register::FP,
            op0_register: Register::FP,
            op1_addr: Op1Addr::FP,
            res: Res::Op1,
            pc_update: PcUpdate::Jump,
            ap_update: ApUpdate::Regular,
            fp_update: FpUpdate::Dst,
            opcode: Opcode::Ret,
        } => Instruction {
            body: InstructionBody::Ret(RetInstruction {}),
            inc_ap: false,
            hints: Vec::new(),
        },
        _ => panic!(),
    }
}

struct ResDescription {
    off1: isize,
    off2: isize,
    imm: Option<Felt>,
    op0_register: Register,
    op1_addr: Op1Addr,
    res: Res,
}

fn decode_res_operand(desc: ResDescription) -> ResOperand {
    match desc {
        ResDescription {
            off1: -1,
            off2,
            imm: _,
            op0_register: Register::FP,
            op1_addr: op1_addr @ (Op1Addr::AP | Op1Addr::FP),
            res: Res::Op1,
        } => ResOperand::Deref(CellRef {
            register: match op1_addr {
                Op1Addr::AP => cairo_lang_casm::operand::Register::AP,
                Op1Addr::FP => cairo_lang_casm::operand::Register::FP,
                _ => unreachable!(),
            },
            offset: off2 as i16,
        }),
        ResDescription {
            off1,
            off2,
            imm: _,
            op0_register,
            op1_addr: Op1Addr::Op0,
            res: Res::Op1,
        } => ResOperand::DoubleDeref(
            CellRef {
                register: match op0_register {
                    Register::AP => cairo_lang_casm::operand::Register::AP,
                    Register::FP => cairo_lang_casm::operand::Register::FP,
                },
                offset: off1 as i16,
            },
            off2 as i16,
        ),
        ResDescription {
            off1: -1,
            off2: 1,
            imm: Some(imm),
            op0_register: Register::FP,
            op1_addr: Op1Addr::Imm,
            res: Res::Op1,
        } => ResOperand::Immediate(BigIntAsHex {
            value: imm.to_bigint(),
        }),
        ResDescription {
            off1,
            off2,
            imm,
            op0_register,
            op1_addr: op1_addr @ (Op1Addr::AP | Op1Addr::FP | Op1Addr::Imm),
            res: res @ (Res::Add | Res::Mul),
        } => ResOperand::BinOp(BinOpOperand {
            op: match res {
                Res::Add => Operation::Add,
                Res::Mul => Operation::Mul,
                _ => unreachable!(),
            },
            a: CellRef {
                register: match op0_register {
                    Register::AP => cairo_lang_casm::operand::Register::AP,
                    Register::FP => cairo_lang_casm::operand::Register::FP,
                },
                offset: off1 as i16,
            },
            b: match op1_addr {
                Op1Addr::Imm => {
                    assert_eq!(off2, 1);
                    DerefOrImmediate::Immediate(BigIntAsHex {
                        value: imm.unwrap().to_bigint(),
                    })
                }
                Op1Addr::AP => DerefOrImmediate::Deref(CellRef {
                    register: cairo_lang_casm::operand::Register::AP,
                    offset: off2 as i16,
                }),
                Op1Addr::FP => DerefOrImmediate::Deref(CellRef {
                    register: cairo_lang_casm::operand::Register::FP,
                    offset: off2 as i16,
                }),
                _ => unreachable!(),
            },
        }),
        _ => panic!(),
    }
}
