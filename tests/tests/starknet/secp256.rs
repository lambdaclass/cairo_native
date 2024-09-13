use crate::common::{load_cairo_path, run_native_program};
use cairo_lang_runner::SierraCasmRunner;
use cairo_lang_sierra::program::Program;
use cairo_native::{
    starknet::{Secp256k1Point, Secp256r1Point, StarknetSyscallHandler, SyscallResult, U256},
    JitValue,
};
use lazy_static::lazy_static;
use pretty_assertions_sorted::assert_eq;
use starknet_types_core::felt::Felt;
use std::collections::VecDeque;

#[derive(Debug, Default)]
struct SyscallHandler {
    secp256k1_new: (VecDeque<(U256, U256)>, VecDeque<Option<Secp256k1Point>>),
    secp256k1_add: (
        VecDeque<(Secp256k1Point, Secp256k1Point)>,
        VecDeque<Secp256k1Point>,
    ),
    secp256k1_mul: (VecDeque<(Secp256k1Point, U256)>, VecDeque<Secp256k1Point>),
    secp256k1_get_point_from_x: (VecDeque<(U256, bool)>, VecDeque<Option<Secp256k1Point>>),
    secp256k1_get_xy: (VecDeque<Secp256k1Point>, VecDeque<(U256, U256)>),

    secp256r1_new: (VecDeque<(U256, U256)>, VecDeque<Option<Secp256r1Point>>),
    secp256r1_add: (
        VecDeque<(Secp256r1Point, Secp256r1Point)>,
        VecDeque<Secp256r1Point>,
    ),
    secp256r1_mul: (VecDeque<(Secp256r1Point, U256)>, VecDeque<Secp256r1Point>),
    secp256r1_get_point_from_x: (VecDeque<(U256, bool)>, VecDeque<Option<Secp256r1Point>>),
    secp256r1_get_xy: (VecDeque<Secp256r1Point>, VecDeque<(U256, U256)>),
}

impl StarknetSyscallHandler for &mut SyscallHandler {
    fn get_block_hash(
        &mut self,
        _block_number: u64,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Felt> {
        unimplemented!()
    }

    fn get_execution_info(
        &mut self,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<cairo_native::starknet::ExecutionInfo> {
        unimplemented!()
    }

    fn get_execution_info_v2(
        &mut self,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<cairo_native::starknet::ExecutionInfoV2> {
        unimplemented!()
    }

    fn deploy(
        &mut self,
        _class_hash: Felt,
        _contract_address_salt: Felt,
        _calldata: &[Felt],
        _deploy_from_zero: bool,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<(Felt, Vec<Felt>)> {
        unimplemented!()
    }

    fn replace_class(&mut self, _class_hash: Felt, _remaining_gas: &mut u128) -> SyscallResult<()> {
        unimplemented!()
    }

    fn library_call(
        &mut self,
        _class_hash: Felt,
        _function_selector: Felt,
        _calldata: &[Felt],
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Vec<Felt>> {
        unimplemented!()
    }

    fn call_contract(
        &mut self,
        _address: Felt,
        _entry_point_selector: Felt,
        _calldata: &[Felt],
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Vec<Felt>> {
        unimplemented!()
    }

    fn storage_read(
        &mut self,
        _address_domain: u32,
        _address: Felt,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Felt> {
        unimplemented!()
    }

    fn storage_write(
        &mut self,
        _address_domain: u32,
        _address: Felt,
        _value: Felt,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<()> {
        unimplemented!()
    }

    fn emit_event(
        &mut self,
        _keys: &[Felt],
        _data: &[Felt],
        _remaining_gas: &mut u128,
    ) -> SyscallResult<()> {
        unimplemented!()
    }

    fn send_message_to_l1(
        &mut self,
        _to_address: Felt,
        _payload: &[Felt],
        _remaining_gas: &mut u128,
    ) -> SyscallResult<()> {
        unimplemented!()
    }

    fn keccak(&mut self, _input: &[u64], _remaining_gas: &mut u128) -> SyscallResult<U256> {
        unimplemented!()
    }

    fn secp256k1_new(
        &mut self,
        x: U256,
        y: U256,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256k1Point>> {
        let (args, rets) = &mut self.secp256k1_new;

        args.push_back((x, y));
        Ok(rets.pop_front().unwrap())
    }

    fn secp256k1_add(
        &mut self,
        p0: Secp256k1Point,
        p1: Secp256k1Point,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Secp256k1Point> {
        let (args, rets) = &mut self.secp256k1_add;

        args.push_back((p0, p1));
        Ok(rets.pop_front().unwrap())
    }

    fn secp256k1_mul(
        &mut self,
        p: Secp256k1Point,
        m: U256,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Secp256k1Point> {
        let (args, rets) = &mut self.secp256k1_mul;

        args.push_back((p, m));
        Ok(rets.pop_front().unwrap())
    }

    fn secp256k1_get_point_from_x(
        &mut self,
        x: U256,
        y_parity: bool,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256k1Point>> {
        let (args, rets) = &mut self.secp256k1_get_point_from_x;

        args.push_back((x, y_parity));
        Ok(rets.pop_front().unwrap())
    }

    fn secp256k1_get_xy(
        &mut self,
        p: Secp256k1Point,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<(U256, U256)> {
        let (args, rets) = &mut self.secp256k1_get_xy;

        args.push_back(p);
        Ok(rets.pop_front().unwrap())
    }

    fn secp256r1_new(
        &mut self,
        x: U256,
        y: U256,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256r1Point>> {
        let (args, rets) = &mut self.secp256r1_new;

        args.push_back((x, y));
        Ok(rets.pop_front().unwrap())
    }

    fn secp256r1_add(
        &mut self,
        p0: Secp256r1Point,
        p1: Secp256r1Point,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Secp256r1Point> {
        let (args, rets) = &mut self.secp256r1_add;

        args.push_back((p0, p1));
        Ok(rets.pop_front().unwrap())
    }

    fn secp256r1_mul(
        &mut self,
        p: Secp256r1Point,
        m: U256,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Secp256r1Point> {
        let (args, rets) = &mut self.secp256r1_mul;

        args.push_back((p, m));
        Ok(rets.pop_front().unwrap())
    }

    fn secp256r1_get_point_from_x(
        &mut self,
        x: U256,
        y_parity: bool,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256r1Point>> {
        let (args, rets) = &mut self.secp256r1_get_point_from_x;

        args.push_back((x, y_parity));
        Ok(rets.pop_front().unwrap())
    }

    fn secp256r1_get_xy(
        &mut self,
        p: Secp256r1Point,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<(U256, U256)> {
        let (args, rets) = &mut self.secp256r1_get_xy;

        args.push_back(p);
        Ok(rets.pop_front().unwrap())
    }

    fn sha256_process_block(
        &mut self,
        _prev_state: &[u32; 8],
        _current_block: &[u32; 16],
        _remaining_gas: &mut u128,
    ) -> SyscallResult<[u32; 8]> {
        unimplemented!()
    }
}

lazy_static! {
    static ref SECP256_PROGRAM: (String, Program, SierraCasmRunner) =
        load_cairo_path("tests/tests/starknet/programs/secp256.cairo");
}

#[test]
fn secp256k1_new() {
    let mut syscall_handler = SyscallHandler {
        secp256k1_new: (
            VecDeque::from([]),
            VecDeque::from([
                None,
                Some(Secp256k1Point {
                    x: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
                }),
                Some(Secp256k1Point {
                    x: U256 {
                        hi: u128::MAX,
                        lo: u128::MAX,
                    },
                    y: U256 {
                        hi: u128::MAX,
                        lo: u128::MAX,
                    },
                }),
            ]),
        ),
        ..Default::default()
    };

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256k1_new",
        &[
            JitValue::Struct {
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                debug_name: None,
            },
            JitValue::Struct {
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                debug_name: None,
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Enum {
                tag: 1,
                value: Box::new(JitValue::Struct {
                    fields: vec![],
                    debug_name: None
                }),
                debug_name: None,
            }),
            debug_name: None,
        },
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256k1_new",
        &[
            JitValue::Struct {
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
                debug_name: None,
            },
            JitValue::Struct {
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                debug_name: None,
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Enum {
                tag: 0,
                value: Box::new(JitValue::Secp256K1Point {
                    x: (0, 0),
                    y: (0, 0),
                }),
                debug_name: None,
            }),
            debug_name: None,
        },
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256k1_new",
        &[
            JitValue::Struct {
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
                debug_name: None,
            },
            JitValue::Struct {
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
                debug_name: None,
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Enum {
                tag: 0,
                value: Box::new(JitValue::Secp256K1Point {
                    x: (u128::MAX, u128::MAX),
                    y: (u128::MAX, u128::MAX),
                }),
                debug_name: None,
            }),
            debug_name: None,
        },
    );

    assert_eq!(
        syscall_handler.secp256k1_new.0,
        [
            (U256 { hi: 0, lo: 0 }, U256 { hi: 0, lo: 0 }),
            (
                U256 {
                    lo: 0,
                    hi: u128::MAX
                },
                U256 {
                    lo: u128::MAX,
                    hi: 0
                }
            ),
            (
                U256 {
                    hi: u128::MAX,
                    lo: u128::MAX
                },
                U256 {
                    hi: u128::MAX,
                    lo: u128::MAX
                }
            ),
        ],
    );
    assert!(syscall_handler.secp256k1_new.1.is_empty());
}

#[test]
fn secp256k1_add() {
    let mut syscall_handler = SyscallHandler {
        secp256k1_add: (
            VecDeque::from([]),
            VecDeque::from([
                Secp256k1Point {
                    x: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
                },
                Secp256k1Point {
                    x: U256 {
                        lo: u128::MAX,
                        hi: 0,
                    },
                    y: U256 {
                        lo: 0,
                        hi: u128::MAX,
                    },
                },
                Secp256k1Point {
                    x: U256 {
                        lo: u128::MAX,
                        hi: u128::MAX,
                    },
                    y: U256 {
                        lo: u128::MAX,
                        hi: u128::MAX,
                    },
                },
            ]),
        ),
        ..Default::default()
    };

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256k1_add",
        &[
            JitValue::Secp256K1Point {
                x: (0, 0),
                y: (0, 0),
            },
            JitValue::Secp256K1Point {
                x: (0, 0),
                y: (0, 0),
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Secp256K1Point {
                x: (0, 0),
                y: (0, 0),
            }),
            debug_name: None,
        },
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256k1_add",
        &[
            JitValue::Secp256K1Point {
                x: (0, u128::MAX),
                y: (u128::MAX, 0),
            },
            JitValue::Secp256K1Point {
                x: (u128::MAX, 0),
                y: (0, u128::MAX),
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Secp256K1Point {
                x: (u128::MAX, 0),
                y: (0, u128::MAX),
            }),
            debug_name: None,
        },
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256k1_add",
        &[
            JitValue::Secp256K1Point {
                x: (u128::MAX, u128::MAX),
                y: (u128::MAX, u128::MAX),
            },
            JitValue::Secp256K1Point {
                x: (u128::MAX, u128::MAX),
                y: (u128::MAX, u128::MAX),
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Secp256K1Point {
                x: (u128::MAX, u128::MAX),
                y: (u128::MAX, u128::MAX),
            }),
            debug_name: None,
        },
    );

    assert_eq!(
        syscall_handler.secp256k1_add.0,
        [
            (
                Secp256k1Point {
                    x: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
                },
                Secp256k1Point {
                    x: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
                },
            ),
            (
                Secp256k1Point {
                    x: U256 {
                        hi: u128::MAX,
                        lo: 0
                    },
                    y: U256 {
                        hi: 0,
                        lo: u128::MAX
                    },
                },
                Secp256k1Point {
                    x: U256 {
                        hi: 0,
                        lo: u128::MAX
                    },
                    y: U256 {
                        hi: u128::MAX,
                        lo: 0
                    },
                },
            ),
            (
                Secp256k1Point {
                    x: U256 {
                        lo: u128::MAX,
                        hi: u128::MAX
                    },
                    y: U256 {
                        lo: u128::MAX,
                        hi: u128::MAX
                    },
                },
                Secp256k1Point {
                    x: U256 {
                        lo: u128::MAX,
                        hi: u128::MAX
                    },
                    y: U256 {
                        lo: u128::MAX,
                        hi: u128::MAX
                    },
                },
            ),
        ],
    );
    assert!(syscall_handler.secp256k1_add.1.is_empty());
}

#[test]
fn secp256k1_mul() {
    let mut syscall_handler = SyscallHandler {
        secp256k1_mul: (
            VecDeque::from([]),
            VecDeque::from([
                Secp256k1Point {
                    x: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
                },
                Secp256k1Point {
                    x: U256 {
                        hi: u128::MAX,
                        lo: 0,
                    },
                    y: U256 {
                        hi: 0,
                        lo: u128::MAX,
                    },
                },
                Secp256k1Point {
                    x: U256 {
                        hi: u128::MAX,
                        lo: u128::MAX,
                    },
                    y: U256 {
                        hi: u128::MAX,
                        lo: u128::MAX,
                    },
                },
            ]),
        ),
        ..Default::default()
    };

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256k1_mul",
        &[
            JitValue::Secp256K1Point {
                x: (0, 0),
                y: (0, 0),
            },
            JitValue::Struct {
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                debug_name: None,
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Secp256K1Point {
                x: (0, 0),
                y: (0, 0),
            }),
            debug_name: None,
        },
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256k1_mul",
        &[
            JitValue::Secp256K1Point {
                x: (u128::MAX, 0),
                y: (0, u128::MAX),
            },
            JitValue::Struct {
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                debug_name: None,
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Secp256K1Point {
                x: (0, u128::MAX),
                y: (u128::MAX, 0),
            }),
            debug_name: None,
        },
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256k1_mul",
        &[
            JitValue::Secp256K1Point {
                x: (u128::MAX, 0),
                y: (0, u128::MAX),
            },
            JitValue::Struct {
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                debug_name: None,
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Secp256K1Point {
                x: (u128::MAX, u128::MAX),
                y: (u128::MAX, u128::MAX),
            }),
            debug_name: None,
        },
    );

    assert_eq!(
        syscall_handler.secp256k1_mul.0,
        [
            (
                Secp256k1Point {
                    x: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
                },
                U256 { hi: 0, lo: 0 },
            ),
            (
                Secp256k1Point {
                    x: U256 {
                        lo: u128::MAX,
                        hi: 0
                    },
                    y: U256 {
                        lo: 0,
                        hi: u128::MAX,
                    },
                },
                U256 {
                    lo: u128::MAX,
                    hi: 0,
                },
            ),
            (
                Secp256k1Point {
                    x: U256 {
                        hi: 0,
                        lo: u128::MAX,
                    },
                    y: U256 {
                        hi: u128::MAX,
                        lo: 0,
                    },
                },
                U256 {
                    hi: 0,
                    lo: u128::MAX,
                },
            ),
        ],
    );
    assert!(syscall_handler.secp256k1_mul.1.is_empty());
}

#[test]
fn secp256k1_get_point_from_x() {
    let mut syscall_handler = SyscallHandler {
        secp256k1_get_point_from_x: (
            VecDeque::from([]),
            VecDeque::from([
                None,
                Some(Secp256k1Point {
                    x: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
                }),
                Some(Secp256k1Point {
                    x: U256 {
                        hi: 0,
                        lo: u128::MAX,
                    },
                    y: U256 {
                        hi: u128::MAX,
                        lo: 0,
                    },
                }),
                Some(Secp256k1Point {
                    x: U256 {
                        hi: u128::MAX,
                        lo: 0,
                    },
                    y: U256 {
                        hi: 0,
                        lo: u128::MAX,
                    },
                }),
            ]),
        ),
        ..Default::default()
    };

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256k1_get_point_from_x",
        &[
            JitValue::Struct {
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                debug_name: None,
            },
            JitValue::Enum {
                tag: 0,
                value: Box::new(JitValue::Struct {
                    fields: vec![],
                    debug_name: None,
                }),
                debug_name: None,
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Enum {
                tag: 1,
                value: Box::new(JitValue::Struct {
                    fields: vec![],
                    debug_name: None,
                }),
                debug_name: None,
            }),
            debug_name: None
        },
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256k1_get_point_from_x",
        &[
            JitValue::Struct {
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
                debug_name: None,
            },
            JitValue::Enum {
                tag: 1,
                value: Box::new(JitValue::Struct {
                    fields: vec![],
                    debug_name: None,
                }),
                debug_name: None,
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Enum {
                tag: 0,
                value: Box::new(JitValue::Secp256K1Point {
                    x: (0, 0),
                    y: (0, 0),
                }),
                debug_name: None,
            }),
            debug_name: None,
        },
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256k1_get_point_from_x",
        &[
            JitValue::Struct {
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                debug_name: None,
            },
            JitValue::Enum {
                tag: 0,
                value: Box::new(JitValue::Struct {
                    fields: vec![],
                    debug_name: None,
                }),
                debug_name: None,
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Enum {
                tag: 0,
                value: Box::new(JitValue::Secp256K1Point {
                    x: (u128::MAX, 0),
                    y: (0, u128::MAX),
                }),
                debug_name: None,
            }),
            debug_name: None,
        },
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256k1_get_point_from_x",
        &[
            JitValue::Struct {
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
                debug_name: None,
            },
            JitValue::Enum {
                tag: 1,
                value: Box::new(JitValue::Struct {
                    fields: vec![],
                    debug_name: None,
                }),
                debug_name: None,
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Enum {
                tag: 0,
                value: Box::new(JitValue::Secp256K1Point {
                    x: (0, u128::MAX),
                    y: (u128::MAX, 0),
                }),
                debug_name: None,
            }),
            debug_name: None,
        },
    );

    assert_eq!(
        syscall_handler.secp256k1_get_point_from_x.0,
        [
            (U256 { hi: 0, lo: 0 }, false),
            (
                U256 {
                    lo: 0,
                    hi: u128::MAX,
                },
                true,
            ),
            (
                U256 {
                    lo: u128::MAX,
                    hi: 0,
                },
                false,
            ),
            (
                U256 {
                    hi: u128::MAX,
                    lo: u128::MAX,
                },
                true,
            ),
        ],
    );
    assert!(syscall_handler.secp256k1_get_point_from_x.1.is_empty());
}

#[test]
fn secp256k1_get_xy() {
    let mut syscall_handler = SyscallHandler {
        secp256k1_get_xy: (
            VecDeque::from([]),
            VecDeque::from([
                (U256 { hi: 0, lo: 0 }, U256 { hi: 0, lo: 0 }),
                (
                    U256 {
                        hi: 0,
                        lo: u128::MAX,
                    },
                    U256 {
                        hi: u128::MAX,
                        lo: 0,
                    },
                ),
                (
                    U256 {
                        hi: u128::MAX,
                        lo: 0,
                    },
                    U256 {
                        hi: 0,
                        lo: u128::MAX,
                    },
                ),
                (
                    U256 {
                        hi: u128::MAX,
                        lo: u128::MAX,
                    },
                    U256 {
                        hi: u128::MAX,
                        lo: u128::MAX,
                    },
                ),
            ]),
        ),
        ..Default::default()
    };

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256k1_get_xy",
        &[JitValue::Secp256K1Point {
            x: (0, 0),
            y: (0, 0),
        }],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                fields: vec![
                    JitValue::Struct {
                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                        debug_name: None,
                    },
                    JitValue::Struct {
                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                        debug_name: None,
                    },
                ],
                debug_name: None,
            }),
            debug_name: None,
        }
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256k1_get_xy",
        &[JitValue::Secp256K1Point {
            x: (0, u128::MAX),
            y: (u128::MAX, 0),
        }],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                fields: vec![
                    JitValue::Struct {
                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                        debug_name: None,
                    },
                    JitValue::Struct {
                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
                        debug_name: None,
                    },
                ],
                debug_name: None,
            }),
            debug_name: None,
        }
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256k1_get_xy",
        &[JitValue::Secp256K1Point {
            x: (u128::MAX, 0),
            y: (0, u128::MAX),
        }],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                fields: vec![
                    JitValue::Struct {
                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
                        debug_name: None,
                    },
                    JitValue::Struct {
                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                        debug_name: None,
                    },
                ],
                debug_name: None,
            }),
            debug_name: None,
        }
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256k1_get_xy",
        &[JitValue::Secp256K1Point {
            x: (u128::MAX, u128::MAX),
            y: (u128::MAX, u128::MAX),
        }],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                fields: vec![
                    JitValue::Struct {
                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
                        debug_name: None,
                    },
                    JitValue::Struct {
                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
                        debug_name: None,
                    },
                ],
                debug_name: None,
            }),
            debug_name: None,
        }
    );

    assert_eq!(
        syscall_handler.secp256k1_get_xy.0,
        [
            Secp256k1Point {
                x: U256 { hi: 0, lo: 0 },
                y: U256 { hi: 0, lo: 0 },
            },
            Secp256k1Point {
                x: U256 {
                    lo: 0,
                    hi: u128::MAX,
                },
                y: U256 {
                    lo: u128::MAX,
                    hi: 0,
                },
            },
            Secp256k1Point {
                x: U256 {
                    lo: u128::MAX,
                    hi: 0,
                },
                y: U256 {
                    lo: 0,
                    hi: u128::MAX,
                },
            },
            Secp256k1Point {
                x: U256 {
                    hi: u128::MAX,
                    lo: u128::MAX,
                },
                y: U256 {
                    hi: u128::MAX,
                    lo: u128::MAX,
                },
            },
        ],
    );
    assert!(syscall_handler.secp256k1_get_xy.1.is_empty());
}

#[test]
fn secp256r1_new() {
    let mut syscall_handler = SyscallHandler {
        secp256r1_new: (
            VecDeque::from([]),
            VecDeque::from([
                None,
                Some(Secp256r1Point {
                    x: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
                }),
                Some(Secp256r1Point {
                    x: U256 {
                        hi: u128::MAX,
                        lo: u128::MAX,
                    },
                    y: U256 {
                        hi: u128::MAX,
                        lo: u128::MAX,
                    },
                }),
            ]),
        ),
        ..Default::default()
    };

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256r1_new",
        &[
            JitValue::Struct {
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                debug_name: None,
            },
            JitValue::Struct {
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                debug_name: None,
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Enum {
                tag: 1,
                value: Box::new(JitValue::Struct {
                    fields: vec![],
                    debug_name: None
                }),
                debug_name: None,
            }),
            debug_name: None,
        },
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256r1_new",
        &[
            JitValue::Struct {
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
                debug_name: None,
            },
            JitValue::Struct {
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                debug_name: None,
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Enum {
                tag: 0,
                value: Box::new(JitValue::Secp256R1Point {
                    x: (0, 0),
                    y: (0, 0),
                }),
                debug_name: None,
            }),
            debug_name: None,
        },
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256r1_new",
        &[
            JitValue::Struct {
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
                debug_name: None,
            },
            JitValue::Struct {
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
                debug_name: None,
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Enum {
                tag: 0,
                value: Box::new(JitValue::Secp256R1Point {
                    x: (u128::MAX, u128::MAX),
                    y: (u128::MAX, u128::MAX),
                }),
                debug_name: None,
            }),
            debug_name: None,
        },
    );

    assert_eq!(
        syscall_handler.secp256r1_new.0,
        [
            (U256 { hi: 0, lo: 0 }, U256 { hi: 0, lo: 0 }),
            (
                U256 {
                    hi: u128::MAX,
                    lo: 0
                },
                U256 {
                    hi: 0,
                    lo: u128::MAX
                }
            ),
            (
                U256 {
                    lo: u128::MAX,
                    hi: u128::MAX
                },
                U256 {
                    hi: u128::MAX,
                    lo: u128::MAX
                }
            ),
        ],
    );
    assert!(syscall_handler.secp256r1_new.1.is_empty());
}

#[test]
fn secp256r1_add() {
    let mut syscall_handler = SyscallHandler {
        secp256r1_add: (
            VecDeque::from([]),
            VecDeque::from([
                Secp256r1Point {
                    x: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
                },
                Secp256r1Point {
                    x: U256 {
                        hi: u128::MAX,
                        lo: 0,
                    },
                    y: U256 {
                        hi: 0,
                        lo: u128::MAX,
                    },
                },
                Secp256r1Point {
                    x: U256 {
                        hi: u128::MAX,
                        lo: u128::MAX,
                    },
                    y: U256 {
                        hi: u128::MAX,
                        lo: u128::MAX,
                    },
                },
            ]),
        ),
        ..Default::default()
    };

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256r1_add",
        &[
            JitValue::Secp256R1Point {
                x: (0, 0),
                y: (0, 0),
            },
            JitValue::Secp256R1Point {
                x: (0, 0),
                y: (0, 0),
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Secp256R1Point {
                x: (0, 0),
                y: (0, 0),
            }),
            debug_name: None,
        },
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256r1_add",
        &[
            JitValue::Secp256R1Point {
                x: (u128::MAX, 0),
                y: (0, u128::MAX),
            },
            JitValue::Secp256R1Point {
                x: (0, u128::MAX),
                y: (u128::MAX, 0),
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Secp256R1Point {
                x: (0, u128::MAX),
                y: (u128::MAX, 0),
            }),
            debug_name: None,
        },
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256r1_add",
        &[
            JitValue::Secp256R1Point {
                x: (u128::MAX, u128::MAX),
                y: (u128::MAX, u128::MAX),
            },
            JitValue::Secp256R1Point {
                x: (u128::MAX, u128::MAX),
                y: (u128::MAX, u128::MAX),
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Secp256R1Point {
                x: (u128::MAX, u128::MAX),
                y: (u128::MAX, u128::MAX),
            }),
            debug_name: None,
        },
    );

    assert_eq!(
        syscall_handler.secp256r1_add.0,
        [
            (
                Secp256r1Point {
                    x: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
                },
                Secp256r1Point {
                    x: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
                },
            ),
            (
                Secp256r1Point {
                    x: U256 {
                        lo: u128::MAX,
                        hi: 0
                    },
                    y: U256 {
                        lo: 0,
                        hi: u128::MAX
                    },
                },
                Secp256r1Point {
                    x: U256 {
                        lo: 0,
                        hi: u128::MAX
                    },
                    y: U256 {
                        lo: u128::MAX,
                        hi: 0
                    },
                },
            ),
            (
                Secp256r1Point {
                    x: U256 {
                        hi: u128::MAX,
                        lo: u128::MAX
                    },
                    y: U256 {
                        hi: u128::MAX,
                        lo: u128::MAX
                    },
                },
                Secp256r1Point {
                    x: U256 {
                        hi: u128::MAX,
                        lo: u128::MAX
                    },
                    y: U256 {
                        hi: u128::MAX,
                        lo: u128::MAX
                    },
                },
            ),
        ],
    );
    assert!(syscall_handler.secp256r1_add.1.is_empty());
}

#[test]
fn secp256r1_mul() {
    let mut syscall_handler = SyscallHandler {
        secp256r1_mul: (
            VecDeque::from([]),
            VecDeque::from([
                Secp256r1Point {
                    x: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
                },
                Secp256r1Point {
                    x: U256 {
                        hi: u128::MAX,
                        lo: 0,
                    },
                    y: U256 {
                        hi: 0,
                        lo: u128::MAX,
                    },
                },
                Secp256r1Point {
                    x: U256 {
                        hi: u128::MAX,
                        lo: u128::MAX,
                    },
                    y: U256 {
                        hi: u128::MAX,
                        lo: u128::MAX,
                    },
                },
            ]),
        ),
        ..Default::default()
    };

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256r1_mul",
        &[
            JitValue::Secp256R1Point {
                x: (0, 0),
                y: (0, 0),
            },
            JitValue::Struct {
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                debug_name: None,
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Secp256R1Point {
                x: (0, 0),
                y: (0, 0),
            }),
            debug_name: None,
        },
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256r1_mul",
        &[
            JitValue::Secp256R1Point {
                x: (u128::MAX, 0),
                y: (0, u128::MAX),
            },
            JitValue::Struct {
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                debug_name: None,
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Secp256R1Point {
                x: (0, u128::MAX),
                y: (u128::MAX, 0),
            }),
            debug_name: None,
        },
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256r1_mul",
        &[
            JitValue::Secp256R1Point {
                x: (0, u128::MAX),
                y: (u128::MAX, 0),
            },
            JitValue::Struct {
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
                debug_name: None,
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Secp256R1Point {
                x: (u128::MAX, u128::MAX),
                y: (u128::MAX, u128::MAX),
            }),
            debug_name: None,
        },
    );

    assert_eq!(
        syscall_handler.secp256r1_mul.0,
        [
            (
                Secp256r1Point {
                    x: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
                },
                U256 { hi: 0, lo: 0 },
            ),
            (
                Secp256r1Point {
                    x: U256 {
                        lo: u128::MAX,
                        hi: 0
                    },
                    y: U256 {
                        lo: 0,
                        hi: u128::MAX,
                    },
                },
                U256 {
                    lo: u128::MAX,
                    hi: 0,
                },
            ),
            (
                Secp256r1Point {
                    x: U256 {
                        lo: 0,
                        hi: u128::MAX,
                    },
                    y: U256 {
                        lo: u128::MAX,
                        hi: 0,
                    },
                },
                U256 {
                    lo: 0,
                    hi: u128::MAX,
                },
            ),
        ],
    );
    assert!(syscall_handler.secp256r1_mul.1.is_empty());
}

#[test]
fn secp256r1_get_point_from_x() {
    let mut syscall_handler = SyscallHandler {
        secp256r1_get_point_from_x: (
            VecDeque::from([]),
            VecDeque::from([
                None,
                Some(Secp256r1Point {
                    x: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
                }),
                Some(Secp256r1Point {
                    x: U256 {
                        hi: 0,
                        lo: u128::MAX,
                    },
                    y: U256 {
                        hi: u128::MAX,
                        lo: 0,
                    },
                }),
                Some(Secp256r1Point {
                    x: U256 {
                        hi: u128::MAX,
                        lo: 0,
                    },
                    y: U256 {
                        hi: 0,
                        lo: u128::MAX,
                    },
                }),
            ]),
        ),
        ..Default::default()
    };

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256r1_get_point_from_x",
        &[
            JitValue::Struct {
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                debug_name: None,
            },
            JitValue::Enum {
                tag: 0,
                value: Box::new(JitValue::Struct {
                    fields: vec![],
                    debug_name: None,
                }),
                debug_name: None,
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Enum {
                tag: 1,
                value: Box::new(JitValue::Struct {
                    fields: vec![],
                    debug_name: None,
                }),
                debug_name: None,
            }),
            debug_name: None
        },
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256r1_get_point_from_x",
        &[
            JitValue::Struct {
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
                debug_name: None,
            },
            JitValue::Enum {
                tag: 1,
                value: Box::new(JitValue::Struct {
                    fields: vec![],
                    debug_name: None,
                }),
                debug_name: None,
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Enum {
                tag: 0,
                value: Box::new(JitValue::Secp256R1Point {
                    x: (0, 0),
                    y: (0, 0),
                }),
                debug_name: None,
            }),
            debug_name: None,
        },
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256r1_get_point_from_x",
        &[
            JitValue::Struct {
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                debug_name: None,
            },
            JitValue::Enum {
                tag: 0,
                value: Box::new(JitValue::Struct {
                    fields: vec![],
                    debug_name: None,
                }),
                debug_name: None,
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Enum {
                tag: 0,
                value: Box::new(JitValue::Secp256R1Point {
                    x: (u128::MAX, 0),
                    y: (0, u128::MAX),
                }),
                debug_name: None,
            }),
            debug_name: None,
        },
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256r1_get_point_from_x",
        &[
            JitValue::Struct {
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
                debug_name: None,
            },
            JitValue::Enum {
                tag: 1,
                value: Box::new(JitValue::Struct {
                    fields: vec![],
                    debug_name: None,
                }),
                debug_name: None,
            },
        ],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Enum {
                tag: 0,
                value: Box::new(JitValue::Secp256R1Point {
                    x: (0, u128::MAX),
                    y: (u128::MAX, 0),
                }),
                debug_name: None,
            }),
            debug_name: None,
        },
    );

    assert_eq!(
        syscall_handler.secp256r1_get_point_from_x.0,
        [
            (U256 { hi: 0, lo: 0 }, false),
            (
                U256 {
                    lo: 0,
                    hi: u128::MAX,
                },
                true,
            ),
            (
                U256 {
                    lo: u128::MAX,
                    hi: 0,
                },
                false,
            ),
            (
                U256 {
                    hi: u128::MAX,
                    lo: u128::MAX,
                },
                true,
            ),
        ],
    );
    assert!(syscall_handler.secp256r1_get_point_from_x.1.is_empty());
}

#[test]
fn secp256r1_get_xy() {
    let mut syscall_handler = SyscallHandler {
        secp256r1_get_xy: (
            VecDeque::from([]),
            VecDeque::from([
                (U256 { hi: 0, lo: 0 }, U256 { hi: 0, lo: 0 }),
                (
                    U256 {
                        hi: 0,
                        lo: u128::MAX,
                    },
                    U256 {
                        hi: u128::MAX,
                        lo: 0,
                    },
                ),
                (
                    U256 {
                        hi: u128::MAX,
                        lo: 0,
                    },
                    U256 {
                        hi: 0,
                        lo: u128::MAX,
                    },
                ),
                (
                    U256 {
                        hi: u128::MAX,
                        lo: u128::MAX,
                    },
                    U256 {
                        hi: u128::MAX,
                        lo: u128::MAX,
                    },
                ),
            ]),
        ),
        ..Default::default()
    };

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256r1_get_xy",
        &[JitValue::Secp256R1Point {
            x: (0, 0),
            y: (0, 0),
        }],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                fields: vec![
                    JitValue::Struct {
                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                        debug_name: None,
                    },
                    JitValue::Struct {
                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                        debug_name: None,
                    },
                ],
                debug_name: None,
            }),
            debug_name: None,
        }
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256r1_get_xy",
        &[JitValue::Secp256R1Point {
            x: (0, u128::MAX),
            y: (u128::MAX, 0),
        }],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                fields: vec![
                    JitValue::Struct {
                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                        debug_name: None,
                    },
                    JitValue::Struct {
                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
                        debug_name: None,
                    },
                ],
                debug_name: None,
            }),
            debug_name: None,
        }
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256r1_get_xy",
        &[JitValue::Secp256R1Point {
            x: (u128::MAX, 0),
            y: (0, u128::MAX),
        }],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                fields: vec![
                    JitValue::Struct {
                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
                        debug_name: None,
                    },
                    JitValue::Struct {
                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                        debug_name: None,
                    },
                ],
                debug_name: None,
            }),
            debug_name: None,
        }
    );

    let result = run_native_program(
        &SECP256_PROGRAM,
        "secp256r1_get_xy",
        &[JitValue::Secp256R1Point {
            x: (u128::MAX, u128::MAX),
            y: (u128::MAX, u128::MAX),
        }],
        Some(u128::MAX),
        Some(&mut syscall_handler),
    );
    assert_eq!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                fields: vec![
                    JitValue::Struct {
                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
                        debug_name: None,
                    },
                    JitValue::Struct {
                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
                        debug_name: None,
                    },
                ],
                debug_name: None,
            }),
            debug_name: None,
        }
    );

    assert_eq!(
        syscall_handler.secp256r1_get_xy.0,
        [
            Secp256r1Point {
                x: U256 { hi: 0, lo: 0 },
                y: U256 { hi: 0, lo: 0 },
            },
            Secp256r1Point {
                x: U256 {
                    lo: 0,
                    hi: u128::MAX,
                },
                y: U256 {
                    lo: u128::MAX,
                    hi: 0,
                },
            },
            Secp256r1Point {
                x: U256 {
                    lo: u128::MAX,
                    hi: 0,
                },
                y: U256 {
                    lo: 0,
                    hi: u128::MAX,
                },
            },
            Secp256r1Point {
                x: U256 {
                    hi: u128::MAX,
                    lo: u128::MAX,
                },
                y: U256 {
                    hi: u128::MAX,
                    lo: u128::MAX,
                },
            },
        ],
    );
    assert!(syscall_handler.secp256r1_get_xy.1.is_empty());
}
