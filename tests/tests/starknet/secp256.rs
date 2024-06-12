//use crate::common::{load_cairo_path, run_native_program};
use crate::common::{load_cairo_path, run_native_program};
//use cairo_lang_runner::SierraCasmRunner;
use cairo_lang_runner::SierraCasmRunner;
//use cairo_lang_sierra::program::Program;
use cairo_lang_sierra::program::Program;
//use cairo_native::{
use cairo_native::{
//    starknet::{Secp256k1Point, Secp256r1Point, StarknetSyscallHandler, SyscallResult, U256},
    starknet::{Secp256k1Point, Secp256r1Point, StarknetSyscallHandler, SyscallResult, U256},
//    values::JitValue,
    values::JitValue,
//};
};
//use lazy_static::lazy_static;
use lazy_static::lazy_static;
//use pretty_assertions_sorted::assert_eq;
use pretty_assertions_sorted::assert_eq;
//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
//use std::collections::VecDeque;
use std::collections::VecDeque;
//

//#[derive(Debug, Default)]
#[derive(Debug, Default)]
//struct SyscallHandler {
struct SyscallHandler {
//    secp256k1_new: (VecDeque<(U256, U256)>, VecDeque<Option<Secp256k1Point>>),
    secp256k1_new: (VecDeque<(U256, U256)>, VecDeque<Option<Secp256k1Point>>),
//    secp256k1_add: (
    secp256k1_add: (
//        VecDeque<(Secp256k1Point, Secp256k1Point)>,
        VecDeque<(Secp256k1Point, Secp256k1Point)>,
//        VecDeque<Secp256k1Point>,
        VecDeque<Secp256k1Point>,
//    ),
    ),
//    secp256k1_mul: (VecDeque<(Secp256k1Point, U256)>, VecDeque<Secp256k1Point>),
    secp256k1_mul: (VecDeque<(Secp256k1Point, U256)>, VecDeque<Secp256k1Point>),
//    secp256k1_get_point_from_x: (VecDeque<(U256, bool)>, VecDeque<Option<Secp256k1Point>>),
    secp256k1_get_point_from_x: (VecDeque<(U256, bool)>, VecDeque<Option<Secp256k1Point>>),
//    secp256k1_get_xy: (VecDeque<Secp256k1Point>, VecDeque<(U256, U256)>),
    secp256k1_get_xy: (VecDeque<Secp256k1Point>, VecDeque<(U256, U256)>),
//

//    secp256r1_new: (VecDeque<(U256, U256)>, VecDeque<Option<Secp256r1Point>>),
    secp256r1_new: (VecDeque<(U256, U256)>, VecDeque<Option<Secp256r1Point>>),
//    secp256r1_add: (
    secp256r1_add: (
//        VecDeque<(Secp256r1Point, Secp256r1Point)>,
        VecDeque<(Secp256r1Point, Secp256r1Point)>,
//        VecDeque<Secp256r1Point>,
        VecDeque<Secp256r1Point>,
//    ),
    ),
//    secp256r1_mul: (VecDeque<(Secp256r1Point, U256)>, VecDeque<Secp256r1Point>),
    secp256r1_mul: (VecDeque<(Secp256r1Point, U256)>, VecDeque<Secp256r1Point>),
//    secp256r1_get_point_from_x: (VecDeque<(U256, bool)>, VecDeque<Option<Secp256r1Point>>),
    secp256r1_get_point_from_x: (VecDeque<(U256, bool)>, VecDeque<Option<Secp256r1Point>>),
//    secp256r1_get_xy: (VecDeque<Secp256r1Point>, VecDeque<(U256, U256)>),
    secp256r1_get_xy: (VecDeque<Secp256r1Point>, VecDeque<(U256, U256)>),
//}
}
//

//impl StarknetSyscallHandler for &mut SyscallHandler {
impl StarknetSyscallHandler for &mut SyscallHandler {
//    fn get_block_hash(
    fn get_block_hash(
//        &mut self,
        &mut self,
//        _block_number: u64,
        _block_number: u64,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Felt> {
    ) -> SyscallResult<Felt> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn get_execution_info(
    fn get_execution_info(
//        &mut self,
        &mut self,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<cairo_native::starknet::ExecutionInfo> {
    ) -> SyscallResult<cairo_native::starknet::ExecutionInfo> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn get_execution_info_v2(
    fn get_execution_info_v2(
//        &mut self,
        &mut self,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<cairo_native::starknet::ExecutionInfoV2> {
    ) -> SyscallResult<cairo_native::starknet::ExecutionInfoV2> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn deploy(
    fn deploy(
//        &mut self,
        &mut self,
//        _class_hash: Felt,
        _class_hash: Felt,
//        _contract_address_salt: Felt,
        _contract_address_salt: Felt,
//        _calldata: &[Felt],
        _calldata: &[Felt],
//        _deploy_from_zero: bool,
        _deploy_from_zero: bool,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<(Felt, Vec<Felt>)> {
    ) -> SyscallResult<(Felt, Vec<Felt>)> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn replace_class(&mut self, _class_hash: Felt, _remaining_gas: &mut u128) -> SyscallResult<()> {
    fn replace_class(&mut self, _class_hash: Felt, _remaining_gas: &mut u128) -> SyscallResult<()> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn library_call(
    fn library_call(
//        &mut self,
        &mut self,
//        _class_hash: Felt,
        _class_hash: Felt,
//        _function_selector: Felt,
        _function_selector: Felt,
//        _calldata: &[Felt],
        _calldata: &[Felt],
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Vec<Felt>> {
    ) -> SyscallResult<Vec<Felt>> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn call_contract(
    fn call_contract(
//        &mut self,
        &mut self,
//        _address: Felt,
        _address: Felt,
//        _entry_point_selector: Felt,
        _entry_point_selector: Felt,
//        _calldata: &[Felt],
        _calldata: &[Felt],
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Vec<Felt>> {
    ) -> SyscallResult<Vec<Felt>> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn storage_read(
    fn storage_read(
//        &mut self,
        &mut self,
//        _address_domain: u32,
        _address_domain: u32,
//        _address: Felt,
        _address: Felt,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Felt> {
    ) -> SyscallResult<Felt> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn storage_write(
    fn storage_write(
//        &mut self,
        &mut self,
//        _address_domain: u32,
        _address_domain: u32,
//        _address: Felt,
        _address: Felt,
//        _value: Felt,
        _value: Felt,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<()> {
    ) -> SyscallResult<()> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn emit_event(
    fn emit_event(
//        &mut self,
        &mut self,
//        _keys: &[Felt],
        _keys: &[Felt],
//        _data: &[Felt],
        _data: &[Felt],
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<()> {
    ) -> SyscallResult<()> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn send_message_to_l1(
    fn send_message_to_l1(
//        &mut self,
        &mut self,
//        _to_address: Felt,
        _to_address: Felt,
//        _payload: &[Felt],
        _payload: &[Felt],
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<()> {
    ) -> SyscallResult<()> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn keccak(&mut self, _input: &[u64], _remaining_gas: &mut u128) -> SyscallResult<U256> {
    fn keccak(&mut self, _input: &[u64], _remaining_gas: &mut u128) -> SyscallResult<U256> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn secp256k1_new(
    fn secp256k1_new(
//        &mut self,
        &mut self,
//        x: U256,
        x: U256,
//        y: U256,
        y: U256,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Option<Secp256k1Point>> {
    ) -> SyscallResult<Option<Secp256k1Point>> {
//        let (args, rets) = &mut self.secp256k1_new;
        let (args, rets) = &mut self.secp256k1_new;
//

//        args.push_back((x, y));
        args.push_back((x, y));
//        Ok(rets.pop_front().unwrap())
        Ok(rets.pop_front().unwrap())
//    }
    }
//

//    fn secp256k1_add(
    fn secp256k1_add(
//        &mut self,
        &mut self,
//        p0: Secp256k1Point,
        p0: Secp256k1Point,
//        p1: Secp256k1Point,
        p1: Secp256k1Point,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Secp256k1Point> {
    ) -> SyscallResult<Secp256k1Point> {
//        let (args, rets) = &mut self.secp256k1_add;
        let (args, rets) = &mut self.secp256k1_add;
//

//        args.push_back((p0, p1));
        args.push_back((p0, p1));
//        Ok(rets.pop_front().unwrap())
        Ok(rets.pop_front().unwrap())
//    }
    }
//

//    fn secp256k1_mul(
    fn secp256k1_mul(
//        &mut self,
        &mut self,
//        p: Secp256k1Point,
        p: Secp256k1Point,
//        m: U256,
        m: U256,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Secp256k1Point> {
    ) -> SyscallResult<Secp256k1Point> {
//        let (args, rets) = &mut self.secp256k1_mul;
        let (args, rets) = &mut self.secp256k1_mul;
//

//        args.push_back((p, m));
        args.push_back((p, m));
//        Ok(rets.pop_front().unwrap())
        Ok(rets.pop_front().unwrap())
//    }
    }
//

//    fn secp256k1_get_point_from_x(
    fn secp256k1_get_point_from_x(
//        &mut self,
        &mut self,
//        x: U256,
        x: U256,
//        y_parity: bool,
        y_parity: bool,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Option<Secp256k1Point>> {
    ) -> SyscallResult<Option<Secp256k1Point>> {
//        let (args, rets) = &mut self.secp256k1_get_point_from_x;
        let (args, rets) = &mut self.secp256k1_get_point_from_x;
//

//        args.push_back((x, y_parity));
        args.push_back((x, y_parity));
//        Ok(rets.pop_front().unwrap())
        Ok(rets.pop_front().unwrap())
//    }
    }
//

//    fn secp256k1_get_xy(
    fn secp256k1_get_xy(
//        &mut self,
        &mut self,
//        p: Secp256k1Point,
        p: Secp256k1Point,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<(U256, U256)> {
    ) -> SyscallResult<(U256, U256)> {
//        let (args, rets) = &mut self.secp256k1_get_xy;
        let (args, rets) = &mut self.secp256k1_get_xy;
//

//        args.push_back(p);
        args.push_back(p);
//        Ok(rets.pop_front().unwrap())
        Ok(rets.pop_front().unwrap())
//    }
    }
//

//    fn secp256r1_new(
    fn secp256r1_new(
//        &mut self,
        &mut self,
//        x: U256,
        x: U256,
//        y: U256,
        y: U256,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Option<Secp256r1Point>> {
    ) -> SyscallResult<Option<Secp256r1Point>> {
//        let (args, rets) = &mut self.secp256r1_new;
        let (args, rets) = &mut self.secp256r1_new;
//

//        args.push_back((x, y));
        args.push_back((x, y));
//        Ok(rets.pop_front().unwrap())
        Ok(rets.pop_front().unwrap())
//    }
    }
//

//    fn secp256r1_add(
    fn secp256r1_add(
//        &mut self,
        &mut self,
//        p0: Secp256r1Point,
        p0: Secp256r1Point,
//        p1: Secp256r1Point,
        p1: Secp256r1Point,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Secp256r1Point> {
    ) -> SyscallResult<Secp256r1Point> {
//        let (args, rets) = &mut self.secp256r1_add;
        let (args, rets) = &mut self.secp256r1_add;
//

//        args.push_back((p0, p1));
        args.push_back((p0, p1));
//        Ok(rets.pop_front().unwrap())
        Ok(rets.pop_front().unwrap())
//    }
    }
//

//    fn secp256r1_mul(
    fn secp256r1_mul(
//        &mut self,
        &mut self,
//        p: Secp256r1Point,
        p: Secp256r1Point,
//        m: U256,
        m: U256,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Secp256r1Point> {
    ) -> SyscallResult<Secp256r1Point> {
//        let (args, rets) = &mut self.secp256r1_mul;
        let (args, rets) = &mut self.secp256r1_mul;
//

//        args.push_back((p, m));
        args.push_back((p, m));
//        Ok(rets.pop_front().unwrap())
        Ok(rets.pop_front().unwrap())
//    }
    }
//

//    fn secp256r1_get_point_from_x(
    fn secp256r1_get_point_from_x(
//        &mut self,
        &mut self,
//        x: U256,
        x: U256,
//        y_parity: bool,
        y_parity: bool,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Option<Secp256r1Point>> {
    ) -> SyscallResult<Option<Secp256r1Point>> {
//        let (args, rets) = &mut self.secp256r1_get_point_from_x;
        let (args, rets) = &mut self.secp256r1_get_point_from_x;
//

//        args.push_back((x, y_parity));
        args.push_back((x, y_parity));
//        Ok(rets.pop_front().unwrap())
        Ok(rets.pop_front().unwrap())
//    }
    }
//

//    fn secp256r1_get_xy(
    fn secp256r1_get_xy(
//        &mut self,
        &mut self,
//        p: Secp256r1Point,
        p: Secp256r1Point,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<(U256, U256)> {
    ) -> SyscallResult<(U256, U256)> {
//        let (args, rets) = &mut self.secp256r1_get_xy;
        let (args, rets) = &mut self.secp256r1_get_xy;
//

//        args.push_back(p);
        args.push_back(p);
//        Ok(rets.pop_front().unwrap())
        Ok(rets.pop_front().unwrap())
//    }
    }
//}
}
//

//lazy_static! {
lazy_static! {
//    static ref SECP256_PROGRAM: (String, Program, SierraCasmRunner) =
    static ref SECP256_PROGRAM: (String, Program, SierraCasmRunner) =
//        load_cairo_path("tests/tests/starknet/programs/secp256.cairo");
        load_cairo_path("tests/tests/starknet/programs/secp256.cairo");
//}
}
//

//#[test]
#[test]
//fn secp256k1_new() {
fn secp256k1_new() {
//    let mut syscall_handler = SyscallHandler {
    let mut syscall_handler = SyscallHandler {
//        secp256k1_new: (
        secp256k1_new: (
//            VecDeque::from([]),
            VecDeque::from([]),
//            VecDeque::from([
            VecDeque::from([
//                None,
                None,
//                Some(Secp256k1Point {
                Some(Secp256k1Point {
//                    x: U256 { hi: 0, lo: 0 },
                    x: U256 { hi: 0, lo: 0 },
//                    y: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
//                }),
                }),
//                Some(Secp256k1Point {
                Some(Secp256k1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                }),
                }),
//            ]),
            ]),
//        ),
        ),
//        ..Default::default()
        ..Default::default()
//    };
    };
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256k1_new",
        "secp256k1_new",
//        &[
        &[
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Enum {
            value: Box::new(JitValue::Enum {
//                tag: 1,
                tag: 1,
//                value: Box::new(JitValue::Struct {
                value: Box::new(JitValue::Struct {
//                    fields: vec![],
                    fields: vec![],
//                    debug_name: None
                    debug_name: None
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256k1_new",
        "secp256k1_new",
//        &[
        &[
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Enum {
            value: Box::new(JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Secp256K1Point {
                value: Box::new(JitValue::Secp256K1Point {
//                    x: (0, 0),
                    x: (0, 0),
//                    y: (0, 0),
                    y: (0, 0),
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256k1_new",
        "secp256k1_new",
//        &[
        &[
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Enum {
            value: Box::new(JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Secp256K1Point {
                value: Box::new(JitValue::Secp256K1Point {
//                    x: (u128::MAX, u128::MAX),
                    x: (u128::MAX, u128::MAX),
//                    y: (u128::MAX, u128::MAX),
                    y: (u128::MAX, u128::MAX),
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    assert_eq!(
    assert_eq!(
//        syscall_handler.secp256k1_new.0,
        syscall_handler.secp256k1_new.0,
//        [
        [
//            (U256 { hi: 0, lo: 0 }, U256 { hi: 0, lo: 0 }),
            (U256 { hi: 0, lo: 0 }, U256 { hi: 0, lo: 0 }),
//            (
            (
//                U256 {
                U256 {
//                    hi: u128::MAX,
                    hi: u128::MAX,
//                    lo: 0
                    lo: 0
//                },
                },
//                U256 {
                U256 {
//                    hi: 0,
                    hi: 0,
//                    lo: u128::MAX
                    lo: u128::MAX
//                }
                }
//            ),
            ),
//            (
            (
//                U256 {
                U256 {
//                    hi: u128::MAX,
                    hi: u128::MAX,
//                    lo: u128::MAX
                    lo: u128::MAX
//                },
                },
//                U256 {
                U256 {
//                    hi: u128::MAX,
                    hi: u128::MAX,
//                    lo: u128::MAX
                    lo: u128::MAX
//                }
                }
//            ),
            ),
//        ],
        ],
//    );
    );
//    assert!(syscall_handler.secp256k1_new.1.is_empty());
    assert!(syscall_handler.secp256k1_new.1.is_empty());
//}
}
//

//#[test]
#[test]
//fn secp256k1_add() {
fn secp256k1_add() {
//    let mut syscall_handler = SyscallHandler {
    let mut syscall_handler = SyscallHandler {
//        secp256k1_add: (
        secp256k1_add: (
//            VecDeque::from([]),
            VecDeque::from([]),
//            VecDeque::from([
            VecDeque::from([
//                Secp256k1Point {
                Secp256k1Point {
//                    x: U256 { hi: 0, lo: 0 },
                    x: U256 { hi: 0, lo: 0 },
//                    y: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
//                },
                },
//                Secp256k1Point {
                Secp256k1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: 0,
                        lo: 0,
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: 0,
                        hi: 0,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                },
                },
//                Secp256k1Point {
                Secp256k1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                },
                },
//            ]),
            ]),
//        ),
        ),
//        ..Default::default()
        ..Default::default()
//    };
    };
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256k1_add",
        "secp256k1_add",
//        &[
        &[
//            JitValue::Secp256K1Point {
            JitValue::Secp256K1Point {
//                x: (0, 0),
                x: (0, 0),
//                y: (0, 0),
                y: (0, 0),
//            },
            },
//            JitValue::Secp256K1Point {
            JitValue::Secp256K1Point {
//                x: (0, 0),
                x: (0, 0),
//                y: (0, 0),
                y: (0, 0),
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Secp256K1Point {
            value: Box::new(JitValue::Secp256K1Point {
//                x: (0, 0),
                x: (0, 0),
//                y: (0, 0),
                y: (0, 0),
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256k1_add",
        "secp256k1_add",
//        &[
        &[
//            JitValue::Secp256K1Point {
            JitValue::Secp256K1Point {
//                x: (u128::MAX, 0),
                x: (u128::MAX, 0),
//                y: (0, u128::MAX),
                y: (0, u128::MAX),
//            },
            },
//            JitValue::Secp256K1Point {
            JitValue::Secp256K1Point {
//                x: (0, u128::MAX),
                x: (0, u128::MAX),
//                y: (u128::MAX, 0),
                y: (u128::MAX, 0),
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Secp256K1Point {
            value: Box::new(JitValue::Secp256K1Point {
//                x: (u128::MAX, 0),
                x: (u128::MAX, 0),
//                y: (0, u128::MAX),
                y: (0, u128::MAX),
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256k1_add",
        "secp256k1_add",
//        &[
        &[
//            JitValue::Secp256K1Point {
            JitValue::Secp256K1Point {
//                x: (u128::MAX, u128::MAX),
                x: (u128::MAX, u128::MAX),
//                y: (u128::MAX, u128::MAX),
                y: (u128::MAX, u128::MAX),
//            },
            },
//            JitValue::Secp256K1Point {
            JitValue::Secp256K1Point {
//                x: (u128::MAX, u128::MAX),
                x: (u128::MAX, u128::MAX),
//                y: (u128::MAX, u128::MAX),
                y: (u128::MAX, u128::MAX),
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Secp256K1Point {
            value: Box::new(JitValue::Secp256K1Point {
//                x: (u128::MAX, u128::MAX),
                x: (u128::MAX, u128::MAX),
//                y: (u128::MAX, u128::MAX),
                y: (u128::MAX, u128::MAX),
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    assert_eq!(
    assert_eq!(
//        syscall_handler.secp256k1_add.0,
        syscall_handler.secp256k1_add.0,
//        [
        [
//            (
            (
//                Secp256k1Point {
                Secp256k1Point {
//                    x: U256 { hi: 0, lo: 0 },
                    x: U256 { hi: 0, lo: 0 },
//                    y: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
//                },
                },
//                Secp256k1Point {
                Secp256k1Point {
//                    x: U256 { hi: 0, lo: 0 },
                    x: U256 { hi: 0, lo: 0 },
//                    y: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
//                },
                },
//            ),
            ),
//            (
            (
//                Secp256k1Point {
                Secp256k1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: 0
                        lo: 0
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: 0,
                        hi: 0,
//                        lo: u128::MAX
                        lo: u128::MAX
//                    },
                    },
//                },
                },
//                Secp256k1Point {
                Secp256k1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: 0,
                        hi: 0,
//                        lo: u128::MAX
                        lo: u128::MAX
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: 0
                        lo: 0
//                    },
                    },
//                },
                },
//            ),
            ),
//            (
            (
//                Secp256k1Point {
                Secp256k1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX
                        lo: u128::MAX
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX
                        lo: u128::MAX
//                    },
                    },
//                },
                },
//                Secp256k1Point {
                Secp256k1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX
                        lo: u128::MAX
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX
                        lo: u128::MAX
//                    },
                    },
//                },
                },
//            ),
            ),
//        ],
        ],
//    );
    );
//    assert!(syscall_handler.secp256k1_add.1.is_empty());
    assert!(syscall_handler.secp256k1_add.1.is_empty());
//}
}
//

//#[test]
#[test]
//fn secp256k1_mul() {
fn secp256k1_mul() {
//    let mut syscall_handler = SyscallHandler {
    let mut syscall_handler = SyscallHandler {
//        secp256k1_mul: (
        secp256k1_mul: (
//            VecDeque::from([]),
            VecDeque::from([]),
//            VecDeque::from([
            VecDeque::from([
//                Secp256k1Point {
                Secp256k1Point {
//                    x: U256 { hi: 0, lo: 0 },
                    x: U256 { hi: 0, lo: 0 },
//                    y: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
//                },
                },
//                Secp256k1Point {
                Secp256k1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: 0,
                        lo: 0,
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: 0,
                        hi: 0,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                },
                },
//                Secp256k1Point {
                Secp256k1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                },
                },
//            ]),
            ]),
//        ),
        ),
//        ..Default::default()
        ..Default::default()
//    };
    };
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256k1_mul",
        "secp256k1_mul",
//        &[
        &[
//            JitValue::Secp256K1Point {
            JitValue::Secp256K1Point {
//                x: (0, 0),
                x: (0, 0),
//                y: (0, 0),
                y: (0, 0),
//            },
            },
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Secp256K1Point {
            value: Box::new(JitValue::Secp256K1Point {
//                x: (0, 0),
                x: (0, 0),
//                y: (0, 0),
                y: (0, 0),
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256k1_mul",
        "secp256k1_mul",
//        &[
        &[
//            JitValue::Secp256K1Point {
            JitValue::Secp256K1Point {
//                x: (u128::MAX, 0),
                x: (u128::MAX, 0),
//                y: (0, u128::MAX),
                y: (0, u128::MAX),
//            },
            },
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Secp256K1Point {
            value: Box::new(JitValue::Secp256K1Point {
//                x: (u128::MAX, 0),
                x: (u128::MAX, 0),
//                y: (0, u128::MAX),
                y: (0, u128::MAX),
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256k1_mul",
        "secp256k1_mul",
//        &[
        &[
//            JitValue::Secp256K1Point {
            JitValue::Secp256K1Point {
//                x: (0, u128::MAX),
                x: (0, u128::MAX),
//                y: (u128::MAX, 0),
                y: (u128::MAX, 0),
//            },
            },
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Secp256K1Point {
            value: Box::new(JitValue::Secp256K1Point {
//                x: (u128::MAX, u128::MAX),
                x: (u128::MAX, u128::MAX),
//                y: (u128::MAX, u128::MAX),
                y: (u128::MAX, u128::MAX),
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    assert_eq!(
    assert_eq!(
//        syscall_handler.secp256k1_mul.0,
        syscall_handler.secp256k1_mul.0,
//        [
        [
//            (
            (
//                Secp256k1Point {
                Secp256k1Point {
//                    x: U256 { hi: 0, lo: 0 },
                    x: U256 { hi: 0, lo: 0 },
//                    y: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
//                },
                },
//                U256 { hi: 0, lo: 0 },
                U256 { hi: 0, lo: 0 },
//            ),
            ),
//            (
            (
//                Secp256k1Point {
                Secp256k1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: 0
                        lo: 0
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: 0,
                        hi: 0,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                },
                },
//                U256 {
                U256 {
//                    hi: u128::MAX,
                    hi: u128::MAX,
//                    lo: 0,
                    lo: 0,
//                },
                },
//            ),
            ),
//            (
            (
//                Secp256k1Point {
                Secp256k1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: 0,
                        hi: 0,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: 0,
                        lo: 0,
//                    },
                    },
//                },
                },
//                U256 {
                U256 {
//                    hi: 0,
                    hi: 0,
//                    lo: u128::MAX,
                    lo: u128::MAX,
//                },
                },
//            ),
            ),
//        ],
        ],
//    );
    );
//    assert!(syscall_handler.secp256k1_mul.1.is_empty());
    assert!(syscall_handler.secp256k1_mul.1.is_empty());
//}
}
//

//#[test]
#[test]
//fn secp256k1_get_point_from_x() {
fn secp256k1_get_point_from_x() {
//    let mut syscall_handler = SyscallHandler {
    let mut syscall_handler = SyscallHandler {
//        secp256k1_get_point_from_x: (
        secp256k1_get_point_from_x: (
//            VecDeque::from([]),
            VecDeque::from([]),
//            VecDeque::from([
            VecDeque::from([
//                None,
                None,
//                Some(Secp256k1Point {
                Some(Secp256k1Point {
//                    x: U256 { hi: 0, lo: 0 },
                    x: U256 { hi: 0, lo: 0 },
//                    y: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
//                }),
                }),
//                Some(Secp256k1Point {
                Some(Secp256k1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: 0,
                        hi: 0,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: 0,
                        lo: 0,
//                    },
                    },
//                }),
                }),
//                Some(Secp256k1Point {
                Some(Secp256k1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: 0,
                        lo: 0,
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: 0,
                        hi: 0,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                }),
                }),
//            ]),
            ]),
//        ),
        ),
//        ..Default::default()
        ..Default::default()
//    };
    };
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256k1_get_point_from_x",
        "secp256k1_get_point_from_x",
//        &[
        &[
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//            JitValue::Enum {
            JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Struct {
                value: Box::new(JitValue::Struct {
//                    fields: vec![],
                    fields: vec![],
//                    debug_name: None,
                    debug_name: None,
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Enum {
            value: Box::new(JitValue::Enum {
//                tag: 1,
                tag: 1,
//                value: Box::new(JitValue::Struct {
                value: Box::new(JitValue::Struct {
//                    fields: vec![],
                    fields: vec![],
//                    debug_name: None,
                    debug_name: None,
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None
            debug_name: None
//        },
        },
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256k1_get_point_from_x",
        "secp256k1_get_point_from_x",
//        &[
        &[
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//            JitValue::Enum {
            JitValue::Enum {
//                tag: 1,
                tag: 1,
//                value: Box::new(JitValue::Struct {
                value: Box::new(JitValue::Struct {
//                    fields: vec![],
                    fields: vec![],
//                    debug_name: None,
                    debug_name: None,
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Enum {
            value: Box::new(JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Secp256K1Point {
                value: Box::new(JitValue::Secp256K1Point {
//                    x: (0, 0),
                    x: (0, 0),
//                    y: (0, 0),
                    y: (0, 0),
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256k1_get_point_from_x",
        "secp256k1_get_point_from_x",
//        &[
        &[
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//            JitValue::Enum {
            JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Struct {
                value: Box::new(JitValue::Struct {
//                    fields: vec![],
                    fields: vec![],
//                    debug_name: None,
                    debug_name: None,
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Enum {
            value: Box::new(JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Secp256K1Point {
                value: Box::new(JitValue::Secp256K1Point {
//                    x: (0, u128::MAX),
                    x: (0, u128::MAX),
//                    y: (u128::MAX, 0),
                    y: (u128::MAX, 0),
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256k1_get_point_from_x",
        "secp256k1_get_point_from_x",
//        &[
        &[
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//            JitValue::Enum {
            JitValue::Enum {
//                tag: 1,
                tag: 1,
//                value: Box::new(JitValue::Struct {
                value: Box::new(JitValue::Struct {
//                    fields: vec![],
                    fields: vec![],
//                    debug_name: None,
                    debug_name: None,
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Enum {
            value: Box::new(JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Secp256K1Point {
                value: Box::new(JitValue::Secp256K1Point {
//                    x: (u128::MAX, 0),
                    x: (u128::MAX, 0),
//                    y: (0, u128::MAX),
                    y: (0, u128::MAX),
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    assert_eq!(
    assert_eq!(
//        syscall_handler.secp256k1_get_point_from_x.0,
        syscall_handler.secp256k1_get_point_from_x.0,
//        [
        [
//            (U256 { hi: 0, lo: 0 }, false),
            (U256 { hi: 0, lo: 0 }, false),
//            (
            (
//                U256 {
                U256 {
//                    hi: 0,
                    hi: 0,
//                    lo: u128::MAX,
                    lo: u128::MAX,
//                },
                },
//                true,
                true,
//            ),
            ),
//            (
            (
//                U256 {
                U256 {
//                    hi: u128::MAX,
                    hi: u128::MAX,
//                    lo: 0,
                    lo: 0,
//                },
                },
//                false,
                false,
//            ),
            ),
//            (
            (
//                U256 {
                U256 {
//                    hi: u128::MAX,
                    hi: u128::MAX,
//                    lo: u128::MAX,
                    lo: u128::MAX,
//                },
                },
//                true,
                true,
//            ),
            ),
//        ],
        ],
//    );
    );
//    assert!(syscall_handler.secp256k1_get_point_from_x.1.is_empty());
    assert!(syscall_handler.secp256k1_get_point_from_x.1.is_empty());
//}
}
//

//#[test]
#[test]
//fn secp256k1_get_xy() {
fn secp256k1_get_xy() {
//    let mut syscall_handler = SyscallHandler {
    let mut syscall_handler = SyscallHandler {
//        secp256k1_get_xy: (
        secp256k1_get_xy: (
//            VecDeque::from([]),
            VecDeque::from([]),
//            VecDeque::from([
            VecDeque::from([
//                (U256 { hi: 0, lo: 0 }, U256 { hi: 0, lo: 0 }),
                (U256 { hi: 0, lo: 0 }, U256 { hi: 0, lo: 0 }),
//                (
                (
//                    U256 {
                    U256 {
//                        hi: 0,
                        hi: 0,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                    U256 {
                    U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: 0,
                        lo: 0,
//                    },
                    },
//                ),
                ),
//                (
                (
//                    U256 {
                    U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: 0,
                        lo: 0,
//                    },
                    },
//                    U256 {
                    U256 {
//                        hi: 0,
                        hi: 0,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                ),
                ),
//                (
                (
//                    U256 {
                    U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                    U256 {
                    U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                ),
                ),
//            ]),
            ]),
//        ),
        ),
//        ..Default::default()
        ..Default::default()
//    };
    };
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256k1_get_xy",
        "secp256k1_get_xy",
//        &[JitValue::Secp256K1Point {
        &[JitValue::Secp256K1Point {
//            x: (0, 0),
            x: (0, 0),
//            y: (0, 0),
            y: (0, 0),
//        }],
        }],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Struct {
            value: Box::new(JitValue::Struct {
//                fields: vec![
                fields: vec![
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                ],
                ],
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        }
        }
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256k1_get_xy",
        "secp256k1_get_xy",
//        &[JitValue::Secp256K1Point {
        &[JitValue::Secp256K1Point {
//            x: (0, u128::MAX),
            x: (0, u128::MAX),
//            y: (u128::MAX, 0),
            y: (u128::MAX, 0),
//        }],
        }],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Struct {
            value: Box::new(JitValue::Struct {
//                fields: vec![
                fields: vec![
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                ],
                ],
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        }
        }
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256k1_get_xy",
        "secp256k1_get_xy",
//        &[JitValue::Secp256K1Point {
        &[JitValue::Secp256K1Point {
//            x: (u128::MAX, 0),
            x: (u128::MAX, 0),
//            y: (0, u128::MAX),
            y: (0, u128::MAX),
//        }],
        }],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Struct {
            value: Box::new(JitValue::Struct {
//                fields: vec![
                fields: vec![
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                ],
                ],
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        }
        }
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256k1_get_xy",
        "secp256k1_get_xy",
//        &[JitValue::Secp256K1Point {
        &[JitValue::Secp256K1Point {
//            x: (u128::MAX, u128::MAX),
            x: (u128::MAX, u128::MAX),
//            y: (u128::MAX, u128::MAX),
            y: (u128::MAX, u128::MAX),
//        }],
        }],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Struct {
            value: Box::new(JitValue::Struct {
//                fields: vec![
                fields: vec![
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                ],
                ],
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        }
        }
//    );
    );
//

//    assert_eq!(
    assert_eq!(
//        syscall_handler.secp256k1_get_xy.0,
        syscall_handler.secp256k1_get_xy.0,
//        [
        [
//            Secp256k1Point {
            Secp256k1Point {
//                x: U256 { hi: 0, lo: 0 },
                x: U256 { hi: 0, lo: 0 },
//                y: U256 { hi: 0, lo: 0 },
                y: U256 { hi: 0, lo: 0 },
//            },
            },
//            Secp256k1Point {
            Secp256k1Point {
//                x: U256 {
                x: U256 {
//                    hi: 0,
                    hi: 0,
//                    lo: u128::MAX,
                    lo: u128::MAX,
//                },
                },
//                y: U256 {
                y: U256 {
//                    hi: u128::MAX,
                    hi: u128::MAX,
//                    lo: 0,
                    lo: 0,
//                },
                },
//            },
            },
//            Secp256k1Point {
            Secp256k1Point {
//                x: U256 {
                x: U256 {
//                    hi: u128::MAX,
                    hi: u128::MAX,
//                    lo: 0,
                    lo: 0,
//                },
                },
//                y: U256 {
                y: U256 {
//                    hi: 0,
                    hi: 0,
//                    lo: u128::MAX,
                    lo: u128::MAX,
//                },
                },
//            },
            },
//            Secp256k1Point {
            Secp256k1Point {
//                x: U256 {
                x: U256 {
//                    hi: u128::MAX,
                    hi: u128::MAX,
//                    lo: u128::MAX,
                    lo: u128::MAX,
//                },
                },
//                y: U256 {
                y: U256 {
//                    hi: u128::MAX,
                    hi: u128::MAX,
//                    lo: u128::MAX,
                    lo: u128::MAX,
//                },
                },
//            },
            },
//        ],
        ],
//    );
    );
//    assert!(syscall_handler.secp256k1_get_xy.1.is_empty());
    assert!(syscall_handler.secp256k1_get_xy.1.is_empty());
//}
}
//

//#[test]
#[test]
//fn secp256r1_new() {
fn secp256r1_new() {
//    let mut syscall_handler = SyscallHandler {
    let mut syscall_handler = SyscallHandler {
//        secp256r1_new: (
        secp256r1_new: (
//            VecDeque::from([]),
            VecDeque::from([]),
//            VecDeque::from([
            VecDeque::from([
//                None,
                None,
//                Some(Secp256r1Point {
                Some(Secp256r1Point {
//                    x: U256 { hi: 0, lo: 0 },
                    x: U256 { hi: 0, lo: 0 },
//                    y: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
//                }),
                }),
//                Some(Secp256r1Point {
                Some(Secp256r1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                }),
                }),
//            ]),
            ]),
//        ),
        ),
//        ..Default::default()
        ..Default::default()
//    };
    };
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256r1_new",
        "secp256r1_new",
//        &[
        &[
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Enum {
            value: Box::new(JitValue::Enum {
//                tag: 1,
                tag: 1,
//                value: Box::new(JitValue::Struct {
                value: Box::new(JitValue::Struct {
//                    fields: vec![],
                    fields: vec![],
//                    debug_name: None
                    debug_name: None
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256r1_new",
        "secp256r1_new",
//        &[
        &[
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Enum {
            value: Box::new(JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Secp256R1Point {
                value: Box::new(JitValue::Secp256R1Point {
//                    x: (0, 0),
                    x: (0, 0),
//                    y: (0, 0),
                    y: (0, 0),
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256r1_new",
        "secp256r1_new",
//        &[
        &[
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Enum {
            value: Box::new(JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Secp256R1Point {
                value: Box::new(JitValue::Secp256R1Point {
//                    x: (u128::MAX, u128::MAX),
                    x: (u128::MAX, u128::MAX),
//                    y: (u128::MAX, u128::MAX),
                    y: (u128::MAX, u128::MAX),
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    assert_eq!(
    assert_eq!(
//        syscall_handler.secp256r1_new.0,
        syscall_handler.secp256r1_new.0,
//        [
        [
//            (U256 { hi: 0, lo: 0 }, U256 { hi: 0, lo: 0 }),
            (U256 { hi: 0, lo: 0 }, U256 { hi: 0, lo: 0 }),
//            (
            (
//                U256 {
                U256 {
//                    hi: u128::MAX,
                    hi: u128::MAX,
//                    lo: 0
                    lo: 0
//                },
                },
//                U256 {
                U256 {
//                    hi: 0,
                    hi: 0,
//                    lo: u128::MAX
                    lo: u128::MAX
//                }
                }
//            ),
            ),
//            (
            (
//                U256 {
                U256 {
//                    hi: u128::MAX,
                    hi: u128::MAX,
//                    lo: u128::MAX
                    lo: u128::MAX
//                },
                },
//                U256 {
                U256 {
//                    hi: u128::MAX,
                    hi: u128::MAX,
//                    lo: u128::MAX
                    lo: u128::MAX
//                }
                }
//            ),
            ),
//        ],
        ],
//    );
    );
//    assert!(syscall_handler.secp256r1_new.1.is_empty());
    assert!(syscall_handler.secp256r1_new.1.is_empty());
//}
}
//

//#[test]
#[test]
//fn secp256r1_add() {
fn secp256r1_add() {
//    let mut syscall_handler = SyscallHandler {
    let mut syscall_handler = SyscallHandler {
//        secp256r1_add: (
        secp256r1_add: (
//            VecDeque::from([]),
            VecDeque::from([]),
//            VecDeque::from([
            VecDeque::from([
//                Secp256r1Point {
                Secp256r1Point {
//                    x: U256 { hi: 0, lo: 0 },
                    x: U256 { hi: 0, lo: 0 },
//                    y: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
//                },
                },
//                Secp256r1Point {
                Secp256r1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: 0,
                        lo: 0,
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: 0,
                        hi: 0,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                },
                },
//                Secp256r1Point {
                Secp256r1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                },
                },
//            ]),
            ]),
//        ),
        ),
//        ..Default::default()
        ..Default::default()
//    };
    };
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256r1_add",
        "secp256r1_add",
//        &[
        &[
//            JitValue::Secp256R1Point {
            JitValue::Secp256R1Point {
//                x: (0, 0),
                x: (0, 0),
//                y: (0, 0),
                y: (0, 0),
//            },
            },
//            JitValue::Secp256R1Point {
            JitValue::Secp256R1Point {
//                x: (0, 0),
                x: (0, 0),
//                y: (0, 0),
                y: (0, 0),
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Secp256R1Point {
            value: Box::new(JitValue::Secp256R1Point {
//                x: (0, 0),
                x: (0, 0),
//                y: (0, 0),
                y: (0, 0),
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256r1_add",
        "secp256r1_add",
//        &[
        &[
//            JitValue::Secp256R1Point {
            JitValue::Secp256R1Point {
//                x: (u128::MAX, 0),
                x: (u128::MAX, 0),
//                y: (0, u128::MAX),
                y: (0, u128::MAX),
//            },
            },
//            JitValue::Secp256R1Point {
            JitValue::Secp256R1Point {
//                x: (0, u128::MAX),
                x: (0, u128::MAX),
//                y: (u128::MAX, 0),
                y: (u128::MAX, 0),
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Secp256R1Point {
            value: Box::new(JitValue::Secp256R1Point {
//                x: (u128::MAX, 0),
                x: (u128::MAX, 0),
//                y: (0, u128::MAX),
                y: (0, u128::MAX),
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256r1_add",
        "secp256r1_add",
//        &[
        &[
//            JitValue::Secp256R1Point {
            JitValue::Secp256R1Point {
//                x: (u128::MAX, u128::MAX),
                x: (u128::MAX, u128::MAX),
//                y: (u128::MAX, u128::MAX),
                y: (u128::MAX, u128::MAX),
//            },
            },
//            JitValue::Secp256R1Point {
            JitValue::Secp256R1Point {
//                x: (u128::MAX, u128::MAX),
                x: (u128::MAX, u128::MAX),
//                y: (u128::MAX, u128::MAX),
                y: (u128::MAX, u128::MAX),
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Secp256R1Point {
            value: Box::new(JitValue::Secp256R1Point {
//                x: (u128::MAX, u128::MAX),
                x: (u128::MAX, u128::MAX),
//                y: (u128::MAX, u128::MAX),
                y: (u128::MAX, u128::MAX),
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    assert_eq!(
    assert_eq!(
//        syscall_handler.secp256r1_add.0,
        syscall_handler.secp256r1_add.0,
//        [
        [
//            (
            (
//                Secp256r1Point {
                Secp256r1Point {
//                    x: U256 { hi: 0, lo: 0 },
                    x: U256 { hi: 0, lo: 0 },
//                    y: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
//                },
                },
//                Secp256r1Point {
                Secp256r1Point {
//                    x: U256 { hi: 0, lo: 0 },
                    x: U256 { hi: 0, lo: 0 },
//                    y: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
//                },
                },
//            ),
            ),
//            (
            (
//                Secp256r1Point {
                Secp256r1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: 0
                        lo: 0
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: 0,
                        hi: 0,
//                        lo: u128::MAX
                        lo: u128::MAX
//                    },
                    },
//                },
                },
//                Secp256r1Point {
                Secp256r1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: 0,
                        hi: 0,
//                        lo: u128::MAX
                        lo: u128::MAX
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: 0
                        lo: 0
//                    },
                    },
//                },
                },
//            ),
            ),
//            (
            (
//                Secp256r1Point {
                Secp256r1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX
                        lo: u128::MAX
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX
                        lo: u128::MAX
//                    },
                    },
//                },
                },
//                Secp256r1Point {
                Secp256r1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX
                        lo: u128::MAX
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX
                        lo: u128::MAX
//                    },
                    },
//                },
                },
//            ),
            ),
//        ],
        ],
//    );
    );
//    assert!(syscall_handler.secp256r1_add.1.is_empty());
    assert!(syscall_handler.secp256r1_add.1.is_empty());
//}
}
//

//#[test]
#[test]
//fn secp256r1_mul() {
fn secp256r1_mul() {
//    let mut syscall_handler = SyscallHandler {
    let mut syscall_handler = SyscallHandler {
//        secp256r1_mul: (
        secp256r1_mul: (
//            VecDeque::from([]),
            VecDeque::from([]),
//            VecDeque::from([
            VecDeque::from([
//                Secp256r1Point {
                Secp256r1Point {
//                    x: U256 { hi: 0, lo: 0 },
                    x: U256 { hi: 0, lo: 0 },
//                    y: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
//                },
                },
//                Secp256r1Point {
                Secp256r1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: 0,
                        lo: 0,
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: 0,
                        hi: 0,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                },
                },
//                Secp256r1Point {
                Secp256r1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                },
                },
//            ]),
            ]),
//        ),
        ),
//        ..Default::default()
        ..Default::default()
//    };
    };
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256r1_mul",
        "secp256r1_mul",
//        &[
        &[
//            JitValue::Secp256R1Point {
            JitValue::Secp256R1Point {
//                x: (0, 0),
                x: (0, 0),
//                y: (0, 0),
                y: (0, 0),
//            },
            },
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Secp256R1Point {
            value: Box::new(JitValue::Secp256R1Point {
//                x: (0, 0),
                x: (0, 0),
//                y: (0, 0),
                y: (0, 0),
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256r1_mul",
        "secp256r1_mul",
//        &[
        &[
//            JitValue::Secp256R1Point {
            JitValue::Secp256R1Point {
//                x: (u128::MAX, 0),
                x: (u128::MAX, 0),
//                y: (0, u128::MAX),
                y: (0, u128::MAX),
//            },
            },
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Secp256R1Point {
            value: Box::new(JitValue::Secp256R1Point {
//                x: (u128::MAX, 0),
                x: (u128::MAX, 0),
//                y: (0, u128::MAX),
                y: (0, u128::MAX),
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256r1_mul",
        "secp256r1_mul",
//        &[
        &[
//            JitValue::Secp256R1Point {
            JitValue::Secp256R1Point {
//                x: (0, u128::MAX),
                x: (0, u128::MAX),
//                y: (u128::MAX, 0),
                y: (u128::MAX, 0),
//            },
            },
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Secp256R1Point {
            value: Box::new(JitValue::Secp256R1Point {
//                x: (u128::MAX, u128::MAX),
                x: (u128::MAX, u128::MAX),
//                y: (u128::MAX, u128::MAX),
                y: (u128::MAX, u128::MAX),
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    assert_eq!(
    assert_eq!(
//        syscall_handler.secp256r1_mul.0,
        syscall_handler.secp256r1_mul.0,
//        [
        [
//            (
            (
//                Secp256r1Point {
                Secp256r1Point {
//                    x: U256 { hi: 0, lo: 0 },
                    x: U256 { hi: 0, lo: 0 },
//                    y: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
//                },
                },
//                U256 { hi: 0, lo: 0 },
                U256 { hi: 0, lo: 0 },
//            ),
            ),
//            (
            (
//                Secp256r1Point {
                Secp256r1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: 0
                        lo: 0
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: 0,
                        hi: 0,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                },
                },
//                U256 {
                U256 {
//                    hi: u128::MAX,
                    hi: u128::MAX,
//                    lo: 0,
                    lo: 0,
//                },
                },
//            ),
            ),
//            (
            (
//                Secp256r1Point {
                Secp256r1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: 0,
                        hi: 0,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: 0,
                        lo: 0,
//                    },
                    },
//                },
                },
//                U256 {
                U256 {
//                    hi: 0,
                    hi: 0,
//                    lo: u128::MAX,
                    lo: u128::MAX,
//                },
                },
//            ),
            ),
//        ],
        ],
//    );
    );
//    assert!(syscall_handler.secp256r1_mul.1.is_empty());
    assert!(syscall_handler.secp256r1_mul.1.is_empty());
//}
}
//

//#[test]
#[test]
//fn secp256r1_get_point_from_x() {
fn secp256r1_get_point_from_x() {
//    let mut syscall_handler = SyscallHandler {
    let mut syscall_handler = SyscallHandler {
//        secp256r1_get_point_from_x: (
        secp256r1_get_point_from_x: (
//            VecDeque::from([]),
            VecDeque::from([]),
//            VecDeque::from([
            VecDeque::from([
//                None,
                None,
//                Some(Secp256r1Point {
                Some(Secp256r1Point {
//                    x: U256 { hi: 0, lo: 0 },
                    x: U256 { hi: 0, lo: 0 },
//                    y: U256 { hi: 0, lo: 0 },
                    y: U256 { hi: 0, lo: 0 },
//                }),
                }),
//                Some(Secp256r1Point {
                Some(Secp256r1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: 0,
                        hi: 0,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: 0,
                        lo: 0,
//                    },
                    },
//                }),
                }),
//                Some(Secp256r1Point {
                Some(Secp256r1Point {
//                    x: U256 {
                    x: U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: 0,
                        lo: 0,
//                    },
                    },
//                    y: U256 {
                    y: U256 {
//                        hi: 0,
                        hi: 0,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                }),
                }),
//            ]),
            ]),
//        ),
        ),
//        ..Default::default()
        ..Default::default()
//    };
    };
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256r1_get_point_from_x",
        "secp256r1_get_point_from_x",
//        &[
        &[
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//            JitValue::Enum {
            JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Struct {
                value: Box::new(JitValue::Struct {
//                    fields: vec![],
                    fields: vec![],
//                    debug_name: None,
                    debug_name: None,
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Enum {
            value: Box::new(JitValue::Enum {
//                tag: 1,
                tag: 1,
//                value: Box::new(JitValue::Struct {
                value: Box::new(JitValue::Struct {
//                    fields: vec![],
                    fields: vec![],
//                    debug_name: None,
                    debug_name: None,
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None
            debug_name: None
//        },
        },
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256r1_get_point_from_x",
        "secp256r1_get_point_from_x",
//        &[
        &[
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
                fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//            JitValue::Enum {
            JitValue::Enum {
//                tag: 1,
                tag: 1,
//                value: Box::new(JitValue::Struct {
                value: Box::new(JitValue::Struct {
//                    fields: vec![],
                    fields: vec![],
//                    debug_name: None,
                    debug_name: None,
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Enum {
            value: Box::new(JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Secp256R1Point {
                value: Box::new(JitValue::Secp256R1Point {
//                    x: (0, 0),
                    x: (0, 0),
//                    y: (0, 0),
                    y: (0, 0),
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256r1_get_point_from_x",
        "secp256r1_get_point_from_x",
//        &[
        &[
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//            JitValue::Enum {
            JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Struct {
                value: Box::new(JitValue::Struct {
//                    fields: vec![],
                    fields: vec![],
//                    debug_name: None,
                    debug_name: None,
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Enum {
            value: Box::new(JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Secp256R1Point {
                value: Box::new(JitValue::Secp256R1Point {
//                    x: (0, u128::MAX),
                    x: (0, u128::MAX),
//                    y: (u128::MAX, 0),
                    y: (u128::MAX, 0),
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256r1_get_point_from_x",
        "secp256r1_get_point_from_x",
//        &[
        &[
//            JitValue::Struct {
            JitValue::Struct {
//                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
                fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
//                debug_name: None,
                debug_name: None,
//            },
            },
//            JitValue::Enum {
            JitValue::Enum {
//                tag: 1,
                tag: 1,
//                value: Box::new(JitValue::Struct {
                value: Box::new(JitValue::Struct {
//                    fields: vec![],
                    fields: vec![],
//                    debug_name: None,
                    debug_name: None,
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        ],
        ],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Enum {
            value: Box::new(JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Secp256R1Point {
                value: Box::new(JitValue::Secp256R1Point {
//                    x: (u128::MAX, 0),
                    x: (u128::MAX, 0),
//                    y: (0, u128::MAX),
                    y: (0, u128::MAX),
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//

//    assert_eq!(
    assert_eq!(
//        syscall_handler.secp256r1_get_point_from_x.0,
        syscall_handler.secp256r1_get_point_from_x.0,
//        [
        [
//            (U256 { hi: 0, lo: 0 }, false),
            (U256 { hi: 0, lo: 0 }, false),
//            (
            (
//                U256 {
                U256 {
//                    hi: 0,
                    hi: 0,
//                    lo: u128::MAX,
                    lo: u128::MAX,
//                },
                },
//                true,
                true,
//            ),
            ),
//            (
            (
//                U256 {
                U256 {
//                    hi: u128::MAX,
                    hi: u128::MAX,
//                    lo: 0,
                    lo: 0,
//                },
                },
//                false,
                false,
//            ),
            ),
//            (
            (
//                U256 {
                U256 {
//                    hi: u128::MAX,
                    hi: u128::MAX,
//                    lo: u128::MAX,
                    lo: u128::MAX,
//                },
                },
//                true,
                true,
//            ),
            ),
//        ],
        ],
//    );
    );
//    assert!(syscall_handler.secp256r1_get_point_from_x.1.is_empty());
    assert!(syscall_handler.secp256r1_get_point_from_x.1.is_empty());
//}
}
//

//#[test]
#[test]
//fn secp256r1_get_xy() {
fn secp256r1_get_xy() {
//    let mut syscall_handler = SyscallHandler {
    let mut syscall_handler = SyscallHandler {
//        secp256r1_get_xy: (
        secp256r1_get_xy: (
//            VecDeque::from([]),
            VecDeque::from([]),
//            VecDeque::from([
            VecDeque::from([
//                (U256 { hi: 0, lo: 0 }, U256 { hi: 0, lo: 0 }),
                (U256 { hi: 0, lo: 0 }, U256 { hi: 0, lo: 0 }),
//                (
                (
//                    U256 {
                    U256 {
//                        hi: 0,
                        hi: 0,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                    U256 {
                    U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: 0,
                        lo: 0,
//                    },
                    },
//                ),
                ),
//                (
                (
//                    U256 {
                    U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: 0,
                        lo: 0,
//                    },
                    },
//                    U256 {
                    U256 {
//                        hi: 0,
                        hi: 0,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                ),
                ),
//                (
                (
//                    U256 {
                    U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                    U256 {
                    U256 {
//                        hi: u128::MAX,
                        hi: u128::MAX,
//                        lo: u128::MAX,
                        lo: u128::MAX,
//                    },
                    },
//                ),
                ),
//            ]),
            ]),
//        ),
        ),
//        ..Default::default()
        ..Default::default()
//    };
    };
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256r1_get_xy",
        "secp256r1_get_xy",
//        &[JitValue::Secp256R1Point {
        &[JitValue::Secp256R1Point {
//            x: (0, 0),
            x: (0, 0),
//            y: (0, 0),
            y: (0, 0),
//        }],
        }],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Struct {
            value: Box::new(JitValue::Struct {
//                fields: vec![
                fields: vec![
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(0)],
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                ],
                ],
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        }
        }
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256r1_get_xy",
        "secp256r1_get_xy",
//        &[JitValue::Secp256R1Point {
        &[JitValue::Secp256R1Point {
//            x: (0, u128::MAX),
            x: (0, u128::MAX),
//            y: (u128::MAX, 0),
            y: (u128::MAX, 0),
//        }],
        }],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Struct {
            value: Box::new(JitValue::Struct {
//                fields: vec![
                fields: vec![
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                ],
                ],
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        }
        }
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256r1_get_xy",
        "secp256r1_get_xy",
//        &[JitValue::Secp256R1Point {
        &[JitValue::Secp256R1Point {
//            x: (u128::MAX, 0),
            x: (u128::MAX, 0),
//            y: (0, u128::MAX),
            y: (0, u128::MAX),
//        }],
        }],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Struct {
            value: Box::new(JitValue::Struct {
//                fields: vec![
                fields: vec![
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(0)],
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
                        fields: vec![JitValue::Uint128(0), JitValue::Uint128(u128::MAX)],
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                ],
                ],
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        }
        }
//    );
    );
//

//    let result = run_native_program(
    let result = run_native_program(
//        &SECP256_PROGRAM,
        &SECP256_PROGRAM,
//        "secp256r1_get_xy",
        "secp256r1_get_xy",
//        &[JitValue::Secp256R1Point {
        &[JitValue::Secp256R1Point {
//            x: (u128::MAX, u128::MAX),
            x: (u128::MAX, u128::MAX),
//            y: (u128::MAX, u128::MAX),
            y: (u128::MAX, u128::MAX),
//        }],
        }],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(&mut syscall_handler),
        Some(&mut syscall_handler),
//    );
    );
//    assert_eq!(
    assert_eq!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Struct {
            value: Box::new(JitValue::Struct {
//                fields: vec![
                fields: vec![
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
                        fields: vec![JitValue::Uint128(u128::MAX), JitValue::Uint128(u128::MAX)],
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                ],
                ],
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        }
        }
//    );
    );
//

//    assert_eq!(
    assert_eq!(
//        syscall_handler.secp256r1_get_xy.0,
        syscall_handler.secp256r1_get_xy.0,
//        [
        [
//            Secp256r1Point {
            Secp256r1Point {
//                x: U256 { hi: 0, lo: 0 },
                x: U256 { hi: 0, lo: 0 },
//                y: U256 { hi: 0, lo: 0 },
                y: U256 { hi: 0, lo: 0 },
//            },
            },
//            Secp256r1Point {
            Secp256r1Point {
//                x: U256 {
                x: U256 {
//                    hi: 0,
                    hi: 0,
//                    lo: u128::MAX,
                    lo: u128::MAX,
//                },
                },
//                y: U256 {
                y: U256 {
//                    hi: u128::MAX,
                    hi: u128::MAX,
//                    lo: 0,
                    lo: 0,
//                },
                },
//            },
            },
//            Secp256r1Point {
            Secp256r1Point {
//                x: U256 {
                x: U256 {
//                    hi: u128::MAX,
                    hi: u128::MAX,
//                    lo: 0,
                    lo: 0,
//                },
                },
//                y: U256 {
                y: U256 {
//                    hi: 0,
                    hi: 0,
//                    lo: u128::MAX,
                    lo: u128::MAX,
//                },
                },
//            },
            },
//            Secp256r1Point {
            Secp256r1Point {
//                x: U256 {
                x: U256 {
//                    hi: u128::MAX,
                    hi: u128::MAX,
//                    lo: u128::MAX,
                    lo: u128::MAX,
//                },
                },
//                y: U256 {
                y: U256 {
//                    hi: u128::MAX,
                    hi: u128::MAX,
//                    lo: u128::MAX,
                    lo: u128::MAX,
//                },
                },
//            },
            },
//        ],
        ],
//    );
    );
//    assert!(syscall_handler.secp256r1_get_xy.1.is_empty());
    assert!(syscall_handler.secp256r1_get_xy.1.is_empty());
//}
}
