use crate::common::{load_cairo_path, run_native_program};
use cairo_lang_runner::SierraCasmRunner;
use cairo_lang_sierra::program::Program;
use cairo_native::{
    starknet::{
        BlockInfo, ExecutionInfo, ExecutionInfoV2, Secp256k1Point, Secp256r1Point,
        StarknetSyscallHandler, SyscallResult, TxInfo, TxV2Info, U256,
    },
    values::JitValue,
};
use lazy_static::lazy_static;
use pretty_assertions_sorted::assert_eq_sorted;
use starknet_types_core::felt::Felt;

#[derive(Debug, Default)]
#[allow(dead_code)] // TODO(julian): implement testing syscall
struct TestingState {
    sequencer_address: Felt,
    block_number: u64,
    block_timestamp: u64,
    caller_address: Felt,
    contract_address: Felt,
    version: Felt,
    account_contract_address: Felt,
    max_fee: u64,
    transaction_hash: Felt,
    chain_id: Felt,
    nonce: Felt,
    signature: Vec<Felt>,
    logs: Vec<(Vec<Felt>, Vec<Felt>)>,
}

struct SyscallHandler {
    #[allow(dead_code)] // TODO(julian): implement testing syscall
    testing_state: TestingState,
}

impl SyscallHandler {
    fn new() -> Self {
        Self {
            testing_state: TestingState::default(),
        }
    }
}

impl StarknetSyscallHandler for SyscallHandler {
    fn get_block_hash(
        &mut self,
        _block_number: u64,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Felt> {
        Ok(Felt::from_dec_str(
            "1158579293198495875788224011889333769139150068959598053296510642728083832673",
        )
        .unwrap())
    }

    fn get_execution_info(&mut self, _remaining_gas: &mut u128) -> SyscallResult<ExecutionInfo> {
        Ok(ExecutionInfo {
            block_info: BlockInfo {
                block_number: 10057862467973663535,
                block_timestamp: 13878668747512495966,
                sequencer_address: Felt::from_dec_str(
                    "1126241460712630201003776917997524449163698107789103849210792326381258973685",
                )
                .unwrap(),
            },
            tx_info: TxInfo {
                version: Felt::from_dec_str(
                    "1724985403142256920476849371638528834056988111202434162793214195754735917407",
                )
                .unwrap(),
                account_contract_address: Felt::from_dec_str(
                    "2419272378964094005143278046496347854926114240785059742454535261490265649110",
                )
                .unwrap(),
                max_fee: 67871905340377755668863509019681938001,
                signature: Vec::new(),
                transaction_hash: Felt::from_dec_str(
                    "2073267424102447009330753642820908998776456851902897601865334818765025369132",
                )
                .unwrap(),
                chain_id: Felt::from_dec_str(
                    "1727570805086347994328356733148206517040691113666039929118050093237140484117",
                )
                .unwrap(),
                nonce: Felt::from_dec_str(
                    "2223335940097352947792108259394621577330089800429182023415494612506457867705",
                )
                .unwrap(),
            },
            caller_address: Felt::from_dec_str(
                "2367044879643293830108311482898145302930693201376043522909298679498599559539",
            )
            .unwrap(),
            contract_address: Felt::from_dec_str(
                "2322490563038631685097154208793293355074547843057070254216662565231428808211",
            )
            .unwrap(),
            entry_point_selector: Felt::from_dec_str(
                "1501296828847480842982002010206952982741090100977850506550982801410247026532",
            )
            .unwrap(),
        })
    }

    fn get_execution_info_v2(
        &mut self,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<ExecutionInfoV2> {
        Ok(ExecutionInfoV2 {
            block_info: BlockInfo {
                block_number: 10290342497028289173,
                block_timestamp: 8376161426686560326,
                sequencer_address: Felt::from_dec_str(
                    "1815189516202718271265591469295511271015058493881778555617445818147186579905",
                )
                .unwrap(),
            },
            tx_info: TxV2Info {
                version: Felt::from_dec_str(
                    "1946630339019864531118751968563861838541265142438690346764722398811248737786",
                )
                .unwrap(),
                account_contract_address: Felt::from_dec_str(
                    "2501333093425095943815772537228190103182643237630648877273495185321298605376",
                )
                .unwrap(),
                max_fee: 268753657614351187400966367706860329387,
                signature: Vec::new(),
                transaction_hash: Felt::from_dec_str(
                    "1123336726531770778820945049824733201592457249587063926479184903627272350002",
                )
                .unwrap(),
                chain_id: Felt::from_dec_str(
                    "2128916697180095451339935431635121484141376377516602728602049361615810538124",
                )
                .unwrap(),
                nonce: Felt::from_dec_str(
                    "3012936192361023209451741736298028332652992971202997279327088951248532774884",
                )
                .unwrap(),
                resource_bounds: Vec::new(),
                tip: 215444579144685671333997376989135077200,
                paymaster_data: Vec::new(),
                nonce_data_availability_mode: 140600095,
                fee_data_availability_mode: 988370659,
                account_deployment_data: Vec::new(),
            },
            caller_address: Felt::from_dec_str(
                "1185632056775552928459345712365014492063999606476424661067102766803470217687",
            )
            .unwrap(),
            contract_address: Felt::from_dec_str(
                "741063429140548584082645215539704615048011618665759826371923004739480130327",
            )
            .unwrap(),
            entry_point_selector: Felt::from_dec_str(
                "477501848519111015718660527024172361930966806556174677443839145770405114061",
            )
            .unwrap(),
        })
    }

    fn deploy(
        &mut self,
        _class_hash: Felt,
        _contract_address_salt: Felt,
        _calldata: &[Felt],
        _deploy_from_zero: bool,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<(Felt, Vec<Felt>)> {
        Ok((
            Felt::from_dec_str(
                "1833707083418045616336697070784512826809940908236872124572250196391719980392",
            )
            .unwrap(),
            Vec::new(),
        ))
    }

    fn replace_class(&mut self, _class_hash: Felt, _remaining_gas: &mut u128) -> SyscallResult<()> {
        Ok(())
    }

    fn library_call(
        &mut self,
        _class_hash: Felt,
        _function_selector: Felt,
        _calldata: &[Felt],
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Vec<Felt>> {
        Ok(vec![
            Felt::from_dec_str(
                "3358892263739032253767642605669710712087178958719188919195252597609334880396",
            )
            .unwrap(),
            Felt::from_dec_str(
                "1104291043781086177955655234103730593173963850634630109574183288837411031513",
            )
            .unwrap(),
            Felt::from_dec_str(
                "3346377229881115874907650557159666001431249650068516742483979624047277128413",
            )
            .unwrap(),
        ])
    }

    fn call_contract(
        &mut self,
        _address: Felt,
        _entry_point_selector: Felt,
        _calldata: &[Felt],
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Vec<Felt>> {
        Ok(vec![
            Felt::from_dec_str(
                "3358892263739032253767642605669710712087178958719188919195252597609334880396",
            )
            .unwrap(),
            Felt::from_dec_str(
                "1104291043781086177955655234103730593173963850634630109574183288837411031513",
            )
            .unwrap(),
            Felt::from_dec_str(
                "3346377229881115874907650557159666001431249650068516742483979624047277128413",
            )
            .unwrap(),
        ])
    }

    fn storage_read(
        &mut self,
        _address_domain: u32,
        _address: Felt,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Felt> {
        Ok(Felt::from_dec_str(
            "1013181629378419652272218169322268188846114273878719855200100663863924329981",
        )
        .unwrap())
    }

    fn storage_write(
        &mut self,
        _address_domain: u32,
        _address: Felt,
        _value: Felt,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<()> {
        Ok(())
    }

    fn emit_event(
        &mut self,
        _keys: &[Felt],
        _data: &[Felt],
        _remaining_gas: &mut u128,
    ) -> SyscallResult<()> {
        Ok(())
    }

    fn send_message_to_l1(
        &mut self,
        to_address: Felt,
        payload: &[Felt],
        _remaining_gas: &mut u128,
    ) -> SyscallResult<()> {
        assert_eq!(
            to_address,
            3.into(),
            "send_message_to_l1 to_address mismatch"
        );
        assert_eq!(payload, &[2.into()], "send_message_to_l1 payload mismatch");
        Ok(())
    }

    fn keccak(&mut self, _input: &[u64], _remaining_gas: &mut u128) -> SyscallResult<U256> {
        Ok(U256 {
            hi: 330939983442938156232262046592599923289,
            lo: 288102973244655531496349286021939642254,
        })
    }

    fn secp256k1_new(
        &mut self,
        _x: U256,
        _y: U256,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256k1Point>> {
        // Tested in `tests/tests/starknet/secp256.rs`.
        unimplemented!()
    }

    fn secp256k1_add(
        &mut self,
        _p0: Secp256k1Point,
        _p1: Secp256k1Point,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Secp256k1Point> {
        // Tested in `tests/tests/starknet/secp256.rs`.
        unimplemented!()
    }

    fn secp256k1_mul(
        &mut self,
        _p: Secp256k1Point,
        _m: U256,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Secp256k1Point> {
        // Tested in `tests/tests/starknet/secp256.rs`.
        unimplemented!()
    }

    fn secp256k1_get_point_from_x(
        &mut self,
        _x: U256,
        _y_parity: bool,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256k1Point>> {
        // Tested in `tests/tests/starknet/secp256.rs`.
        unimplemented!()
    }

    fn secp256k1_get_xy(
        &mut self,
        _p: Secp256k1Point,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<(U256, U256)> {
        // Tested in `tests/tests/starknet/secp256.rs`.
        unimplemented!()
    }

    fn secp256r1_new(
        &mut self,
        _x: U256,
        _y: U256,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256r1Point>> {
        // Tested in `tests/tests/starknet/secp256.rs`.
        unimplemented!()
    }

    fn secp256r1_add(
        &mut self,
        _p0: Secp256r1Point,
        _p1: Secp256r1Point,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Secp256r1Point> {
        // Tested in `tests/tests/starknet/secp256.rs`.
        unimplemented!()
    }

    fn secp256r1_mul(
        &mut self,
        _p: Secp256r1Point,
        _m: U256,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Secp256r1Point> {
        // Tested in `tests/tests/starknet/secp256.rs`.
        unimplemented!()
    }

    fn secp256r1_get_point_from_x(
        &mut self,
        _x: U256,
        _y_parity: bool,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256r1Point>> {
        // Tested in `tests/tests/starknet/secp256.rs`.
        unimplemented!()
    }

    fn secp256r1_get_xy(
        &mut self,
        _p: Secp256r1Point,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<(U256, U256)> {
        // Tested in `tests/tests/starknet/secp256.rs`.
        unimplemented!()
    }

    fn cheatcode(&mut self, _input: &[Felt]) -> SyscallResult<()> {
        todo!()
    }
}

lazy_static! {
    static ref SYSCALLS_PROGRAM: (String, Program, SierraCasmRunner) =
        load_cairo_path("tests/tests/starknet/programs/syscalls.cairo");
}

#[test]
fn get_block_hash() {
    let result = run_native_program(
        &SYSCALLS_PROGRAM,
        "get_block_hash",
        &[],
        Some(u128::MAX),
        Some(SyscallHandler::new()),
    );

    assert_eq_sorted!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Felt252(
                Felt::from_dec_str(
                    "1158579293198495875788224011889333769139150068959598053296510642728083832673",
                )
                .unwrap()
            )),
            debug_name: None,
        },
    );
}

#[test]
fn get_execution_info() {
    let result = run_native_program(
        &SYSCALLS_PROGRAM,
        "get_execution_info",
        &[],
        Some(u128::MAX),
        Some(SyscallHandler::new()),
    );

    assert_eq_sorted!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                fields: vec![
                    JitValue::Struct {
                        fields: vec![
                            JitValue::Uint64(10057862467973663535),
                            JitValue::Uint64(13878668747512495966),
                            JitValue::Felt252(Felt::from_dec_str(
                                "1126241460712630201003776917997524449163698107789103849210792326381258973685",
                            )
                            .unwrap()),
                        ],
                        debug_name: None,
                    },
                    JitValue::Struct {
                        fields: vec![
                            JitValue::Felt252(Felt::from_dec_str(
                                "1724985403142256920476849371638528834056988111202434162793214195754735917407",
                            )
                            .unwrap()),
                            JitValue::Felt252(Felt::from_dec_str(
                                "2419272378964094005143278046496347854926114240785059742454535261490265649110",
                            )
                            .unwrap()),
                            JitValue::Uint128(67871905340377755668863509019681938001),
                            JitValue::Struct {
                                fields: vec![
                                    JitValue::Array(Vec::new()),
                                ],
                                debug_name: None
                            },
                            JitValue::Felt252(Felt::from_dec_str(
                                "2073267424102447009330753642820908998776456851902897601865334818765025369132",
                            )
                            .unwrap()),
                            JitValue::Felt252(Felt::from_dec_str(
                                "1727570805086347994328356733148206517040691113666039929118050093237140484117",
                            )
                            .unwrap()),
                            JitValue::Felt252(Felt::from_dec_str(
                                "2223335940097352947792108259394621577330089800429182023415494612506457867705",
                            )
                            .unwrap()),
                        ],
                        debug_name: None,
                    },
                    JitValue::Felt252(Felt::from_dec_str(
                        "2367044879643293830108311482898145302930693201376043522909298679498599559539",
                    )
                    .unwrap()),
                    JitValue::Felt252(Felt::from_dec_str(
                        "2322490563038631685097154208793293355074547843057070254216662565231428808211",
                    )
                    .unwrap()),
                    JitValue::Felt252(Felt::from_dec_str(
                        "1501296828847480842982002010206952982741090100977850506550982801410247026532",
                    )
                    .unwrap()),
                ],
                debug_name: None,
            }),
            debug_name: None,
        },
    );
}

#[test]
fn get_execution_info_v2() {
    let result = run_native_program(
        &SYSCALLS_PROGRAM,
        "get_execution_info_v2",
        &[],
        Some(u128::MAX),
        Some(SyscallHandler::new()),
    );

    assert_eq_sorted!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                fields: vec![
                    JitValue::Struct {
                        fields: vec![
                            JitValue::Uint64(10290342497028289173),
                            JitValue::Uint64(8376161426686560326),
                            JitValue::Felt252(Felt::from_dec_str(
                                "1815189516202718271265591469295511271015058493881778555617445818147186579905",
                            )
                            .unwrap()),
                        ],
                        debug_name: None,
                    },
                    JitValue::Struct {
                        fields: vec![
                            JitValue::Felt252(Felt::from_dec_str(
                                "1946630339019864531118751968563861838541265142438690346764722398811248737786",
                            )
                            .unwrap()),
                            JitValue::Felt252(Felt::from_dec_str(
                                "2501333093425095943815772537228190103182643237630648877273495185321298605376",
                            )
                            .unwrap()),
                            JitValue::Uint128(268753657614351187400966367706860329387),
                            JitValue::Struct {
                                fields: vec![JitValue::Array(Vec::new())],
                                debug_name: None,
                            },
                            JitValue::Felt252(Felt::from_dec_str(
                                "1123336726531770778820945049824733201592457249587063926479184903627272350002",
                            )
                            .unwrap()),
                            JitValue::Felt252(Felt::from_dec_str(
                                "2128916697180095451339935431635121484141376377516602728602049361615810538124",
                            )
                            .unwrap()),
                            JitValue::Felt252(Felt::from_dec_str(
                                "3012936192361023209451741736298028332652992971202997279327088951248532774884",
                            )
                            .unwrap()),
                            JitValue::Struct {
                                fields: vec![JitValue::Array(Vec::new())],
                                debug_name: None,
                            },
                            JitValue::Uint128(215444579144685671333997376989135077200),
                            JitValue::Struct {
                                fields: vec![JitValue::Array(Vec::new())],
                                debug_name: None,
                            },
                            JitValue::Uint32(140600095),
                            JitValue::Uint32(988370659),
                            JitValue::Struct {
                                fields: vec![JitValue::Array(Vec::new())],
                                debug_name: None,
                            },
                        ],
                        debug_name: None,
                    },
                    JitValue::Felt252(Felt::from_dec_str(
                        "1185632056775552928459345712365014492063999606476424661067102766803470217687",
                    )
                    .unwrap()),
                    JitValue::Felt252(Felt::from_dec_str(
                        "741063429140548584082645215539704615048011618665759826371923004739480130327",
                    )
                    .unwrap()),
                    JitValue::Felt252(Felt::from_dec_str(
                        "477501848519111015718660527024172361930966806556174677443839145770405114061",
                    )
                    .unwrap()),
                ],
                debug_name: None,
            }),
            debug_name: None,
        },
    );
}

#[test]
fn deploy() {
    let result = run_native_program(
        &SYSCALLS_PROGRAM,
        "deploy",
        &[],
        Some(u128::MAX),
        Some(SyscallHandler::new()),
    );

    assert_eq_sorted!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                fields: vec![
                    JitValue::Felt252(Felt::from_dec_str(
                        "1833707083418045616336697070784512826809940908236872124572250196391719980392",
                    )
                    .unwrap()),
                    JitValue::Struct {
                        fields: vec![JitValue::Array(Vec::new())],
                        debug_name: None,
                    },
                ],
                debug_name: None,
            }),
            debug_name: None,
        },
    )
}

#[test]
fn replace_class() {
    let result = run_native_program(
        &SYSCALLS_PROGRAM,
        "replace_class",
        &[],
        Some(u128::MAX),
        Some(SyscallHandler::new()),
    );

    assert_eq_sorted!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                fields: Vec::new(),
                debug_name: None,
            }),
            debug_name: None,
        },
    );
}

#[test]
fn library_call() {
    let result = run_native_program(
        &SYSCALLS_PROGRAM,
        "library_call",
        &[],
        Some(u128::MAX),
        Some(SyscallHandler::new()),
    );

    assert_eq_sorted!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                fields: vec![JitValue::Array(vec![
                    JitValue::Felt252(Felt::from_dec_str(
                        "3358892263739032253767642605669710712087178958719188919195252597609334880396"
                    )
                    .unwrap()),
                    JitValue::Felt252(Felt::from_dec_str(
                        "1104291043781086177955655234103730593173963850634630109574183288837411031513"
                    )
                    .unwrap()),
                    JitValue::Felt252(Felt::from_dec_str(
                        "3346377229881115874907650557159666001431249650068516742483979624047277128413"
                    )
                    .unwrap()),
                ])],
                debug_name: None,
            }),
            debug_name: None,
        },
    );
}

#[test]
fn call_contract() {
    let result = run_native_program(
        &SYSCALLS_PROGRAM,
        "call_contract",
        &[],
        Some(u128::MAX),
        Some(SyscallHandler::new()),
    );

    assert_eq_sorted!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                fields: vec![JitValue::Array(vec![
                    JitValue::Felt252(Felt::from_dec_str(
                        "3358892263739032253767642605669710712087178958719188919195252597609334880396"
                    )
                    .unwrap()),
                    JitValue::Felt252(Felt::from_dec_str(
                        "1104291043781086177955655234103730593173963850634630109574183288837411031513"
                    )
                    .unwrap()),
                    JitValue::Felt252(Felt::from_dec_str(
                        "3346377229881115874907650557159666001431249650068516742483979624047277128413"
                    )
                    .unwrap()),
                ])],
                debug_name: None,
            }),
            debug_name: None,
        },
    );
}

#[test]
fn storage_read() {
    let result = run_native_program(
        &SYSCALLS_PROGRAM,
        "storage_read",
        &[],
        Some(u128::MAX),
        Some(SyscallHandler::new()),
    );

    assert_eq_sorted!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                fields: vec![
                    JitValue::Felt252(Felt::from_dec_str(
                        "1013181629378419652272218169322268188846114273878719855200100663863924329981",
                    )
                    .unwrap()),
                ],
                debug_name: None,
            }),
            debug_name: None,
        },
    );
}

#[test]
fn storage_write() {
    let result = run_native_program(
        &SYSCALLS_PROGRAM,
        "storage_write",
        &[],
        Some(u128::MAX),
        Some(SyscallHandler::new()),
    );

    assert_eq_sorted!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                fields: vec![JitValue::Struct {
                    fields: Vec::new(),
                    debug_name: None,
                }],
                debug_name: None,
            }),
            debug_name: None,
        },
    );
}

#[test]
fn emit_event() {
    let result = run_native_program(
        &SYSCALLS_PROGRAM,
        "emit_event",
        &[],
        Some(u128::MAX),
        Some(SyscallHandler::new()),
    );

    assert_eq_sorted!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                fields: Vec::new(),
                debug_name: None,
            }),
            debug_name: None,
        },
    );
}

#[test]
fn send_message_to_l1() {
    let result = run_native_program(
        &SYSCALLS_PROGRAM,
        "send_message_to_l1",
        &[],
        Some(u128::MAX),
        Some(SyscallHandler::new()),
    );

    assert_eq_sorted!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                fields: Vec::new(),
                debug_name: None,
            }),
            debug_name: None,
        },
    );
}

#[test]
fn keccak() {
    let result = run_native_program(
        &SYSCALLS_PROGRAM,
        "keccak",
        &[],
        Some(u128::MAX),
        Some(SyscallHandler::new()),
    );

    assert_eq_sorted!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                fields: vec![
                    JitValue::Uint128(330939983442938156232262046592599923289),
                    JitValue::Uint128(288102973244655531496349286021939642254),
                ],
                debug_name: None,
            }),
            debug_name: None,
        },
    );
}

#[test]
fn set_sequencer_address() {
    let result = run_native_program(
        &SYSCALLS_PROGRAM,
        "set_sequencer_address",
        &[],
        Some(u128::MAX),
        Some(SyscallHandler::new()),
    );

    assert_eq_sorted!(
        result.return_value,
        JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                fields: vec![JitValue::Struct {
                    fields: Vec::new(),
                    debug_name: None,
                }],
                debug_name: None,
            }),
            debug_name: None,
        },
    );
}
