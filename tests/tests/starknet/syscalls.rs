//use crate::common::{load_cairo_path, run_native_program};
use crate::common::{load_cairo_path, run_native_program};
//use cairo_lang_runner::SierraCasmRunner;
use cairo_lang_runner::SierraCasmRunner;
//use cairo_lang_sierra::program::Program;
use cairo_lang_sierra::program::Program;
//use cairo_native::{
use cairo_native::{
//    starknet::{
    starknet::{
//        BlockInfo, ExecutionInfo, ExecutionInfoV2, Secp256k1Point, Secp256r1Point,
        BlockInfo, ExecutionInfo, ExecutionInfoV2, Secp256k1Point, Secp256r1Point,
//        StarknetSyscallHandler, SyscallResult, TxInfo, TxV2Info, U256,
        StarknetSyscallHandler, SyscallResult, TxInfo, TxV2Info, U256,
//    },
    },
//    values::JitValue,
    values::JitValue,
//};
};
//use lazy_static::lazy_static;
use lazy_static::lazy_static;
//use pretty_assertions_sorted::assert_eq_sorted;
use pretty_assertions_sorted::assert_eq_sorted;
//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
//

//struct SyscallHandler;
struct SyscallHandler;
//

//impl StarknetSyscallHandler for SyscallHandler {
impl StarknetSyscallHandler for SyscallHandler {
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
//        Ok(Felt::from_dec_str(
        Ok(Felt::from_dec_str(
//            "1158579293198495875788224011889333769139150068959598053296510642728083832673",
            "1158579293198495875788224011889333769139150068959598053296510642728083832673",
//        )
        )
//        .unwrap())
        .unwrap())
//    }
    }
//

//    fn get_execution_info(&mut self, _remaining_gas: &mut u128) -> SyscallResult<ExecutionInfo> {
    fn get_execution_info(&mut self, _remaining_gas: &mut u128) -> SyscallResult<ExecutionInfo> {
//        Ok(ExecutionInfo {
        Ok(ExecutionInfo {
//            block_info: BlockInfo {
            block_info: BlockInfo {
//                block_number: 10057862467973663535,
                block_number: 10057862467973663535,
//                block_timestamp: 13878668747512495966,
                block_timestamp: 13878668747512495966,
//                sequencer_address: Felt::from_dec_str(
                sequencer_address: Felt::from_dec_str(
//                    "1126241460712630201003776917997524449163698107789103849210792326381258973685",
                    "1126241460712630201003776917997524449163698107789103849210792326381258973685",
//                )
                )
//                .unwrap(),
                .unwrap(),
//            },
            },
//            tx_info: TxInfo {
            tx_info: TxInfo {
//                version: Felt::from_dec_str(
                version: Felt::from_dec_str(
//                    "1724985403142256920476849371638528834056988111202434162793214195754735917407",
                    "1724985403142256920476849371638528834056988111202434162793214195754735917407",
//                )
                )
//                .unwrap(),
                .unwrap(),
//                account_contract_address: Felt::from_dec_str(
                account_contract_address: Felt::from_dec_str(
//                    "2419272378964094005143278046496347854926114240785059742454535261490265649110",
                    "2419272378964094005143278046496347854926114240785059742454535261490265649110",
//                )
                )
//                .unwrap(),
                .unwrap(),
//                max_fee: 67871905340377755668863509019681938001,
                max_fee: 67871905340377755668863509019681938001,
//                signature: Vec::new(),
                signature: Vec::new(),
//                transaction_hash: Felt::from_dec_str(
                transaction_hash: Felt::from_dec_str(
//                    "2073267424102447009330753642820908998776456851902897601865334818765025369132",
                    "2073267424102447009330753642820908998776456851902897601865334818765025369132",
//                )
                )
//                .unwrap(),
                .unwrap(),
//                chain_id: Felt::from_dec_str(
                chain_id: Felt::from_dec_str(
//                    "1727570805086347994328356733148206517040691113666039929118050093237140484117",
                    "1727570805086347994328356733148206517040691113666039929118050093237140484117",
//                )
                )
//                .unwrap(),
                .unwrap(),
//                nonce: Felt::from_dec_str(
                nonce: Felt::from_dec_str(
//                    "2223335940097352947792108259394621577330089800429182023415494612506457867705",
                    "2223335940097352947792108259394621577330089800429182023415494612506457867705",
//                )
                )
//                .unwrap(),
                .unwrap(),
//            },
            },
//            caller_address: Felt::from_dec_str(
            caller_address: Felt::from_dec_str(
//                "2367044879643293830108311482898145302930693201376043522909298679498599559539",
                "2367044879643293830108311482898145302930693201376043522909298679498599559539",
//            )
            )
//            .unwrap(),
            .unwrap(),
//            contract_address: Felt::from_dec_str(
            contract_address: Felt::from_dec_str(
//                "2322490563038631685097154208793293355074547843057070254216662565231428808211",
                "2322490563038631685097154208793293355074547843057070254216662565231428808211",
//            )
            )
//            .unwrap(),
            .unwrap(),
//            entry_point_selector: Felt::from_dec_str(
            entry_point_selector: Felt::from_dec_str(
//                "1501296828847480842982002010206952982741090100977850506550982801410247026532",
                "1501296828847480842982002010206952982741090100977850506550982801410247026532",
//            )
            )
//            .unwrap(),
            .unwrap(),
//        })
        })
//    }
    }
//

//    fn get_execution_info_v2(
    fn get_execution_info_v2(
//        &mut self,
        &mut self,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<ExecutionInfoV2> {
    ) -> SyscallResult<ExecutionInfoV2> {
//        Ok(ExecutionInfoV2 {
        Ok(ExecutionInfoV2 {
//            block_info: BlockInfo {
            block_info: BlockInfo {
//                block_number: 10290342497028289173,
                block_number: 10290342497028289173,
//                block_timestamp: 8376161426686560326,
                block_timestamp: 8376161426686560326,
//                sequencer_address: Felt::from_dec_str(
                sequencer_address: Felt::from_dec_str(
//                    "1815189516202718271265591469295511271015058493881778555617445818147186579905",
                    "1815189516202718271265591469295511271015058493881778555617445818147186579905",
//                )
                )
//                .unwrap(),
                .unwrap(),
//            },
            },
//            tx_info: TxV2Info {
            tx_info: TxV2Info {
//                version: Felt::from_dec_str(
                version: Felt::from_dec_str(
//                    "1946630339019864531118751968563861838541265142438690346764722398811248737786",
                    "1946630339019864531118751968563861838541265142438690346764722398811248737786",
//                )
                )
//                .unwrap(),
                .unwrap(),
//                account_contract_address: Felt::from_dec_str(
                account_contract_address: Felt::from_dec_str(
//                    "2501333093425095943815772537228190103182643237630648877273495185321298605376",
                    "2501333093425095943815772537228190103182643237630648877273495185321298605376",
//                )
                )
//                .unwrap(),
                .unwrap(),
//                max_fee: 268753657614351187400966367706860329387,
                max_fee: 268753657614351187400966367706860329387,
//                signature: Vec::new(),
                signature: Vec::new(),
//                transaction_hash: Felt::from_dec_str(
                transaction_hash: Felt::from_dec_str(
//                    "1123336726531770778820945049824733201592457249587063926479184903627272350002",
                    "1123336726531770778820945049824733201592457249587063926479184903627272350002",
//                )
                )
//                .unwrap(),
                .unwrap(),
//                chain_id: Felt::from_dec_str(
                chain_id: Felt::from_dec_str(
//                    "2128916697180095451339935431635121484141376377516602728602049361615810538124",
                    "2128916697180095451339935431635121484141376377516602728602049361615810538124",
//                )
                )
//                .unwrap(),
                .unwrap(),
//                nonce: Felt::from_dec_str(
                nonce: Felt::from_dec_str(
//                    "3012936192361023209451741736298028332652992971202997279327088951248532774884",
                    "3012936192361023209451741736298028332652992971202997279327088951248532774884",
//                )
                )
//                .unwrap(),
                .unwrap(),
//                resource_bounds: Vec::new(),
                resource_bounds: Vec::new(),
//                tip: 215444579144685671333997376989135077200,
                tip: 215444579144685671333997376989135077200,
//                paymaster_data: Vec::new(),
                paymaster_data: Vec::new(),
//                nonce_data_availability_mode: 140600095,
                nonce_data_availability_mode: 140600095,
//                fee_data_availability_mode: 988370659,
                fee_data_availability_mode: 988370659,
//                account_deployment_data: Vec::new(),
                account_deployment_data: Vec::new(),
//            },
            },
//            caller_address: Felt::from_dec_str(
            caller_address: Felt::from_dec_str(
//                "1185632056775552928459345712365014492063999606476424661067102766803470217687",
                "1185632056775552928459345712365014492063999606476424661067102766803470217687",
//            )
            )
//            .unwrap(),
            .unwrap(),
//            contract_address: Felt::from_dec_str(
            contract_address: Felt::from_dec_str(
//                "741063429140548584082645215539704615048011618665759826371923004739480130327",
                "741063429140548584082645215539704615048011618665759826371923004739480130327",
//            )
            )
//            .unwrap(),
            .unwrap(),
//            entry_point_selector: Felt::from_dec_str(
            entry_point_selector: Felt::from_dec_str(
//                "477501848519111015718660527024172361930966806556174677443839145770405114061",
                "477501848519111015718660527024172361930966806556174677443839145770405114061",
//            )
            )
//            .unwrap(),
            .unwrap(),
//        })
        })
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
//        Ok((
        Ok((
//            Felt::from_dec_str(
            Felt::from_dec_str(
//                "1833707083418045616336697070784512826809940908236872124572250196391719980392",
                "1833707083418045616336697070784512826809940908236872124572250196391719980392",
//            )
            )
//            .unwrap(),
            .unwrap(),
//            Vec::new(),
            Vec::new(),
//        ))
        ))
//    }
    }
//

//    fn replace_class(&mut self, _class_hash: Felt, _remaining_gas: &mut u128) -> SyscallResult<()> {
    fn replace_class(&mut self, _class_hash: Felt, _remaining_gas: &mut u128) -> SyscallResult<()> {
//        Ok(())
        Ok(())
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
//        Ok(vec![
        Ok(vec![
//            Felt::from_dec_str(
            Felt::from_dec_str(
//                "3358892263739032253767642605669710712087178958719188919195252597609334880396",
                "3358892263739032253767642605669710712087178958719188919195252597609334880396",
//            )
            )
//            .unwrap(),
            .unwrap(),
//            Felt::from_dec_str(
            Felt::from_dec_str(
//                "1104291043781086177955655234103730593173963850634630109574183288837411031513",
                "1104291043781086177955655234103730593173963850634630109574183288837411031513",
//            )
            )
//            .unwrap(),
            .unwrap(),
//            Felt::from_dec_str(
            Felt::from_dec_str(
//                "3346377229881115874907650557159666001431249650068516742483979624047277128413",
                "3346377229881115874907650557159666001431249650068516742483979624047277128413",
//            )
            )
//            .unwrap(),
            .unwrap(),
//        ])
        ])
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
//        Ok(vec![
        Ok(vec![
//            Felt::from_dec_str(
            Felt::from_dec_str(
//                "3358892263739032253767642605669710712087178958719188919195252597609334880396",
                "3358892263739032253767642605669710712087178958719188919195252597609334880396",
//            )
            )
//            .unwrap(),
            .unwrap(),
//            Felt::from_dec_str(
            Felt::from_dec_str(
//                "1104291043781086177955655234103730593173963850634630109574183288837411031513",
                "1104291043781086177955655234103730593173963850634630109574183288837411031513",
//            )
            )
//            .unwrap(),
            .unwrap(),
//            Felt::from_dec_str(
            Felt::from_dec_str(
//                "3346377229881115874907650557159666001431249650068516742483979624047277128413",
                "3346377229881115874907650557159666001431249650068516742483979624047277128413",
//            )
            )
//            .unwrap(),
            .unwrap(),
//        ])
        ])
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
//        Ok(Felt::from_dec_str(
        Ok(Felt::from_dec_str(
//            "1013181629378419652272218169322268188846114273878719855200100663863924329981",
            "1013181629378419652272218169322268188846114273878719855200100663863924329981",
//        )
        )
//        .unwrap())
        .unwrap())
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
//        Ok(())
        Ok(())
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
//        Ok(())
        Ok(())
//    }
    }
//

//    fn send_message_to_l1(
    fn send_message_to_l1(
//        &mut self,
        &mut self,
//        to_address: Felt,
        to_address: Felt,
//        payload: &[Felt],
        payload: &[Felt],
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<()> {
    ) -> SyscallResult<()> {
//        assert_eq!(
        assert_eq!(
//            to_address,
            to_address,
//            3.into(),
            3.into(),
//            "send_message_to_l1 to_address mismatch"
            "send_message_to_l1 to_address mismatch"
//        );
        );
//        assert_eq!(payload, &[2.into()], "send_message_to_l1 payload mismatch");
        assert_eq!(payload, &[2.into()], "send_message_to_l1 payload mismatch");
//        Ok(())
        Ok(())
//    }
    }
//

//    fn keccak(&mut self, _input: &[u64], _remaining_gas: &mut u128) -> SyscallResult<U256> {
    fn keccak(&mut self, _input: &[u64], _remaining_gas: &mut u128) -> SyscallResult<U256> {
//        Ok(U256 {
        Ok(U256 {
//            hi: 330939983442938156232262046592599923289,
            hi: 330939983442938156232262046592599923289,
//            lo: 288102973244655531496349286021939642254,
            lo: 288102973244655531496349286021939642254,
//        })
        })
//    }
    }
//

//    fn secp256k1_new(
    fn secp256k1_new(
//        &mut self,
        &mut self,
//        _x: U256,
        _x: U256,
//        _y: U256,
        _y: U256,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Option<Secp256k1Point>> {
    ) -> SyscallResult<Option<Secp256k1Point>> {
//        // Tested in `tests/tests/starknet/secp256.rs`.
        // Tested in `tests/tests/starknet/secp256.rs`.
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn secp256k1_add(
    fn secp256k1_add(
//        &mut self,
        &mut self,
//        _p0: Secp256k1Point,
        _p0: Secp256k1Point,
//        _p1: Secp256k1Point,
        _p1: Secp256k1Point,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Secp256k1Point> {
    ) -> SyscallResult<Secp256k1Point> {
//        // Tested in `tests/tests/starknet/secp256.rs`.
        // Tested in `tests/tests/starknet/secp256.rs`.
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn secp256k1_mul(
    fn secp256k1_mul(
//        &mut self,
        &mut self,
//        _p: Secp256k1Point,
        _p: Secp256k1Point,
//        _m: U256,
        _m: U256,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Secp256k1Point> {
    ) -> SyscallResult<Secp256k1Point> {
//        // Tested in `tests/tests/starknet/secp256.rs`.
        // Tested in `tests/tests/starknet/secp256.rs`.
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn secp256k1_get_point_from_x(
    fn secp256k1_get_point_from_x(
//        &mut self,
        &mut self,
//        _x: U256,
        _x: U256,
//        _y_parity: bool,
        _y_parity: bool,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Option<Secp256k1Point>> {
    ) -> SyscallResult<Option<Secp256k1Point>> {
//        // Tested in `tests/tests/starknet/secp256.rs`.
        // Tested in `tests/tests/starknet/secp256.rs`.
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn secp256k1_get_xy(
    fn secp256k1_get_xy(
//        &mut self,
        &mut self,
//        _p: Secp256k1Point,
        _p: Secp256k1Point,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<(U256, U256)> {
    ) -> SyscallResult<(U256, U256)> {
//        // Tested in `tests/tests/starknet/secp256.rs`.
        // Tested in `tests/tests/starknet/secp256.rs`.
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn secp256r1_new(
    fn secp256r1_new(
//        &mut self,
        &mut self,
//        _x: U256,
        _x: U256,
//        _y: U256,
        _y: U256,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Option<Secp256r1Point>> {
    ) -> SyscallResult<Option<Secp256r1Point>> {
//        // Tested in `tests/tests/starknet/secp256.rs`.
        // Tested in `tests/tests/starknet/secp256.rs`.
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn secp256r1_add(
    fn secp256r1_add(
//        &mut self,
        &mut self,
//        _p0: Secp256r1Point,
        _p0: Secp256r1Point,
//        _p1: Secp256r1Point,
        _p1: Secp256r1Point,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Secp256r1Point> {
    ) -> SyscallResult<Secp256r1Point> {
//        // Tested in `tests/tests/starknet/secp256.rs`.
        // Tested in `tests/tests/starknet/secp256.rs`.
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn secp256r1_mul(
    fn secp256r1_mul(
//        &mut self,
        &mut self,
//        _p: Secp256r1Point,
        _p: Secp256r1Point,
//        _m: U256,
        _m: U256,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Secp256r1Point> {
    ) -> SyscallResult<Secp256r1Point> {
//        // Tested in `tests/tests/starknet/secp256.rs`.
        // Tested in `tests/tests/starknet/secp256.rs`.
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn secp256r1_get_point_from_x(
    fn secp256r1_get_point_from_x(
//        &mut self,
        &mut self,
//        _x: U256,
        _x: U256,
//        _y_parity: bool,
        _y_parity: bool,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Option<Secp256r1Point>> {
    ) -> SyscallResult<Option<Secp256r1Point>> {
//        // Tested in `tests/tests/starknet/secp256.rs`.
        // Tested in `tests/tests/starknet/secp256.rs`.
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn secp256r1_get_xy(
    fn secp256r1_get_xy(
//        &mut self,
        &mut self,
//        _p: Secp256r1Point,
        _p: Secp256r1Point,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<(U256, U256)> {
    ) -> SyscallResult<(U256, U256)> {
//        // Tested in `tests/tests/starknet/secp256.rs`.
        // Tested in `tests/tests/starknet/secp256.rs`.
//        unimplemented!()
        unimplemented!()
//    }
    }
//}
}
//

//lazy_static! {
lazy_static! {
//    static ref SYSCALLS_PROGRAM: (String, Program, SierraCasmRunner) =
    static ref SYSCALLS_PROGRAM: (String, Program, SierraCasmRunner) =
//        load_cairo_path("tests/tests/starknet/programs/syscalls.cairo");
        load_cairo_path("tests/tests/starknet/programs/syscalls.cairo");
//}
}
//

//#[test]
#[test]
//fn get_block_hash() {
fn get_block_hash() {
//    let result = run_native_program(
    let result = run_native_program(
//        &SYSCALLS_PROGRAM,
        &SYSCALLS_PROGRAM,
//        "get_block_hash",
        "get_block_hash",
//        &[],
        &[],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(SyscallHandler),
        Some(SyscallHandler),
//    );
    );
//

//    assert_eq_sorted!(
    assert_eq_sorted!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Felt252(
            value: Box::new(JitValue::Felt252(
//                Felt::from_dec_str(
                Felt::from_dec_str(
//                    "1158579293198495875788224011889333769139150068959598053296510642728083832673",
                    "1158579293198495875788224011889333769139150068959598053296510642728083832673",
//                )
                )
//                .unwrap()
                .unwrap()
//            )),
            )),
//            debug_name: None,
            debug_name: None,
//        },
        },
//    );
    );
//}
}
//

//#[test]
#[test]
//fn get_execution_info() {
fn get_execution_info() {
//    let result = run_native_program(
    let result = run_native_program(
//        &SYSCALLS_PROGRAM,
        &SYSCALLS_PROGRAM,
//        "get_execution_info",
        "get_execution_info",
//        &[],
        &[],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(SyscallHandler),
        Some(SyscallHandler),
//    );
    );
//

//    assert_eq_sorted!(
    assert_eq_sorted!(
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
//                        fields: vec![
                        fields: vec![
//                            JitValue::Uint64(10057862467973663535),
                            JitValue::Uint64(10057862467973663535),
//                            JitValue::Uint64(13878668747512495966),
                            JitValue::Uint64(13878668747512495966),
//                            JitValue::Felt252(Felt::from_dec_str(
                            JitValue::Felt252(Felt::from_dec_str(
//                                "1126241460712630201003776917997524449163698107789103849210792326381258973685",
                                "1126241460712630201003776917997524449163698107789103849210792326381258973685",
//                            )
                            )
//                            .unwrap()),
                            .unwrap()),
//                        ],
                        ],
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: vec![
                        fields: vec![
//                            JitValue::Felt252(Felt::from_dec_str(
                            JitValue::Felt252(Felt::from_dec_str(
//                                "1724985403142256920476849371638528834056988111202434162793214195754735917407",
                                "1724985403142256920476849371638528834056988111202434162793214195754735917407",
//                            )
                            )
//                            .unwrap()),
                            .unwrap()),
//                            JitValue::Felt252(Felt::from_dec_str(
                            JitValue::Felt252(Felt::from_dec_str(
//                                "2419272378964094005143278046496347854926114240785059742454535261490265649110",
                                "2419272378964094005143278046496347854926114240785059742454535261490265649110",
//                            )
                            )
//                            .unwrap()),
                            .unwrap()),
//                            JitValue::Uint128(67871905340377755668863509019681938001),
                            JitValue::Uint128(67871905340377755668863509019681938001),
//                            JitValue::Struct {
                            JitValue::Struct {
//                                fields: vec![
                                fields: vec![
//                                    JitValue::Array(Vec::new()),
                                    JitValue::Array(Vec::new()),
//                                ],
                                ],
//                                debug_name: None
                                debug_name: None
//                            },
                            },
//                            JitValue::Felt252(Felt::from_dec_str(
                            JitValue::Felt252(Felt::from_dec_str(
//                                "2073267424102447009330753642820908998776456851902897601865334818765025369132",
                                "2073267424102447009330753642820908998776456851902897601865334818765025369132",
//                            )
                            )
//                            .unwrap()),
                            .unwrap()),
//                            JitValue::Felt252(Felt::from_dec_str(
                            JitValue::Felt252(Felt::from_dec_str(
//                                "1727570805086347994328356733148206517040691113666039929118050093237140484117",
                                "1727570805086347994328356733148206517040691113666039929118050093237140484117",
//                            )
                            )
//                            .unwrap()),
                            .unwrap()),
//                            JitValue::Felt252(Felt::from_dec_str(
                            JitValue::Felt252(Felt::from_dec_str(
//                                "2223335940097352947792108259394621577330089800429182023415494612506457867705",
                                "2223335940097352947792108259394621577330089800429182023415494612506457867705",
//                            )
                            )
//                            .unwrap()),
                            .unwrap()),
//                        ],
                        ],
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                    JitValue::Felt252(Felt::from_dec_str(
                    JitValue::Felt252(Felt::from_dec_str(
//                        "2367044879643293830108311482898145302930693201376043522909298679498599559539",
                        "2367044879643293830108311482898145302930693201376043522909298679498599559539",
//                    )
                    )
//                    .unwrap()),
                    .unwrap()),
//                    JitValue::Felt252(Felt::from_dec_str(
                    JitValue::Felt252(Felt::from_dec_str(
//                        "2322490563038631685097154208793293355074547843057070254216662565231428808211",
                        "2322490563038631685097154208793293355074547843057070254216662565231428808211",
//                    )
                    )
//                    .unwrap()),
                    .unwrap()),
//                    JitValue::Felt252(Felt::from_dec_str(
                    JitValue::Felt252(Felt::from_dec_str(
//                        "1501296828847480842982002010206952982741090100977850506550982801410247026532",
                        "1501296828847480842982002010206952982741090100977850506550982801410247026532",
//                    )
                    )
//                    .unwrap()),
                    .unwrap()),
//                ],
                ],
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
//}
}
//

//#[test]
#[test]
//fn get_execution_info_v2() {
fn get_execution_info_v2() {
//    let result = run_native_program(
    let result = run_native_program(
//        &SYSCALLS_PROGRAM,
        &SYSCALLS_PROGRAM,
//        "get_execution_info_v2",
        "get_execution_info_v2",
//        &[],
        &[],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(SyscallHandler),
        Some(SyscallHandler),
//    );
    );
//

//    assert_eq_sorted!(
    assert_eq_sorted!(
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
//                        fields: vec![
                        fields: vec![
//                            JitValue::Uint64(10290342497028289173),
                            JitValue::Uint64(10290342497028289173),
//                            JitValue::Uint64(8376161426686560326),
                            JitValue::Uint64(8376161426686560326),
//                            JitValue::Felt252(Felt::from_dec_str(
                            JitValue::Felt252(Felt::from_dec_str(
//                                "1815189516202718271265591469295511271015058493881778555617445818147186579905",
                                "1815189516202718271265591469295511271015058493881778555617445818147186579905",
//                            )
                            )
//                            .unwrap()),
                            .unwrap()),
//                        ],
                        ],
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: vec![
                        fields: vec![
//                            JitValue::Felt252(Felt::from_dec_str(
                            JitValue::Felt252(Felt::from_dec_str(
//                                "1946630339019864531118751968563861838541265142438690346764722398811248737786",
                                "1946630339019864531118751968563861838541265142438690346764722398811248737786",
//                            )
                            )
//                            .unwrap()),
                            .unwrap()),
//                            JitValue::Felt252(Felt::from_dec_str(
                            JitValue::Felt252(Felt::from_dec_str(
//                                "2501333093425095943815772537228190103182643237630648877273495185321298605376",
                                "2501333093425095943815772537228190103182643237630648877273495185321298605376",
//                            )
                            )
//                            .unwrap()),
                            .unwrap()),
//                            JitValue::Uint128(268753657614351187400966367706860329387),
                            JitValue::Uint128(268753657614351187400966367706860329387),
//                            JitValue::Struct {
                            JitValue::Struct {
//                                fields: vec![JitValue::Array(Vec::new())],
                                fields: vec![JitValue::Array(Vec::new())],
//                                debug_name: None,
                                debug_name: None,
//                            },
                            },
//                            JitValue::Felt252(Felt::from_dec_str(
                            JitValue::Felt252(Felt::from_dec_str(
//                                "1123336726531770778820945049824733201592457249587063926479184903627272350002",
                                "1123336726531770778820945049824733201592457249587063926479184903627272350002",
//                            )
                            )
//                            .unwrap()),
                            .unwrap()),
//                            JitValue::Felt252(Felt::from_dec_str(
                            JitValue::Felt252(Felt::from_dec_str(
//                                "2128916697180095451339935431635121484141376377516602728602049361615810538124",
                                "2128916697180095451339935431635121484141376377516602728602049361615810538124",
//                            )
                            )
//                            .unwrap()),
                            .unwrap()),
//                            JitValue::Felt252(Felt::from_dec_str(
                            JitValue::Felt252(Felt::from_dec_str(
//                                "3012936192361023209451741736298028332652992971202997279327088951248532774884",
                                "3012936192361023209451741736298028332652992971202997279327088951248532774884",
//                            )
                            )
//                            .unwrap()),
                            .unwrap()),
//                            JitValue::Struct {
                            JitValue::Struct {
//                                fields: vec![JitValue::Array(Vec::new())],
                                fields: vec![JitValue::Array(Vec::new())],
//                                debug_name: None,
                                debug_name: None,
//                            },
                            },
//                            JitValue::Uint128(215444579144685671333997376989135077200),
                            JitValue::Uint128(215444579144685671333997376989135077200),
//                            JitValue::Struct {
                            JitValue::Struct {
//                                fields: vec![JitValue::Array(Vec::new())],
                                fields: vec![JitValue::Array(Vec::new())],
//                                debug_name: None,
                                debug_name: None,
//                            },
                            },
//                            JitValue::Uint32(140600095),
                            JitValue::Uint32(140600095),
//                            JitValue::Uint32(988370659),
                            JitValue::Uint32(988370659),
//                            JitValue::Struct {
                            JitValue::Struct {
//                                fields: vec![JitValue::Array(Vec::new())],
                                fields: vec![JitValue::Array(Vec::new())],
//                                debug_name: None,
                                debug_name: None,
//                            },
                            },
//                        ],
                        ],
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                    JitValue::Felt252(Felt::from_dec_str(
                    JitValue::Felt252(Felt::from_dec_str(
//                        "1185632056775552928459345712365014492063999606476424661067102766803470217687",
                        "1185632056775552928459345712365014492063999606476424661067102766803470217687",
//                    )
                    )
//                    .unwrap()),
                    .unwrap()),
//                    JitValue::Felt252(Felt::from_dec_str(
                    JitValue::Felt252(Felt::from_dec_str(
//                        "741063429140548584082645215539704615048011618665759826371923004739480130327",
                        "741063429140548584082645215539704615048011618665759826371923004739480130327",
//                    )
                    )
//                    .unwrap()),
                    .unwrap()),
//                    JitValue::Felt252(Felt::from_dec_str(
                    JitValue::Felt252(Felt::from_dec_str(
//                        "477501848519111015718660527024172361930966806556174677443839145770405114061",
                        "477501848519111015718660527024172361930966806556174677443839145770405114061",
//                    )
                    )
//                    .unwrap()),
                    .unwrap()),
//                ],
                ],
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
//}
}
//

//#[test]
#[test]
//fn deploy() {
fn deploy() {
//    let result = run_native_program(
    let result = run_native_program(
//        &SYSCALLS_PROGRAM,
        &SYSCALLS_PROGRAM,
//        "deploy",
        "deploy",
//        &[],
        &[],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(SyscallHandler),
        Some(SyscallHandler),
//    );
    );
//

//    assert_eq_sorted!(
    assert_eq_sorted!(
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
//                    JitValue::Felt252(Felt::from_dec_str(
                    JitValue::Felt252(Felt::from_dec_str(
//                        "1833707083418045616336697070784512826809940908236872124572250196391719980392",
                        "1833707083418045616336697070784512826809940908236872124572250196391719980392",
//                    )
                    )
//                    .unwrap()),
                    .unwrap()),
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: vec![JitValue::Array(Vec::new())],
                        fields: vec![JitValue::Array(Vec::new())],
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
//        },
        },
//    )
    )
//}
}
//

//#[test]
#[test]
//fn replace_class() {
fn replace_class() {
//    let result = run_native_program(
    let result = run_native_program(
//        &SYSCALLS_PROGRAM,
        &SYSCALLS_PROGRAM,
//        "replace_class",
        "replace_class",
//        &[],
        &[],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(SyscallHandler),
        Some(SyscallHandler),
//    );
    );
//

//    assert_eq_sorted!(
    assert_eq_sorted!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Struct {
            value: Box::new(JitValue::Struct {
//                fields: Vec::new(),
                fields: Vec::new(),
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
//}
}
//

//#[test]
#[test]
//fn library_call() {
fn library_call() {
//    let result = run_native_program(
    let result = run_native_program(
//        &SYSCALLS_PROGRAM,
        &SYSCALLS_PROGRAM,
//        "library_call",
        "library_call",
//        &[],
        &[],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(SyscallHandler),
        Some(SyscallHandler),
//    );
    );
//

//    assert_eq_sorted!(
    assert_eq_sorted!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Struct {
            value: Box::new(JitValue::Struct {
//                fields: vec![JitValue::Array(vec![
                fields: vec![JitValue::Array(vec![
//                    JitValue::Felt252(Felt::from_dec_str(
                    JitValue::Felt252(Felt::from_dec_str(
//                        "3358892263739032253767642605669710712087178958719188919195252597609334880396"
                        "3358892263739032253767642605669710712087178958719188919195252597609334880396"
//                    )
                    )
//                    .unwrap()),
                    .unwrap()),
//                    JitValue::Felt252(Felt::from_dec_str(
                    JitValue::Felt252(Felt::from_dec_str(
//                        "1104291043781086177955655234103730593173963850634630109574183288837411031513"
                        "1104291043781086177955655234103730593173963850634630109574183288837411031513"
//                    )
                    )
//                    .unwrap()),
                    .unwrap()),
//                    JitValue::Felt252(Felt::from_dec_str(
                    JitValue::Felt252(Felt::from_dec_str(
//                        "3346377229881115874907650557159666001431249650068516742483979624047277128413"
                        "3346377229881115874907650557159666001431249650068516742483979624047277128413"
//                    )
                    )
//                    .unwrap()),
                    .unwrap()),
//                ])],
                ])],
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
//}
}
//

//#[test]
#[test]
//fn call_contract() {
fn call_contract() {
//    let result = run_native_program(
    let result = run_native_program(
//        &SYSCALLS_PROGRAM,
        &SYSCALLS_PROGRAM,
//        "call_contract",
        "call_contract",
//        &[],
        &[],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(SyscallHandler),
        Some(SyscallHandler),
//    );
    );
//

//    assert_eq_sorted!(
    assert_eq_sorted!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Struct {
            value: Box::new(JitValue::Struct {
//                fields: vec![JitValue::Array(vec![
                fields: vec![JitValue::Array(vec![
//                    JitValue::Felt252(Felt::from_dec_str(
                    JitValue::Felt252(Felt::from_dec_str(
//                        "3358892263739032253767642605669710712087178958719188919195252597609334880396"
                        "3358892263739032253767642605669710712087178958719188919195252597609334880396"
//                    )
                    )
//                    .unwrap()),
                    .unwrap()),
//                    JitValue::Felt252(Felt::from_dec_str(
                    JitValue::Felt252(Felt::from_dec_str(
//                        "1104291043781086177955655234103730593173963850634630109574183288837411031513"
                        "1104291043781086177955655234103730593173963850634630109574183288837411031513"
//                    )
                    )
//                    .unwrap()),
                    .unwrap()),
//                    JitValue::Felt252(Felt::from_dec_str(
                    JitValue::Felt252(Felt::from_dec_str(
//                        "3346377229881115874907650557159666001431249650068516742483979624047277128413"
                        "3346377229881115874907650557159666001431249650068516742483979624047277128413"
//                    )
                    )
//                    .unwrap()),
                    .unwrap()),
//                ])],
                ])],
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
//}
}
//

//#[test]
#[test]
//fn storage_read() {
fn storage_read() {
//    let result = run_native_program(
    let result = run_native_program(
//        &SYSCALLS_PROGRAM,
        &SYSCALLS_PROGRAM,
//        "storage_read",
        "storage_read",
//        &[],
        &[],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(SyscallHandler),
        Some(SyscallHandler),
//    );
    );
//

//    assert_eq_sorted!(
    assert_eq_sorted!(
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
//                    JitValue::Felt252(Felt::from_dec_str(
                    JitValue::Felt252(Felt::from_dec_str(
//                        "1013181629378419652272218169322268188846114273878719855200100663863924329981",
                        "1013181629378419652272218169322268188846114273878719855200100663863924329981",
//                    )
                    )
//                    .unwrap()),
                    .unwrap()),
//                ],
                ],
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
//}
}
//

//#[test]
#[test]
//fn storage_write() {
fn storage_write() {
//    let result = run_native_program(
    let result = run_native_program(
//        &SYSCALLS_PROGRAM,
        &SYSCALLS_PROGRAM,
//        "storage_write",
        "storage_write",
//        &[],
        &[],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(SyscallHandler),
        Some(SyscallHandler),
//    );
    );
//

//    assert_eq_sorted!(
    assert_eq_sorted!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Struct {
            value: Box::new(JitValue::Struct {
//                fields: vec![JitValue::Struct {
                fields: vec![JitValue::Struct {
//                    fields: Vec::new(),
                    fields: Vec::new(),
//                    debug_name: None,
                    debug_name: None,
//                }],
                }],
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
//}
}
//

//#[test]
#[test]
//fn emit_event() {
fn emit_event() {
//    let result = run_native_program(
    let result = run_native_program(
//        &SYSCALLS_PROGRAM,
        &SYSCALLS_PROGRAM,
//        "emit_event",
        "emit_event",
//        &[],
        &[],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(SyscallHandler),
        Some(SyscallHandler),
//    );
    );
//

//    assert_eq_sorted!(
    assert_eq_sorted!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Struct {
            value: Box::new(JitValue::Struct {
//                fields: Vec::new(),
                fields: Vec::new(),
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
//}
}
//

//#[test]
#[test]
//fn send_message_to_l1() {
fn send_message_to_l1() {
//    let result = run_native_program(
    let result = run_native_program(
//        &SYSCALLS_PROGRAM,
        &SYSCALLS_PROGRAM,
//        "send_message_to_l1",
        "send_message_to_l1",
//        &[],
        &[],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(SyscallHandler),
        Some(SyscallHandler),
//    );
    );
//

//    assert_eq_sorted!(
    assert_eq_sorted!(
//        result.return_value,
        result.return_value,
//        JitValue::Enum {
        JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Struct {
            value: Box::new(JitValue::Struct {
//                fields: Vec::new(),
                fields: Vec::new(),
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
//}
}
//

//#[test]
#[test]
//fn keccak() {
fn keccak() {
//    let result = run_native_program(
    let result = run_native_program(
//        &SYSCALLS_PROGRAM,
        &SYSCALLS_PROGRAM,
//        "keccak",
        "keccak",
//        &[],
        &[],
//        Some(u128::MAX),
        Some(u128::MAX),
//        Some(SyscallHandler),
        Some(SyscallHandler),
//    );
    );
//

//    assert_eq_sorted!(
    assert_eq_sorted!(
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
//                    JitValue::Uint128(330939983442938156232262046592599923289),
                    JitValue::Uint128(330939983442938156232262046592599923289),
//                    JitValue::Uint128(288102973244655531496349286021939642254),
                    JitValue::Uint128(288102973244655531496349286021939642254),
//                ],
                ],
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
//}
}
