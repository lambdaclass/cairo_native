//! A (somewhat) usable implementation of the starknet syscall handler trait.

use std::{
    collections::{HashMap, VecDeque},
    iter::once,
};

use crate::starknet::{
    BlockInfo, ExecutionInfo, ExecutionInfoV2, Secp256k1Point, Secp256r1Point,
    StarknetSyscallHandler, SyscallResult, TxInfo, TxV2Info, U256,
};
use k256::elliptic_curve::{
    generic_array::GenericArray,
    sec1::{FromEncodedPoint, ToEncodedPoint},
};
use sec1::point::Coordinates;
use starknet_types_core::felt::Felt;
use tracing::instrument;

/// A (somewhat) usable implementation of the starknet syscall handler trait.
///
/// Currently gas is not deducted.
#[derive(Debug, Clone)]
pub struct StubSyscallHandler {
    pub storage: HashMap<(u32, Felt), Felt>,
    pub events: Vec<StubEvent>,
    pub execution_info: ExecutionInfoV2,
    pub logs: HashMap<Felt, ContractLogs>,
}

impl Default for StubSyscallHandler {
    fn default() -> Self {
        Self {
            storage: HashMap::new(),
            events: Vec::new(),
            execution_info: ExecutionInfoV2 {
                block_info: BlockInfo {
                    block_number: 0,
                    block_timestamp: 0,
                    sequencer_address: 666.into(),
                },
                tx_info: TxV2Info {
                    version: 1.into(),
                    account_contract_address: 1.into(),
                    max_fee: 0,
                    signature: vec![1.into()],
                    transaction_hash: 1.into(),
                    chain_id: 1.into(),
                    nonce: 0.into(),
                    resource_bounds: vec![],
                    tip: 0,
                    paymaster_data: vec![],
                    nonce_data_availability_mode: 0,
                    fee_data_availability_mode: 0,
                    account_deployment_data: vec![],
                },
                caller_address: 2.into(),
                contract_address: 3.into(),
                entry_point_selector: 4.into(),
            },
            logs: HashMap::new(),
        }
    }
}

/// Event emitted by the emit_event syscall.
#[derive(Debug, Clone)]
pub struct StubEvent {
    pub keys: Vec<Felt>,
    pub data: Vec<Felt>,
}

#[derive(Debug, Default, Clone)]
pub struct ContractLogs {
    pub events: VecDeque<StubEvent>,
    pub l2_to_l1_messages: VecDeque<L2ToL1Message>,
}

type L2ToL1Message = (Felt, Vec<Felt>);

impl StarknetSyscallHandler for &mut StubSyscallHandler {
    #[instrument(skip(self))]
    fn get_block_hash(
        &mut self,
        block_number: u64,
        remaining_gas: &mut u128,
    ) -> crate::starknet::SyscallResult<Felt> {
        tracing::debug!("called");
        Ok(block_number.into())
    }

    #[instrument(skip(self))]
    fn get_execution_info(
        &mut self,
        remaining_gas: &mut u128,
    ) -> crate::starknet::SyscallResult<crate::starknet::ExecutionInfo> {
        tracing::debug!("called");
        Ok(ExecutionInfo {
            block_info: self.execution_info.block_info,
            tx_info: TxInfo {
                version: self.execution_info.tx_info.version,
                account_contract_address: self.execution_info.tx_info.account_contract_address,
                max_fee: self.execution_info.tx_info.max_fee,
                signature: self.execution_info.tx_info.signature.clone(),
                transaction_hash: self.execution_info.tx_info.transaction_hash,
                chain_id: self.execution_info.tx_info.chain_id,
                nonce: self.execution_info.tx_info.nonce,
            },
            caller_address: self.execution_info.caller_address,
            contract_address: self.execution_info.contract_address,
            entry_point_selector: self.execution_info.entry_point_selector,
        })
    }

    #[instrument(skip(self))]
    fn get_execution_info_v2(
        &mut self,
        remaining_gas: &mut u128,
    ) -> crate::starknet::SyscallResult<crate::starknet::ExecutionInfoV2> {
        tracing::debug!("called");
        Ok(self.execution_info.clone())
    }

    #[instrument(skip(self))]
    fn deploy(
        &mut self,
        class_hash: Felt,
        contract_address_salt: Felt,
        calldata: &[Felt],
        deploy_from_zero: bool,
        remaining_gas: &mut u128,
    ) -> crate::starknet::SyscallResult<(Felt, Vec<Felt>)> {
        tracing::debug!("called");
        todo!()
    }

    #[instrument(skip(self))]
    fn replace_class(
        &mut self,
        class_hash: Felt,
        remaining_gas: &mut u128,
    ) -> crate::starknet::SyscallResult<()> {
        tracing::debug!("called");
        tracing::warn!("unimplemented");
        Ok(())
    }

    #[instrument(skip(self))]
    fn library_call(
        &mut self,
        class_hash: Felt,
        function_selector: Felt,
        calldata: &[Felt],
        remaining_gas: &mut u128,
    ) -> crate::starknet::SyscallResult<Vec<Felt>> {
        tracing::debug!("called");
        tracing::warn!("unimplemented");
        Ok(vec![])
    }

    #[instrument(skip(self))]
    fn call_contract(
        &mut self,
        address: Felt,
        entry_point_selector: Felt,
        calldata: &[Felt],
        remaining_gas: &mut u128,
    ) -> crate::starknet::SyscallResult<Vec<Felt>> {
        tracing::debug!("called");
        tracing::warn!("unimplemented");
        Ok(vec![])
    }

    fn storage_read(
        &mut self,
        address_domain: u32,
        address: Felt,
        _remaining_gas: &mut u128,
    ) -> crate::starknet::SyscallResult<Felt> {
        tracing::debug!("called");
        if let Some(value) = self.storage.get(&(address_domain, address)) {
            Ok(*value)
        } else {
            Err(vec![Felt::from_bytes_be_slice(b"address not found")])
        }
    }

    #[instrument(skip(self))]
    fn storage_write(
        &mut self,
        address_domain: u32,
        address: Felt,
        value: Felt,
        remaining_gas: &mut u128,
    ) -> crate::starknet::SyscallResult<()> {
        tracing::debug!("called");
        self.storage.insert((address_domain, address), value);
        Ok(())
    }

    #[instrument(skip(self))]
    fn emit_event(
        &mut self,
        keys: &[Felt],
        data: &[Felt],
        remaining_gas: &mut u128,
    ) -> crate::starknet::SyscallResult<()> {
        tracing::debug!("called");
        tracing::warn!("unimplemented but stored");
        self.events.push(StubEvent {
            keys: keys.to_vec(),
            data: data.to_vec(),
        });
        Ok(())
    }

    #[instrument(skip(self))]
    fn send_message_to_l1(
        &mut self,
        to_address: Felt,
        payload: &[Felt],
        remaining_gas: &mut u128,
    ) -> crate::starknet::SyscallResult<()> {
        tracing::debug!("called");
        tracing::warn!("unimplemented");
        Ok(())
    }

    #[instrument(skip(self))]
    fn keccak(&mut self, input: &[u64], gas: &mut u128) -> SyscallResult<U256> {
        tracing::debug!("called");
        let length = input.len();

        if length % 17 != 0 {
            let error_msg = b"Invalid keccak input size";
            let felt_error = Felt::from_bytes_be_slice(error_msg);
            return Err(vec![felt_error]);
        }

        let n_chunks = length / 17;
        let mut state = [0u64; 25];

        for i in 0..n_chunks {
            if *gas < KECCAK_ROUND_COST {
                let error_msg = b"Syscall out of gas";
                let felt_error = Felt::from_bytes_be_slice(error_msg);
                return Err(vec![felt_error]);
            }
            const KECCAK_ROUND_COST: u128 = 180000;
            *gas -= KECCAK_ROUND_COST;
            let chunk = &input[i * 17..(i + 1) * 17]; //(request.input_start + i * 17)?;
            for (i, val) in chunk.iter().enumerate() {
                state[i] ^= val;
            }
            keccak::f1600(&mut state)
        }

        // state[0] and state[1] conform the hash_high (u128)
        // state[2] and state[3] conform the hash_low (u128)
        SyscallResult::Ok(U256 {
            lo: state[0] as u128 | ((state[1] as u128) << 64),
            hi: state[2] as u128 | ((state[3] as u128) << 64),
        })
    }

    #[instrument(skip(self))]
    fn secp256k1_new(
        &mut self,
        x: U256,
        y: U256,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256k1Point>> {
        tracing::debug!("called");
        // The following unwraps should be unreachable because the iterator we provide has the
        // expected number of bytes.
        let point = k256::ProjectivePoint::from_encoded_point(
            &k256::EncodedPoint::from_affine_coordinates(
                &GenericArray::from_exact_iter(
                    x.hi.to_be_bytes().into_iter().chain(x.lo.to_be_bytes()),
                )
                .unwrap(),
                &GenericArray::from_exact_iter(
                    y.hi.to_be_bytes().into_iter().chain(y.lo.to_be_bytes()),
                )
                .unwrap(),
                false,
            ),
        );

        if bool::from(point.is_some()) {
            Ok(Some(Secp256k1Point { x, y }))
        } else {
            Ok(None)
        }
    }

    #[instrument(skip(self))]
    fn secp256k1_add(
        &mut self,
        p0: Secp256k1Point,
        p1: Secp256k1Point,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Secp256k1Point> {
        tracing::debug!("called");
        // The inner unwraps should be unreachable because the iterator we provide has the expected
        // number of bytes. The outer unwraps depend on the felt values, which should be valid since
        // they'll be provided by secp256 syscalls.
        let p0 = k256::ProjectivePoint::from_encoded_point(
            &k256::EncodedPoint::from_affine_coordinates(
                &GenericArray::from_exact_iter(
                    p0.x.hi
                        .to_be_bytes()
                        .into_iter()
                        .chain(p0.x.lo.to_be_bytes()),
                )
                .unwrap(),
                &GenericArray::from_exact_iter(
                    p0.y.hi
                        .to_be_bytes()
                        .into_iter()
                        .chain(p0.y.lo.to_be_bytes()),
                )
                .unwrap(),
                false,
            ),
        )
        .unwrap();
        let p1 = k256::ProjectivePoint::from_encoded_point(
            &k256::EncodedPoint::from_affine_coordinates(
                &GenericArray::from_exact_iter(
                    p1.x.hi
                        .to_be_bytes()
                        .into_iter()
                        .chain(p1.x.lo.to_be_bytes()),
                )
                .unwrap(),
                &GenericArray::from_exact_iter(
                    p1.y.hi
                        .to_be_bytes()
                        .into_iter()
                        .chain(p1.y.lo.to_be_bytes()),
                )
                .unwrap(),
                false,
            ),
        )
        .unwrap();

        let p = p0 + p1;

        let p = p.to_encoded_point(false);
        let (x, y) = match p.coordinates() {
            Coordinates::Uncompressed { x, y } => (x, y),
            _ => {
                // This should be unreachable because we explicitly asked for the uncompressed
                // encoding.
                unreachable!()
            }
        };

        // The following two unwraps should be safe because the array always has 32 bytes. The other
        // four are definitely safe because the slicing guarantees its length to be the right one.
        let x: [u8; 32] = x.as_slice().try_into().unwrap();
        let y: [u8; 32] = y.as_slice().try_into().unwrap();
        Ok(Secp256k1Point {
            x: U256 {
                hi: u128::from_be_bytes(x[0..16].try_into().unwrap()),
                lo: u128::from_be_bytes(x[16..32].try_into().unwrap()),
            },
            y: U256 {
                hi: u128::from_be_bytes(y[0..16].try_into().unwrap()),
                lo: u128::from_be_bytes(y[16..32].try_into().unwrap()),
            },
        })
    }

    #[instrument(skip(self))]
    fn secp256k1_mul(
        &mut self,
        p: Secp256k1Point,
        m: U256,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Secp256k1Point> {
        // The inner unwrap should be unreachable because the iterator we provide has the expected
        // number of bytes. The outer unwrap depends on the felt values, which should be valid since
        // they'll be provided by secp256 syscalls.
        let p = k256::ProjectivePoint::from_encoded_point(
            &k256::EncodedPoint::from_affine_coordinates(
                &GenericArray::from_exact_iter(
                    p.x.hi.to_be_bytes().into_iter().chain(p.x.lo.to_be_bytes()),
                )
                .unwrap(),
                &GenericArray::from_exact_iter(
                    p.y.hi.to_be_bytes().into_iter().chain(p.y.lo.to_be_bytes()),
                )
                .unwrap(),
                false,
            ),
        )
        .unwrap();
        let m: k256::Scalar = k256::elliptic_curve::ScalarPrimitive::from_slice(&{
            let mut buf = [0u8; 32];
            buf[0..16].copy_from_slice(&m.hi.to_be_bytes());
            buf[16..32].copy_from_slice(&m.lo.to_be_bytes());
            buf
        })
        .map_err(|_| {
            vec![Felt::from_bytes_be(
                b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0invalid scalar",
            )]
        })?
        .into();

        let p = p * m;

        let p = p.to_encoded_point(false);
        let (x, y) = match p.coordinates() {
            Coordinates::Uncompressed { x, y } => (x, y),
            _ => {
                // This should be unreachable because we explicitly asked for the uncompressed
                // encoding.
                unreachable!()
            }
        };

        // The following two unwraps should be safe because the array always has 32 bytes. The other
        // four are definitely safe because the slicing guarantees its length to be the right one.
        let x: [u8; 32] = x.as_slice().try_into().unwrap();
        let y: [u8; 32] = y.as_slice().try_into().unwrap();
        Ok(Secp256k1Point {
            x: U256 {
                hi: u128::from_be_bytes(x[0..16].try_into().unwrap()),
                lo: u128::from_be_bytes(x[16..32].try_into().unwrap()),
            },
            y: U256 {
                hi: u128::from_be_bytes(y[0..16].try_into().unwrap()),
                lo: u128::from_be_bytes(y[16..32].try_into().unwrap()),
            },
        })
    }

    #[instrument(skip(self))]
    fn secp256k1_get_point_from_x(
        &mut self,
        x: U256,
        y_parity: bool,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256k1Point>> {
        tracing::debug!("called");
        // The inner unwrap should be unreachable because the iterator we provide has the expected
        // number of bytes. The outer unwrap depends on the encoding format, which should be valid
        // since it's hardcoded..
        let point = k256::ProjectivePoint::from_encoded_point(
            &k256::EncodedPoint::from_bytes(
                k256::CompressedPoint::from_exact_iter(
                    once(0x02 | y_parity as u8)
                        .chain(x.hi.to_be_bytes())
                        .chain(x.lo.to_be_bytes()),
                )
                .unwrap(),
            )
            .unwrap(),
        );

        if bool::from(point.is_some()) {
            // This unwrap has already been checked in the `if` expression's condition.
            let p = point.unwrap();

            let p = p.to_encoded_point(false);
            let y = match p.coordinates() {
                Coordinates::Uncompressed { y, .. } => y,
                _ => {
                    // This should be unreachable because we explicitly asked for the uncompressed
                    // encoding.
                    unreachable!()
                }
            };

            // The following unwrap should be safe because the array always has 32 bytes. The other
            // two are definitely safe because the slicing guarantees its length to be the right
            // one.
            let y: [u8; 32] = y.as_slice().try_into().unwrap();
            Ok(Some(Secp256k1Point {
                x,
                y: U256 {
                    hi: u128::from_be_bytes(y[0..16].try_into().unwrap()),
                    lo: u128::from_be_bytes(y[16..32].try_into().unwrap()),
                },
            }))
        } else {
            Ok(None)
        }
    }

    #[instrument(skip(self))]
    fn secp256k1_get_xy(
        &mut self,
        p: Secp256k1Point,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<(U256, U256)> {
        tracing::debug!("called");
        Ok((p.x, p.y))
    }

    #[instrument(skip(self))]
    fn secp256r1_new(
        &mut self,
        x: U256,
        y: U256,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256r1Point>> {
        tracing::debug!("called");
        // The following unwraps should be unreachable because the iterator we provide has the
        // expected number of bytes.
        let point = p256::ProjectivePoint::from_encoded_point(
            &k256::EncodedPoint::from_affine_coordinates(
                &GenericArray::from_exact_iter(
                    x.hi.to_be_bytes().into_iter().chain(x.lo.to_be_bytes()),
                )
                .unwrap(),
                &GenericArray::from_exact_iter(
                    y.hi.to_be_bytes().into_iter().chain(y.lo.to_be_bytes()),
                )
                .unwrap(),
                false,
            ),
        );

        if bool::from(point.is_some()) {
            Ok(Some(Secp256r1Point { x, y }))
        } else {
            Ok(None)
        }
    }

    #[instrument(skip(self))]
    fn secp256r1_add(
        &mut self,
        p0: Secp256r1Point,
        p1: Secp256r1Point,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Secp256r1Point> {
        tracing::debug!("called");
        // The inner unwraps should be unreachable because the iterator we provide has the expected
        // number of bytes. The outer unwraps depend on the felt values, which should be valid since
        // they'll be provided by secp256 syscalls.
        let p0 = p256::ProjectivePoint::from_encoded_point(
            &p256::EncodedPoint::from_affine_coordinates(
                &GenericArray::from_exact_iter(
                    p0.x.hi
                        .to_be_bytes()
                        .into_iter()
                        .chain(p0.x.lo.to_be_bytes()),
                )
                .unwrap(),
                &GenericArray::from_exact_iter(
                    p0.y.hi
                        .to_be_bytes()
                        .into_iter()
                        .chain(p0.y.lo.to_be_bytes()),
                )
                .unwrap(),
                false,
            ),
        )
        .unwrap();
        let p1 = p256::ProjectivePoint::from_encoded_point(
            &p256::EncodedPoint::from_affine_coordinates(
                &GenericArray::from_exact_iter(
                    p1.x.hi
                        .to_be_bytes()
                        .into_iter()
                        .chain(p1.x.lo.to_be_bytes()),
                )
                .unwrap(),
                &GenericArray::from_exact_iter(
                    p1.y.hi
                        .to_be_bytes()
                        .into_iter()
                        .chain(p1.y.lo.to_be_bytes()),
                )
                .unwrap(),
                false,
            ),
        )
        .unwrap();

        let p = p0 + p1;

        let p = p.to_encoded_point(false);
        let (x, y) = match p.coordinates() {
            Coordinates::Uncompressed { x, y } => (x, y),
            _ => {
                // This should be unreachable because we explicitly asked for the uncompressed
                // encoding.
                unreachable!()
            }
        };

        // The following two unwraps should be safe because the array always has 32 bytes. The other
        // four are definitely safe because the slicing guarantees its length to be the right one.
        let x: [u8; 32] = x.as_slice().try_into().unwrap();
        let y: [u8; 32] = y.as_slice().try_into().unwrap();
        Ok(Secp256r1Point {
            x: U256 {
                hi: u128::from_be_bytes(x[0..16].try_into().unwrap()),
                lo: u128::from_be_bytes(x[16..32].try_into().unwrap()),
            },
            y: U256 {
                hi: u128::from_be_bytes(y[0..16].try_into().unwrap()),
                lo: u128::from_be_bytes(y[16..32].try_into().unwrap()),
            },
        })
    }

    #[instrument(skip(self))]
    fn secp256r1_mul(
        &mut self,
        p: Secp256r1Point,
        m: U256,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Secp256r1Point> {
        // The inner unwrap should be unreachable because the iterator we provide has the expected
        // number of bytes. The outer unwrap depends on the felt values, which should be valid since
        // they'll be provided by secp256 syscalls.
        let p = p256::ProjectivePoint::from_encoded_point(
            &p256::EncodedPoint::from_affine_coordinates(
                &GenericArray::from_exact_iter(
                    p.x.hi.to_be_bytes().into_iter().chain(p.x.lo.to_be_bytes()),
                )
                .unwrap(),
                &GenericArray::from_exact_iter(
                    p.y.hi.to_be_bytes().into_iter().chain(p.y.lo.to_be_bytes()),
                )
                .unwrap(),
                false,
            ),
        )
        .unwrap();
        let m: p256::Scalar = p256::elliptic_curve::ScalarPrimitive::from_slice(&{
            let mut buf = [0u8; 32];
            buf[0..16].copy_from_slice(&m.hi.to_be_bytes());
            buf[16..32].copy_from_slice(&m.lo.to_be_bytes());
            buf
        })
        .map_err(|_| {
            vec![Felt::from_bytes_be(
                b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0invalid scalar",
            )]
        })?
        .into();

        let p = p * m;
        let p = p.to_encoded_point(false);
        let (x, y) = match p.coordinates() {
            Coordinates::Uncompressed { x, y } => (x, y),
            _ => {
                // This should be unreachable because we explicitly asked for the uncompressed
                // encoding.
                unreachable!()
            }
        };

        // The following two unwraps should be safe because the array always has 32 bytes. The other
        // four are definitely safe because the slicing guarantees its length to be the right one.
        let x: [u8; 32] = x.as_slice().try_into().unwrap();
        let y: [u8; 32] = y.as_slice().try_into().unwrap();
        Ok(Secp256r1Point {
            x: U256 {
                hi: u128::from_be_bytes(x[0..16].try_into().unwrap()),
                lo: u128::from_be_bytes(x[16..32].try_into().unwrap()),
            },
            y: U256 {
                hi: u128::from_be_bytes(y[0..16].try_into().unwrap()),
                lo: u128::from_be_bytes(y[16..32].try_into().unwrap()),
            },
        })
    }

    #[instrument(skip(self))]
    fn secp256r1_get_point_from_x(
        &mut self,
        x: U256,
        y_parity: bool,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256r1Point>> {
        let point = p256::ProjectivePoint::from_encoded_point(
            &p256::EncodedPoint::from_bytes(
                p256::CompressedPoint::from_exact_iter(
                    once(0x02 | y_parity as u8)
                        .chain(x.hi.to_be_bytes())
                        .chain(x.lo.to_be_bytes()),
                )
                .unwrap(),
            )
            .unwrap(),
        );

        if bool::from(point.is_some()) {
            let p = point.unwrap();

            let p = p.to_encoded_point(false);
            let y = match p.coordinates() {
                Coordinates::Uncompressed { y, .. } => y,
                _ => unreachable!(),
            };

            let y: [u8; 32] = y.as_slice().try_into().unwrap();
            Ok(Some(Secp256r1Point {
                x,
                y: U256 {
                    hi: u128::from_be_bytes(y[0..16].try_into().unwrap()),
                    lo: u128::from_be_bytes(y[16..32].try_into().unwrap()),
                },
            }))
        } else {
            Ok(None)
        }
    }

    #[instrument(skip(self))]
    fn secp256r1_get_xy(
        &mut self,
        p: Secp256r1Point,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<(U256, U256)> {
        tracing::debug!("called");
        Ok((p.x, p.y))
    }

    #[cfg(feature = "with-cheatcode")]
    #[instrument(skip(self))]
    fn cheatcode(&mut self, selector: Felt, input: &[Felt]) -> Vec<Felt> {
        tracing::debug!("called");
        let selector_bytes = selector.to_bytes_be();

        let selector = match std::str::from_utf8(&selector_bytes) {
            Ok(selector) => selector.trim_start_matches('\0'),
            Err(_) => return Vec::new(),
        };

        match selector {
            "set_sequencer_address" => {
                self.execution_info.block_info.sequencer_address = input[0];
                vec![]
            }
            "set_caller_address" => {
                self.execution_info.caller_address = input[0];
                vec![]
            }
            "set_contract_address" => {
                self.execution_info.contract_address = input[0];
                vec![]
            }
            "set_account_contract_address" => {
                self.execution_info.tx_info.account_contract_address = input[0];
                vec![]
            }
            "set_transaction_hash" => {
                self.execution_info.tx_info.transaction_hash = input[0];
                vec![]
            }
            "set_nonce" => {
                self.execution_info.tx_info.nonce = input[0];
                vec![]
            }
            "set_version" => {
                self.execution_info.tx_info.version = input[0];
                vec![]
            }
            "set_chain_id" => {
                self.execution_info.tx_info.chain_id = input[0];
                vec![]
            }
            "set_max_fee" => {
                let max_fee = input[0].to_biguint().try_into().unwrap();
                self.execution_info.tx_info.max_fee = max_fee;
                vec![]
            }
            "set_block_number" => {
                let block_number = input[0].to_biguint().try_into().unwrap();
                self.execution_info.block_info.block_number = block_number;
                vec![]
            }
            "set_block_timestamp" => {
                let block_timestamp = input[0].to_biguint().try_into().unwrap();
                self.execution_info.block_info.block_timestamp = block_timestamp;
                vec![]
            }
            "set_signature" => {
                self.execution_info.tx_info.signature = input.to_vec();
                vec![]
            }
            "pop_log" => self
                .logs
                .get_mut(&input[0])
                .and_then(|logs| logs.events.pop_front())
                .map(|mut log| {
                    let mut serialized_log = Vec::new();
                    serialized_log.push(log.keys.len().into());
                    serialized_log.append(&mut log.keys);
                    serialized_log.push(log.data.len().into());
                    serialized_log.append(&mut log.data);
                    serialized_log
                })
                .unwrap_or_default(),
            "pop_l2_to_l1_message" => self
                .logs
                .get_mut(&input[0])
                .and_then(|logs| logs.l2_to_l1_messages.pop_front())
                .map(|mut log| {
                    let mut serialized_log = Vec::new();
                    serialized_log.push(log.0);
                    serialized_log.push(log.1.len().into());
                    serialized_log.append(&mut log.1);
                    serialized_log
                })
                .unwrap_or_default(),
            _ => vec![],
        }
    }

    fn sha256_process_block(
        &mut self,
        state: &mut [u32; 8],
        block: &[u32; 16],
        _remaining_gas: &mut u128,
    ) -> SyscallResult<()> {
        // reference impl
        // https://github.com/starkware-libs/cairo/blob/ba3f82b4a09972b6a24bf791e344cabce579bf69/crates/cairo-lang-runner/src/casm_run/mod.rs#L1292
        let data_as_bytes = sha2::digest::generic_array::GenericArray::from_exact_iter(
            block.iter().flat_map(|x| x.to_be_bytes()),
        )
        .unwrap();
        sha2::compress256(state, &[data_as_bytes]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secp256k1_get_xy() {
        let p = Secp256k1Point {
            x: U256 {
                hi: 331229800296699308591929724809569456681,
                lo: 240848751772479376198639683648735950585,
            },
            y: U256 {
                hi: 75181762170223969696219813306313470806,
                lo: 134255467439736302886468555755295925874,
            },
        };

        let mut test_syscall_handler = StubSyscallHandler::default();
        let mut test_syscall_handler = &mut test_syscall_handler;

        assert_eq!(
            test_syscall_handler.secp256k1_get_xy(p, &mut 10).unwrap(),
            (
                U256 {
                    hi: 331229800296699308591929724809569456681,
                    lo: 240848751772479376198639683648735950585,
                },
                U256 {
                    hi: 75181762170223969696219813306313470806,
                    lo: 134255467439736302886468555755295925874,
                }
            )
        )
    }

    #[test]
    fn test_secp256k1_secp256k1_new() {
        let mut test_syscall_handler = StubSyscallHandler::default();
        let mut test_syscall_handler = &mut test_syscall_handler;

        let x = U256 {
            lo: 330631467365974629050427735731901850225,
            hi: 97179038819393695679,
        };
        let y = U256 {
            lo: 68974579539311638391577168388077592842,
            hi: 26163136114030451075775058782541084873,
        };

        assert_eq!(
            test_syscall_handler.secp256k1_new(x, y, &mut 10).unwrap(),
            Some(Secp256k1Point { x, y })
        );
    }

    #[test]
    fn test_secp256k1_secp256k1_new_none() {
        let mut test_syscall_handler = StubSyscallHandler::default();
        let mut test_syscall_handler = &mut test_syscall_handler;

        let x = U256 {
            hi: 330631467365974629050427735731901850225,
            lo: 97179038819393695679,
        };
        let y = U256 { hi: 0, lo: 0 };

        assert!(test_syscall_handler
            .secp256k1_new(x, y, &mut 10)
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_secp256k1_ssecp256k1_add() {
        let mut test_syscall_handler = StubSyscallHandler::default();
        let mut test_syscall_handler = &mut test_syscall_handler;

        let p1 = Secp256k1Point {
            x: U256 {
                lo: 3468390537006497937951914270391801752,
                hi: 161825202758953104525843685720298294023,
            },
            y: U256 {
                lo: 336417762351022071123394393598455764152,
                hi: 96009999919712310848645357523629574312,
            },
        };

        let p2 = p1;

        // 2 * P1
        let p3 = test_syscall_handler.secp256k1_add(p1, p2, &mut 10).unwrap();

        let p1_double = Secp256k1Point {
            x: U256 {
                lo: 122909745026270932982812610085084241637,
                hi: 263210499965038831386353541518668627160,
            },
            y: U256 {
                lo: 329597642124196932058042157271922763050,
                hi: 35730324229579385338853513728577301230,
            },
        };
        assert_eq!(p3, p1_double);
        assert_eq!(
            test_syscall_handler
                .secp256k1_mul(p1, U256 { lo: 2, hi: 0 }, &mut 10)
                .unwrap(),
            p1_double
        );

        // 3 * P1
        let three_p1 = Secp256k1Point {
            x: U256 {
                lo: 240848751772479376198639683648735950585,
                hi: 331229800296699308591929724809569456681,
            },
            y: U256 {
                lo: 134255467439736302886468555755295925874,
                hi: 75181762170223969696219813306313470806,
            },
        };
        assert_eq!(
            test_syscall_handler.secp256k1_add(p1, p3, &mut 10).unwrap(),
            three_p1
        );
        assert_eq!(
            test_syscall_handler
                .secp256k1_mul(p1, U256 { lo: 3, hi: 0 }, &mut 10)
                .unwrap(),
            three_p1
        );
    }

    #[test]
    fn test_secp256k1_get_point_from_x_false_yparity() {
        let mut test_syscall_handler = StubSyscallHandler::default();
        let mut test_syscall_handler = &mut test_syscall_handler;

        assert_eq!(
            test_syscall_handler
                .secp256k1_get_point_from_x(
                    U256 {
                        lo: 330631467365974629050427735731901850225,
                        hi: 97179038819393695679,
                    },
                    false,
                    &mut 10
                )
                .unwrap()
                .unwrap(),
            Secp256k1Point {
                x: U256 {
                    lo: 330631467365974629050427735731901850225,
                    hi: 97179038819393695679,
                },
                y: U256 {
                    lo: 68974579539311638391577168388077592842,
                    hi: 26163136114030451075775058782541084873,
                },
            }
        );
    }

    #[test]
    fn test_secp256k1_get_point_from_x_true_yparity() {
        let mut test_syscall_handler = StubSyscallHandler::default();
        let mut test_syscall_handler = &mut test_syscall_handler;

        assert_eq!(
            test_syscall_handler
                .secp256k1_get_point_from_x(
                    U256 {
                        lo: 330631467365974629050427735731901850225,
                        hi: 97179038819393695679,
                    },
                    true,
                    &mut 10
                )
                .unwrap()
                .unwrap(),
            Secp256k1Point {
                x: U256 {
                    lo: 330631467365974629050427735731901850225,
                    hi: 97179038819393695679,
                },
                y: U256 {
                    lo: 271307787381626825071797439039395650341,
                    hi: 314119230806908012387599548649227126582,
                },
            }
        );
    }

    #[test]
    fn test_secp256k1_get_point_from_x_none() {
        let mut test_syscall_handler = StubSyscallHandler::default();
        let mut test_syscall_handler = &mut test_syscall_handler;

        assert!(test_syscall_handler
            .secp256k1_get_point_from_x(U256 { hi: 0, lo: 0 }, true, &mut 10)
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_secp256r1_new() {
        let mut test_syscall_handler = StubSyscallHandler::default();
        let mut test_syscall_handler = &mut test_syscall_handler;

        let x = U256 {
            lo: 330631467365974629050427735731901850225,
            hi: 97179038819393695679,
        };
        let y = U256 {
            lo: 111045440647474106186537215379882575585,
            hi: 118910939004298029402109603132816090461,
        };

        assert_eq!(
            test_syscall_handler
                .secp256r1_new(x, y, &mut 10)
                .unwrap()
                .unwrap(),
            Secp256r1Point { x, y }
        );
    }

    #[test]
    fn test_secp256r1_new_none() {
        let mut test_syscall_handler = StubSyscallHandler::default();
        let mut test_syscall_handler = &mut test_syscall_handler;

        let x = U256 { hi: 0, lo: 0 };
        let y = U256 { hi: 0, lo: 0 };

        assert!(test_syscall_handler
            .secp256r1_new(x, y, &mut 10)
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_secp256r1_add() {
        let mut test_syscall_handler = StubSyscallHandler::default();
        let mut test_syscall_handler = &mut test_syscall_handler;

        let p1 = Secp256r1Point {
            x: U256 {
                lo: 330631467365974629050427735731901850225,
                hi: 97179038819393695679,
            },
            y: U256 {
                lo: 111045440647474106186537215379882575585,
                hi: 118910939004298029402109603132816090461,
            },
        };

        let p2 = p1;

        // 2 * P1
        let p3 = test_syscall_handler.secp256r1_add(p1, p2, &mut 10).unwrap();

        let p1_double = Secp256r1Point {
            x: U256 {
                lo: 309339945874468445579793098896656960879,
                hi: 280079427190737520201067412903899817878,
            },
            y: U256 {
                lo: 231570843221643745062297421862629788481,
                hi: 84249534056490759701994051847937833933,
            },
        };
        assert_eq!(p3, p1_double);
        assert_eq!(
            test_syscall_handler
                .secp256r1_mul(p1, U256 { lo: 2, hi: 0 }, &mut 10)
                .unwrap(),
            p1_double
        );

        // 3 * P1
        let three_p1 = Secp256r1Point {
            x: U256 {
                lo: 195259625777021303662291420857740525307,
                hi: 23850518908906170876551962912581992002,
            },
            y: U256 {
                lo: 282344931843342117515389970197013120959,
                hi: 178681203065513270100417145499857169664,
            },
        };
        assert_eq!(
            test_syscall_handler.secp256r1_add(p1, p3, &mut 10).unwrap(),
            three_p1
        );
        assert_eq!(
            test_syscall_handler
                .secp256r1_mul(p1, U256 { lo: 3, hi: 0 }, &mut 10)
                .unwrap(),
            three_p1
        );
    }

    #[test]
    fn test_secp256r1_get_point_from_x_true_yparity() {
        let mut test_syscall_handler = StubSyscallHandler::default();
        let mut test_syscall_handler = &mut test_syscall_handler;

        let x = U256 {
            lo: 330631467365974629050427735731901850225,
            hi: 97179038819393695679,
        };

        let y = U256 {
            lo: 111045440647474106186537215379882575585,
            hi: 118910939004298029402109603132816090461,
        };

        assert_eq!(
            test_syscall_handler
                .secp256r1_get_point_from_x(x, true, &mut 10)
                .unwrap()
                .unwrap(),
            Secp256r1Point { x, y }
        );
    }

    #[test]
    fn test_secp256r1_get_point_from_x_false_yparity() {
        let mut test_syscall_handler = StubSyscallHandler::default();
        let mut test_syscall_handler = &mut test_syscall_handler;

        let x = U256 {
            lo: 330631467365974629050427735731901850225,
            hi: 97179038819393695679,
        };

        let y = U256 {
            lo: 229236926352692519791101729645429586206,
            hi: 221371427837412271565447410779117722274,
        };

        assert_eq!(
            test_syscall_handler
                .secp256r1_get_point_from_x(x, false, &mut 10)
                .unwrap()
                .unwrap(),
            Secp256r1Point { x, y }
        );
    }

    #[test]
    fn test_secp256r1_get_point_from_x_none() {
        let mut test_syscall_handler = StubSyscallHandler::default();
        let mut test_syscall_handler = &mut test_syscall_handler;

        let x = U256 { lo: 0, hi: 10 };

        assert!(test_syscall_handler
            .secp256r1_get_point_from_x(x, true, &mut 10)
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_secp256r1_get_xy() {
        let p = Secp256r1Point {
            x: U256 {
                lo: 97179038819393695679,
                hi: 330631467365974629050427735731901850225,
            },
            y: U256 {
                lo: 221371427837412271565447410779117722274,
                hi: 229236926352692519791101729645429586206,
            },
        };

        let mut test_syscall_handler = StubSyscallHandler::default();
        let mut test_syscall_handler = &mut test_syscall_handler;

        assert_eq!(
            test_syscall_handler.secp256r1_get_xy(p, &mut 10).unwrap(),
            (
                U256 {
                    lo: 97179038819393695679,
                    hi: 330631467365974629050427735731901850225,
                },
                U256 {
                    lo: 221371427837412271565447410779117722274,
                    hi: 229236926352692519791101729645429586206,
                }
            )
        )
    }
}
