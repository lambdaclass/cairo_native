//! A (somewhat) usable implementation of the starknet syscall handler trait.

use std::{
    collections::{HashMap, VecDeque},
    fmt,
};

use crate::starknet::{
    ExecutionInfo, ExecutionInfoV2, Secp256k1Point, Secp256r1Point, StarknetSyscallHandler,
    SyscallResult, TxV2Info, U256,
};
use ark_ec::short_weierstrass::{Affine, Projective, SWCurveConfig};
use ark_ff::{BigInt, PrimeField};
use itertools::Itertools;
use num_bigint::BigUint;
use num_traits::Zero;
use starknet_types_core::felt::Felt;
use tracing::instrument;

/// A (somewhat) usable implementation of the starknet syscall handler trait.
///
/// Currently gas is not deducted.
#[derive(Clone, Default)]
pub struct StubSyscallHandler {
    /// The values of addresses in the simulated storage per contract.
    pub storage: HashMap<Felt, HashMap<Felt, Felt>>,
    /// A mapping from contract address to class hash.
    pub deployed_contracts: HashMap<Felt, Felt>,
    /// A mapping from contract address to logs.
    pub logs: HashMap<Felt, ContractLogs>,
    /// The simulated execution info.
    pub execution_info: ExecutionInfo,
    /// A mock history, mapping block number to the class hash.
    pub block_hash: HashMap<u64, Felt>,
}

type Log = (Vec<Felt>, Vec<Felt>);
type L2ToL1Message = (Felt, Vec<Felt>);

#[derive(Debug, Default, Clone)]
pub struct ContractLogs {
    pub events: VecDeque<Log>,
    pub l2_to_l1_messages: VecDeque<L2ToL1Message>,
}

#[derive(PartialEq, Clone, Copy)]
struct Secp256Point<Curve: SWCurveConfig>(Affine<Curve>);

impl<Curve: SWCurveConfig> fmt::Debug for Secp256Point<Curve> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Secp256Point").field(&self.0).finish()
    }
}

impl From<Secp256Point<ark_secp256k1::Config>> for Secp256k1Point {
    fn from(Secp256Point(Affine { x, y, infinity }): Secp256Point<ark_secp256k1::Config>) -> Self {
        Secp256k1Point {
            x: big4int_to_u256(x.into()),
            y: big4int_to_u256(y.into()),
            is_infinity: infinity,
        }
    }
}

impl From<Secp256Point<ark_secp256r1::Config>> for Secp256r1Point {
    fn from(Secp256Point(Affine { x, y, infinity }): Secp256Point<ark_secp256r1::Config>) -> Self {
        Secp256r1Point {
            x: big4int_to_u256(x.into()),
            y: big4int_to_u256(y.into()),
            is_infinity: infinity,
        }
    }
}

impl From<Secp256k1Point> for Secp256Point<ark_secp256k1::Config> {
    fn from(p: Secp256k1Point) -> Self {
        Secp256Point(Affine {
            x: u256_to_biguint(p.x).into(),
            y: u256_to_biguint(p.y).into(),
            infinity: p.is_infinity,
        })
    }
}

impl From<Secp256r1Point> for Secp256Point<ark_secp256r1::Config> {
    fn from(p: Secp256r1Point) -> Self {
        Secp256Point(Affine {
            x: u256_to_biguint(p.x).into(),
            y: u256_to_biguint(p.y).into(),
            infinity: p.is_infinity,
        })
    }
}

pub fn u256_to_biguint(u256: U256) -> BigUint {
    let lo = BigUint::from(u256.lo);
    let hi = BigUint::from(u256.hi);

    (hi << 128) + lo
}

pub fn big4int_to_u256(b_int: BigInt<4>) -> U256 {
    let [a, b, c, d] = b_int.0;

    let lo = u128::from(a) | (u128::from(b) << 64);
    let hi = u128::from(c) | (u128::from(d) << 64);

    U256 { lo, hi }
}

pub fn encode_str_as_felts(msg: &str) -> Vec<Felt> {
    const CHUNK_SIZE: usize = 32;

    let data = msg.as_bytes().chunks(CHUNK_SIZE - 1);
    let mut encoding = vec![Felt::default(); data.len()];
    for (i, data_chunk) in data.enumerate() {
        let mut chunk = [0_u8; CHUNK_SIZE];
        chunk[1..data_chunk.len() + 1].copy_from_slice(data_chunk);
        encoding[i] = Felt::from_bytes_be(&chunk);
    }
    encoding
}

pub fn decode_felts_as_str(encoding: &[Felt]) -> String {
    let bytes_err: Vec<_> = encoding
        .iter()
        .flat_map(|felt| felt.to_bytes_be()[1..32].to_vec())
        .collect();

    match String::from_utf8(bytes_err) {
        Ok(s) => s.trim_matches('\0').to_owned(),
        Err(_) => {
            let err_msgs = encoding
                .iter()
                .map(
                    |felt| match String::from_utf8(felt.to_bytes_be()[1..32].to_vec()) {
                        Ok(s) => format!("{} ({})", s.trim_matches('\0'), felt),
                        Err(_) => felt.to_string(),
                    },
                )
                .join(", ");
            format!("[{}]", err_msgs)
        }
    }
}

impl<Curve: SWCurveConfig> Secp256Point<Curve>
where
    Curve::BaseField: PrimeField, // constraint for get_point_by_id
{
    // Given a (x,y) pair it will
    // - return the point at infinity for (0,0)
    // - Err if either x or y is outside of the modulus
    // - Ok(None) if (x,y) are within the modules but not on the curve
    // - Ok(Some(Point)) if (x,y) are on the curve
    fn new(x: U256, y: U256) -> Result<Option<Self>, Vec<Felt>> {
        let x = u256_to_biguint(x);
        let y = u256_to_biguint(y);
        let modulos = Curve::BaseField::MODULUS.into();

        if x >= modulos || y >= modulos {
            let error = Felt::from_hex(
                "0x00000000000000000000000000000000496e76616c696420617267756d656e74",
            ) // INVALID_ARGUMENT
            .map_err(|err| encode_str_as_felts(&err.to_string()))?;

            return Err(vec![error]);
        }

        Ok(maybe_affine(x.into(), y.into()))
    }

    fn add(p0: Self, p1: Self) -> Self {
        let result: Projective<Curve> = p0.0 + p1.0;
        Secp256Point(result.into())
    }

    fn mul(p: Self, m: U256) -> Self {
        let result = p.0 * Curve::ScalarField::from(u256_to_biguint(m));
        Secp256Point(result.into())
    }

    fn get_point_from_x(x: U256, y_parity: bool) -> Result<Option<Self>, Vec<Felt>> {
        let modulos = Curve::BaseField::MODULUS.into();
        let x = u256_to_biguint(x);

        if x >= modulos {
            let error = Felt::from_hex(
                "0x00000000000000000000000000000000496e76616c696420617267756d656e74",
            ) // INVALID_ARGUMENT
            .map_err(|err| encode_str_as_felts(&err.to_string()))?;

            return Err(vec![error]);
        }

        let x = x.into();
        let maybe_ec_point = Affine::<Curve>::get_ys_from_x_unchecked(x)
            .map(|(smaller, greater)| {
                // Return the correct y coordinate based on the parity.
                if ark_ff::BigInteger::is_odd(&smaller.into_bigint()) == y_parity {
                    smaller
                } else {
                    greater
                }
            })
            .map(|y| Affine::<Curve>::new_unchecked(x, y))
            .filter(|p| p.is_in_correct_subgroup_assuming_on_curve());

        Ok(maybe_ec_point.map(Secp256Point))
    }
}

/// Variation on [`Affine<Curve>::new`] that doesn't panic and maps (x,y) = (0,0) -> infinity
fn maybe_affine<Curve: SWCurveConfig>(
    x: Curve::BaseField,
    y: Curve::BaseField,
) -> Option<Secp256Point<Curve>> {
    let ec_point = if x.is_zero() && y.is_zero() {
        Affine::<Curve>::identity()
    } else {
        Affine::<Curve>::new_unchecked(x, y)
    };

    if ec_point.is_on_curve() && ec_point.is_in_correct_subgroup_assuming_on_curve() {
        Some(Secp256Point(ec_point))
    } else {
        None
    }
}

impl StarknetSyscallHandler for &mut StubSyscallHandler {
    #[instrument(skip(self))]
    fn get_block_hash(
        &mut self,
        block_number: u64,
        remaining_gas: &mut u64,
    ) -> crate::starknet::SyscallResult<Felt> {
        tracing::debug!("called");
        if let Some(block_hash) = self.block_hash.get(&block_number) {
            Ok(*block_hash)
        } else {
            Err(vec![Felt::from_bytes_be_slice(b"GET_BLOCK_HASH_NOT_SET")])
        }
    }

    #[instrument(skip(self))]
    fn get_execution_info(
        &mut self,
        remaining_gas: &mut u64,
    ) -> crate::starknet::SyscallResult<crate::starknet::ExecutionInfo> {
        tracing::debug!("called");
        Ok(self.execution_info.clone())
    }

    #[instrument(skip(self))]
    fn get_execution_info_v2(
        &mut self,
        remaining_gas: &mut u64,
    ) -> crate::starknet::SyscallResult<crate::starknet::ExecutionInfoV2> {
        tracing::debug!("called");
        Ok(ExecutionInfoV2 {
            block_info: self.execution_info.block_info,
            tx_info: TxV2Info {
                version: self.execution_info.tx_info.version,
                account_contract_address: self.execution_info.tx_info.account_contract_address,
                max_fee: self.execution_info.tx_info.max_fee,
                signature: self.execution_info.tx_info.signature.clone(),
                transaction_hash: self.execution_info.tx_info.transaction_hash,
                chain_id: self.execution_info.tx_info.chain_id,
                nonce: self.execution_info.tx_info.nonce,
                ..TxV2Info::default()
            },
            caller_address: self.execution_info.caller_address,
            contract_address: self.execution_info.contract_address,
            entry_point_selector: self.execution_info.entry_point_selector,
        })
    }

    #[instrument(skip(self))]
    fn deploy(
        &mut self,
        class_hash: Felt,
        contract_address_salt: Felt,
        calldata: &[Felt],
        deploy_from_zero: bool,
        remaining_gas: &mut u64,
    ) -> crate::starknet::SyscallResult<(Felt, Vec<Felt>)> {
        tracing::debug!("called");
        tracing::warn!("unimplemented");
        Err(vec![Felt::from_bytes_be_slice(b"deploy unimplemented")])
    }

    #[instrument(skip(self))]
    fn replace_class(
        &mut self,
        class_hash: Felt,
        remaining_gas: &mut u64,
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
        remaining_gas: &mut u64,
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
        remaining_gas: &mut u64,
    ) -> crate::starknet::SyscallResult<Vec<Felt>> {
        tracing::debug!("called");
        tracing::warn!("unimplemented");
        Ok(vec![])
    }

    fn storage_read(
        &mut self,
        address_domain: u32,
        address: Felt,
        _remaining_gas: &mut u64,
    ) -> crate::starknet::SyscallResult<Felt> {
        tracing::debug!("called");
        if !address_domain.is_zero() {
            // Only address_domain 0 is currently supported.
            return Err(vec![Felt::from_bytes_be_slice(
                b"Unsupported address domain",
            )]);
        }

        let value = self
            .storage
            .get(&self.execution_info.contract_address)
            .and_then(|contract_storage| contract_storage.get(&address))
            .cloned()
            .unwrap_or_else(|| Felt::from(0));
        Ok(value)
    }

    #[instrument(skip(self))]
    fn storage_write(
        &mut self,
        address_domain: u32,
        address: Felt,
        value: Felt,
        remaining_gas: &mut u64,
    ) -> crate::starknet::SyscallResult<()> {
        tracing::debug!("called");
        if !address_domain.is_zero() {
            // Only address_domain 0 is currently supported.
            return Err(vec![Felt::from_bytes_be_slice(
                b"Unsupported address domain",
            )]);
        }

        self.storage
            .entry(self.execution_info.contract_address)
            .or_default()
            .insert(address, value);
        Ok(())
    }

    #[instrument(skip(self))]
    fn emit_event(
        &mut self,
        keys: &[Felt],
        data: &[Felt],
        remaining_gas: &mut u64,
    ) -> crate::starknet::SyscallResult<()> {
        tracing::debug!("called");

        let contract = self.execution_info.contract_address;
        self.logs
            .entry(contract)
            .or_default()
            .events
            .push_back((keys.to_vec(), data.to_vec()));

        Ok(())
    }

    #[instrument(skip(self))]
    fn send_message_to_l1(
        &mut self,
        to_address: Felt,
        payload: &[Felt],
        remaining_gas: &mut u64,
    ) -> crate::starknet::SyscallResult<()> {
        tracing::debug!("called");

        let contract = self.execution_info.contract_address;
        self.logs
            .entry(contract)
            .or_default()
            .l2_to_l1_messages
            .push_back((to_address, payload.to_vec()));

        Ok(())
    }

    #[instrument(skip(self))]
    fn keccak(&mut self, input: &[u64], gas: &mut u64) -> SyscallResult<U256> {
        const KECCAK_FULL_RATE_IN_WORDS: usize = 17;

        let length = input.len();
        let (_n_rounds, remainder) = num_integer::div_rem(length, KECCAK_FULL_RATE_IN_WORDS);

        if remainder != 0 {
            // In VM this error is wrapped into `SyscallExecutionError::SyscallError`
            return Err(vec![Felt::from_hex(
                "0x000000000000000000000000496e76616c696420696e707574206c656e677468",
            )
            .unwrap()]);
        }

        let mut state = [0u64; 25];
        for chunk in input.chunks(KECCAK_FULL_RATE_IN_WORDS) {
            for (i, val) in chunk.iter().enumerate() {
                state[i] ^= val;
            }
            keccak::f1600(&mut state)
        }

        Ok(U256 {
            hi: u128::from(state[2]) | (u128::from(state[3]) << 64),
            lo: u128::from(state[0]) | (u128::from(state[1]) << 64),
        })
    }

    #[instrument(skip(self))]
    fn secp256k1_new(
        &mut self,
        x: U256,
        y: U256,
        _remaining_gas: &mut u64,
    ) -> SyscallResult<Option<Secp256k1Point>> {
        tracing::debug!("called");
        Secp256Point::new(x, y).map(|op| op.map(|p| p.into()))
    }

    #[instrument(skip(self))]
    fn secp256k1_add(
        &mut self,
        p0: Secp256k1Point,
        p1: Secp256k1Point,
        _remaining_gas: &mut u64,
    ) -> SyscallResult<Secp256k1Point> {
        tracing::debug!("called");
        Ok(Secp256Point::add(p0.into(), p1.into()).into())
    }

    #[instrument(skip(self))]
    fn secp256k1_mul(
        &mut self,
        p: Secp256k1Point,
        m: U256,
        _remaining_gas: &mut u64,
    ) -> SyscallResult<Secp256k1Point> {
        tracing::debug!("called");
        Ok(Secp256Point::mul(p.into(), m).into())
    }

    #[instrument(skip(self))]
    fn secp256k1_get_point_from_x(
        &mut self,
        x: U256,
        y_parity: bool,
        _remaining_gas: &mut u64,
    ) -> SyscallResult<Option<Secp256k1Point>> {
        tracing::debug!("called");
        Secp256Point::get_point_from_x(x, y_parity).map(|op| op.map(|p| p.into()))
    }

    #[instrument(skip(self))]
    fn secp256k1_get_xy(
        &mut self,
        p: Secp256k1Point,
        _remaining_gas: &mut u64,
    ) -> SyscallResult<(U256, U256)> {
        tracing::debug!("called");
        Ok((p.x, p.y))
    }

    #[instrument(skip(self))]
    fn secp256r1_new(
        &mut self,
        x: U256,
        y: U256,
        _remaining_gas: &mut u64,
    ) -> SyscallResult<Option<Secp256r1Point>> {
        tracing::debug!("called");
        Secp256Point::new(x, y).map(|op| op.map(|p| p.into()))
    }

    #[instrument(skip(self))]
    fn secp256r1_add(
        &mut self,
        p0: Secp256r1Point,
        p1: Secp256r1Point,
        _remaining_gas: &mut u64,
    ) -> SyscallResult<Secp256r1Point> {
        tracing::debug!("called");
        Ok(Secp256Point::add(p0.into(), p1.into()).into())
    }

    #[instrument(skip(self))]
    fn secp256r1_mul(
        &mut self,
        p: Secp256r1Point,
        m: U256,
        _remaining_gas: &mut u64,
    ) -> SyscallResult<Secp256r1Point> {
        tracing::debug!("called");
        Ok(Secp256Point::mul(p.into(), m).into())
    }

    #[instrument(skip(self))]
    fn secp256r1_get_point_from_x(
        &mut self,
        x: U256,
        y_parity: bool,
        _remaining_gas: &mut u64,
    ) -> SyscallResult<Option<Secp256r1Point>> {
        tracing::debug!("called");
        Secp256Point::get_point_from_x(x, y_parity).map(|op| op.map(|p| p.into()))
    }

    #[instrument(skip(self))]
    fn secp256r1_get_xy(
        &mut self,
        p: Secp256r1Point,
        _remaining_gas: &mut u64,
    ) -> SyscallResult<(U256, U256)> {
        tracing::debug!("called");
        Ok((p.x, p.y))
    }

    #[instrument(skip(self))]
    fn meta_tx_v0(
        &mut self,
        address: Felt,
        entry_point_selector: Felt,
        calldata: &[Felt],
        signature: &[Felt],
        remaining_gas: &mut u64,
    ) -> SyscallResult<Vec<Felt>> {
        tracing::warn!("unimplemented");
        Ok(vec![])
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
            "set_block_hash" => {
                let block_number = input[0].to_biguint().try_into().unwrap();
                let block_hash = input[1];
                self.block_hash.insert(block_number, block_hash);
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
                    serialized_log.push(log.0.len().into());
                    serialized_log.append(&mut log.0);
                    serialized_log.push(log.1.len().into());
                    serialized_log.append(&mut log.1);
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
            _ => todo!("cheatcode: {}", selector),
        }
    }

    fn sha256_process_block(
        &mut self,
        state: &mut [u32; 8],
        block: &[u32; 16],
        _remaining_gas: &mut u64,
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

    fn get_class_hash_at(
        &mut self,
        contract_address: Felt,
        _remaining_gas: &mut u64,
    ) -> SyscallResult<Felt> {
        Ok(contract_address)
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
            is_infinity: false,
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
            Some(Secp256k1Point {
                x,
                y,
                is_infinity: false
            })
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
            is_infinity: false,
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
            is_infinity: false,
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
            is_infinity: false,
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
                is_infinity: false
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
                is_infinity: false
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
            Secp256r1Point {
                x,
                y,
                is_infinity: false
            }
        );
    }

    #[test]
    fn test_secp256r1_new_infinity() {
        let mut test_syscall_handler = StubSyscallHandler::default();
        let mut test_syscall_handler = &mut test_syscall_handler;

        let x = U256 { hi: 0, lo: 0 };
        let y = U256 { hi: 0, lo: 0 };

        assert!(
            test_syscall_handler
                .secp256r1_new(x, y, &mut 10)
                .unwrap()
                .unwrap()
                .is_infinity
        );
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
            is_infinity: false,
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
            is_infinity: false,
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
            is_infinity: false,
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
            Secp256r1Point {
                x,
                y,
                is_infinity: false
            }
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
            Secp256r1Point {
                x,
                y,
                is_infinity: false
            }
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
            is_infinity: false,
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
