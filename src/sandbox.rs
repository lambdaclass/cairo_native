use std::{
    io::{BufRead, BufReader},
    path::Path,
    process::{Child, Stdio},
};

use cairo_lang_sierra::program::{Program, VersionedProgram};
use ipc_channel::ipc::{IpcOneShotServer, IpcReceiver, IpcSender};
use lambdaworks_math::unsigned_integer::element::U256;
use serde::{Deserialize, Serialize};
use starknet_types_core::felt::Felt;
use uuid::Uuid;

use crate::{
    execution_result::ContractExecutionResult,
    starknet::{ExecutionInfo, StarkNetSyscallHandler, SyscallResult, U256},
};

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    ExecuteJIT {
        id: Uuid,
        program: VersionedProgram,
        inputs: Vec<Felt>,
        function_idx: usize,
        gas: Option<u128>,
    },
    ExecutionResult {
        id: Uuid,
        result: ContractExecutionResult,
    },
    Ack(Uuid),
    Ping,
    Kill,
    SyscallRequest(SyscallRequest),
    SyscallAnswer(SyscallAnswer),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyscallRequest {
    GetBlockHash {
        block_number: u64,
        gas: u128,
    },
    GetExecutionInfo {
        gas: u128,
    },
    Deploy {
        class_hash: Felt,
        contract_address_salt: Felt,
        calldata: Vec<Felt>,
        deploy_from_zero: bool,
        gas: u128,
    },
    ReplaceClass {
        class_hash: Felt,
        gas: u128,
    },
    LibraryCall {
        class_hash: Felt,
        function_selector: Felt,
        calldata: Vec<Felt>,
        gas: u128,
    },
    CallContract {
        address: Felt,
        entry_point_selector: Felt,
        calldata: Vec<Felt>,
        gas: u128,
    },
    EmitEvent {
        keys: Vec<Felt>,
        data: Vec<Felt>,
        gas: u128,
    },
    SendMessageToL1 {
        to_address: Felt,
        payload: Vec<Felt>,
        gas: u128,
    },
    Keccak {
        input: Vec<u64>,
        gas: u128,
    },
    StorageRead {
        address_domain: u32,
        address: Felt,
        gas: u128,
    },
    StorageWrite {
        address_domain: u32,
        address: Felt,
        value: Felt,
        gas: u128,
    },
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyscallAnswer {
    GetBlockHash {
        result: SyscallResult<Felt>,
        remaining_gas: u128,
    },
    GetExecutionInfo {
        result: SyscallResult<ExecutionInfo>,
        remaining_gas: u128,
    },
    Deploy {
        result: SyscallResult<(Felt, Vec<Felt>)>,
        remaining_gas: u128,
    },
    ReplaceClass {
        result: SyscallResult<()>,
        remaining_gas: u128,
    },
    LibraryCall {
        result: SyscallResult<Vec<Felt>>,
        remaining_gas: u128,
    },
    CallContract {
        result: SyscallResult<Vec<Felt>>,
        remaining_gas: u128,
    },
    StorageRead {
        result: SyscallResult<Felt>,
        remaining_gas: u128,
    },
    StorageWrite {
        result: SyscallResult<()>,
        remaining_gas: u128,
    },
    EmitEvent {
        result: SyscallResult<()>,
        remaining_gas: u128,
    },
    SendMessageToL1 {
        result: SyscallResult<()>,
        remaining_gas: u128,
    },
    Keccak {
        result: SyscallResult<crate::starknet::U256>,
        remaining_gas: u128,
    },
}

impl Message {
    pub fn serialize(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub fn deserialize(value: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(value)
    }

    pub fn wrap(&self) -> Result<WrappedMessage, serde_json::Error> {
        Ok(WrappedMessage::Message(self.serialize()?))
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum WrappedMessage {
    Message(String), // ipc-channel uses bincode and doesnt support serializing Vecs
}

impl WrappedMessage {
    pub fn to_msg(self) -> Result<Message, serde_json::Error> {
        match self {
            WrappedMessage::Message(msg) => Message::deserialize(&msg),
        }
    }
}

pub struct IsolatedExecutor {
    proc: Child,
    sender: IpcSender<WrappedMessage>,
    receiver: IpcReceiver<WrappedMessage>,
}

impl IsolatedExecutor {
    // "target/debug/cairo-executor"
    pub fn new(executor_path: &Path) -> Result<Self, std::io::Error> {
        let (server, server_name) = IpcOneShotServer::new().unwrap();
        tracing::debug!("creating executor with: {:?}", executor_path);

        let mut cmd = std::process::Command::new(executor_path);
        cmd.stdout(Stdio::piped());
        let mut proc = cmd.arg(server_name).spawn()?;
        let stdout = proc.stdout.take().unwrap();
        let mut stdout = BufReader::new(stdout);
        let mut client_name = String::new();
        stdout.read_line(&mut client_name)?;

        // first we accept the connection
        let (receiver, msg) = server.accept().expect("failed to accept receiver");
        tracing::debug!("accepted receiver {receiver:?} with msg {msg:?}");
        // then we connect
        tracing::debug!("connecting to {client_name}");
        let sender = IpcSender::connect(client_name.trim().to_string()).expect("failed to connect");
        sender.send(Message::Ping.wrap()?).unwrap();

        Ok(Self {
            proc,
            sender,
            receiver,
        })
    }

    pub fn run_program(
        &self,
        program: Program,
        inputs: Vec<Felt>,
        gas: Option<u128>,
        function_idx: usize,
        handler: &mut impl StarkNetSyscallHandler,
    ) -> Result<ContractExecutionResult, Box<dyn std::error::Error>> {
        tracing::debug!("running program");
        let id = Uuid::new_v4();

        let msg = Message::ExecuteJIT {
            id,
            program: program.into_artifact(),
            inputs,
            gas,
            function_idx,
        };
        self.sender.send(msg.wrap()?)?;

        loop {
            let msg = self.receiver.recv()?.to_msg()?;
            match msg {
                Message::ExecuteJIT { .. } => unreachable!(),
                Message::ExecutionResult {
                    id: recv_id,
                    result,
                } => {
                    assert_eq!(recv_id, id, "id mismatch");
                    return Ok(result);
                }
                Message::Ack(recv_id) => {
                    assert_eq!(recv_id, id, "id mismatch");
                }
                Message::Ping => unreachable!(),
                Message::Kill => todo!(),
                Message::SyscallRequest(request) => match request {
                    SyscallRequest::GetBlockHash {
                        block_number,
                        mut gas,
                    } => {
                        let result = handler.get_block_hash(block_number, &mut gas);
                        self.sender.send(
                            Message::SyscallAnswer(SyscallAnswer::GetBlockHash {
                                result,
                                remaining_gas: gas,
                            })
                            .wrap()?,
                        )?;
                    }
                    SyscallRequest::GetExecutionInfo { mut gas } => {
                        let result = handler.get_execution_info(&mut gas);
                        self.sender.send(
                            Message::SyscallAnswer(SyscallAnswer::GetExecutionInfo {
                                result,
                                remaining_gas: gas,
                            })
                            .wrap()?,
                        )?;
                    }
                    SyscallRequest::StorageRead {
                        address_domain,
                        address,
                        mut gas,
                    } => {
                        let result = handler.storage_read(address_domain, address, &mut gas);
                        self.sender.send(
                            Message::SyscallAnswer(SyscallAnswer::StorageRead {
                                result,
                                remaining_gas: gas,
                            })
                            .wrap()?,
                        )?;
                    }
                    SyscallRequest::StorageWrite {
                        address_domain,
                        address,
                        value,
                        mut gas,
                    } => {
                        let result =
                            handler.storage_write(address_domain, address, value, &mut gas);
                        self.sender.send(
                            Message::SyscallAnswer(SyscallAnswer::StorageWrite {
                                result,
                                remaining_gas: gas,
                            })
                            .wrap()?,
                        )?;
                    }
                    SyscallRequest::Deploy {
                        class_hash,
                        contract_address_salt,
                        calldata,
                        deploy_from_zero,
                        mut gas,
                    } => {
                        let result = handler.deploy(
                            class_hash,
                            contract_address_salt,
                            &calldata,
                            deploy_from_zero,
                            &mut gas,
                        );
                        self.sender.send(
                            Message::SyscallAnswer(SyscallAnswer::Deploy {
                                result,
                                remaining_gas: gas,
                            })
                            .wrap()?,
                        )?;
                    }
                    SyscallRequest::ReplaceClass { class_hash, gas } => todo!(),
                    SyscallRequest::LibraryCall {
                        class_hash,
                        function_selector,
                        calldata,
                        gas,
                    } => todo!(),
                    SyscallRequest::CallContract {
                        address,
                        entry_point_selector,
                        calldata,
                        gas,
                    } => todo!(),
                    SyscallRequest::EmitEvent { keys, data, gas } => todo!(),
                    SyscallRequest::SendMessageToL1 {
                        to_address,
                        payload,
                        gas,
                    } => todo!(),
                    SyscallRequest::Keccak { input, gas } => todo!(),
                },
                Message::SyscallAnswer(_) => unreachable!(),
            }
        }
    }
}

impl Drop for IsolatedExecutor {
    fn drop(&mut self) {
        let _ = self.sender.send(Message::Kill.wrap().unwrap());
        let _ = self.proc.kill();
    }
}
