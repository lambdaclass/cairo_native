use std::{
    io::{BufRead, BufReader},
    path::Path,
    process::{Child, Stdio},
};

use cairo_lang_sierra::program::{Program, VersionedProgram};
use ipc_channel::ipc::{IpcOneShotServer, IpcReceiver, IpcSender};
use serde::{Deserialize, Serialize};
use starknet_types_core::felt::Felt;
use uuid::Uuid;

use crate::{
    execution_result::{ContractExecutionResult, ExecutionResult},
    starknet::ExecutionInfo,
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
    GetBlockHash { block_number: u64, gas: u128 },
    GetExecutionInfo { gas: u128 },
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyscallAnswer {
    GetBlockHash {
        result: Felt,
        remaining_gas: u128,
    },
    GetExecutionInfo {
        info: ExecutionInfo,
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

pub type OnResult = Box<dyn Fn(ExecutionResult, Uuid)>;

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
        tracing::debug!("accepted receiver {receiver:?} wit msg {msg:?}");
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

        let ack = self.receiver.recv()?.to_msg()?;

        if let Message::Ack(recv_id) = ack {
            assert_eq!(recv_id, id, "id mismatch");
        } else {
            // should match
            panic!("id mismatch");
        }

        let result = self.receiver.recv()?.to_msg()?;

        if let Message::ExecutionResult {
            id: recv_id,
            result,
        } = result
        {
            assert_eq!(recv_id, id, "id mismatch");
            Ok(result)
        } else {
            panic!("wrong msg");
        }
    }
}

impl Drop for IsolatedExecutor {
    fn drop(&mut self) {
        let _ = self.sender.send(Message::Kill.wrap().unwrap());
        let _ = self.proc.kill();
    }
}
