use std::{
    io::{Read, Write},
    path::Path,
    process::{Child, ChildStdin, ChildStdout, Stdio},
};

use cairo_lang_sierra::program::{Program, VersionedProgram};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{execution_result::ExecutionResult, values::JitValue};

#[derive(Debug, Serialize, Deserialize)]
pub enum Message {
    ExecuteJIT {
        id: Uuid,
        program: VersionedProgram,
        inputs: Vec<JitValue>,
        entry_point: String,
    },
    ExecutionResult {
        id: Uuid,
        result: ExecutionResult,
    },
    Ack(Uuid),
}

pub type OnResult = Box<dyn Fn(ExecutionResult, Uuid)>;

pub struct IsolatedExecutor {
    proc: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    read_buffer: [u8; 512],
}

impl IsolatedExecutor {
    // "target/debug/cairo-executor"
    pub fn new(executor_path: &Path) -> Result<Self, std::io::Error> {
        println!("creating executor with: {:?}", executor_path);
        let mut cmd = std::process::Command::new(executor_path);
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped());
        let mut proc = cmd.spawn()?;
        let stdin = proc.stdin.take().unwrap();
        let stdout = proc.stdout.take().unwrap();

        Ok(Self {
            proc,
            stdin,
            stdout,
            read_buffer: [0; 512]
        })
    }

    pub fn run_program(
        &mut self,
        program: Program,
        inputs: Vec<JitValue>,
        entry_point: String,
    ) -> Result<ExecutionResult, Box<dyn std::error::Error>> {
        println!("running program");
        let id = Uuid::new_v4();

        let msg = Message::ExecuteJIT {
            id,
            program: program.into_artifact(),
            inputs,
            entry_point
        };
        self.send_msg(msg)?;

        let ack = self.read_msg()?;

        if let Message::Ack(recv_id) = ack {
            assert_eq!(recv_id, id, "id mismatch");
        } else {
            // should match
            panic!("id mismatch");
        }

        let result = self.read_msg()?;

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

    fn read_msg(&mut self) -> Result<Message, Box<dyn std::error::Error>> {
        let mut msg = Vec::new();
        // check if we have a partial msg before
        for x in self.read_buffer.iter_mut() {
            if *x != b'\0' {
                msg.push(x);
            }
            *x = b'\0';
        }
        let n = self.stdout.read(&mut self.read_buffer)?;

        for x in &mut self.read_buffer[0..n] {
            if *x != b'\0' {
                msg.push(x);
            } else {

            }
        }

        let msg = serde_json::from_slice(&buf)?;
        Ok(msg)
    }

    fn send_msg(&mut self, msg: Message) -> Result<(), Box<dyn std::error::Error>> {
        let msg = serde_json::to_vec(&msg).unwrap();
        self.stdin.write_all(&msg)?;
        self.stdin.write_all(&[b'\0'])?;
        self.stdin.flush()?;
        Ok(())
    }
}

impl Drop for IsolatedExecutor {
    fn drop(&mut self) {
        let _ = self.proc.kill();
    }
}
