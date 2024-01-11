use std::{
    io::Stdout,
    os::unix::process::CommandExt,
    path::Path,
    process::{Child, ChildStdin, ChildStdout, Stdio},
};

use cairo_lang_sierra::program::Program;
use serde::{Deserialize, Serialize};

use crate::{execution_result::ExecutionResult, values::JitValue};

#[derive(Debug, Serialize, Deserialize)]
pub enum Message {
    ExecuteJIT {
        program: Program,
        inputs: Vec<JitValue>,
    },
    ExecutionResult(ExecutionResult),
}

pub struct IsolatedExecutor {
    proc: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
}

impl IsolatedExecutor {
    // "target/debug/cairo-executor"
    pub fn new(executor_path: &Path) -> Result<Self, std::io::Error> {
        let mut cmd = std::process::Command::new(executor_path);
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped());
        let mut proc = cmd.spawn()?;
        let stdin = proc.stdin.take().unwrap();
        let stdout = proc.stdout.take().unwrap();

        Ok(Self {
            proc,
            stdin,
            stdout,
        })
    }
}
