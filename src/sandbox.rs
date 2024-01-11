use std::os::unix::process::CommandExt;

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

pub struct IsolatedExecutor {}

impl IsolatedExecutor {
    pub fn start(&self) -> Result<(), std::io::Error> {
        let path = "target/debug/cairo-executor";
        let mut cmd = std::process::Command::new(path);
        let proc = cmd.spawn()?;

        Ok(())
    }
}
