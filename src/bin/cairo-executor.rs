use std::io::{Read, Write};

use cairo_native::{
    context::NativeContext, executor::JitNativeExecutor, sandbox::Message, utils::find_entry_point,
};

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pid = std::process::id();

    let mut log_file = std::fs::File::create(format!("cairo-executor.{pid}.log"))?;

    let mut stdin = std::io::stdin().lock();
    let mut stdout = std::io::stdout().lock();

    let mut send_msg = |msg: Message| {
        let bytes = serde_json::to_vec(&msg).unwrap();
        stdout.write_all(&bytes).unwrap();
    };

    let native_context = NativeContext::new();
    writeln!(log_file, "initialized native context")?;
    log_file.flush()?;

    let mut buffer = Vec::new();
    loop {
        writeln!(log_file, "waiting fo message")?;
        log_file.flush()?;
        stdin.read_to_end(&mut buffer)?;

        let message: Message = serde_json::from_slice(&buffer)?;
        writeln!(log_file, "got message: {:?}", message)?;
        log_file.flush()?;

        match message {
            Message::ExecuteJIT {
                id,
                program,
                inputs,
                entry_point,
            } => {
                send_msg(Message::Ack(id));
                writeln!(log_file, "sent ack: {:?}", id)?;
                log_file.flush()?;
                let program = program.into_v1()?.program;
                let native_program = native_context.compile(&program)?;

                // Call the echo function from the contract using the generated wrapper.

                let entry_point_fn = find_entry_point(&program, &entry_point).unwrap();

                let fn_id = &entry_point_fn.id;

                let native_executor =
                    JitNativeExecutor::from_native_module(native_program, Default::default());

                let result = native_executor.invoke_dynamic(fn_id, &inputs, None, None)?;

                writeln!(log_file, "invoked with result: {:?}", result)?;
                log_file.flush()?;

                send_msg(Message::ExecutionResult { id, result });

                writeln!(log_file, "sent result msg")?;
                log_file.flush()?;
            }
            Message::ExecutionResult { .. } => {}
            Message::Ack(_) => {}
        }
        buffer.clear();
    }
}
