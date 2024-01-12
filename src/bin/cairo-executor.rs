use std::io::Write;

use cairo_native::{
    context::NativeContext,
    executor::JitNativeExecutor,
    sandbox::{Message, WrappedMessage},
    utils::find_entry_point,
};
use ipc_channel::ipc::{IpcOneShotServer, IpcSender};

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args();

    if args.len() < 2 {
        panic!("missing server ipc name");
    }

    let pid = std::process::id();
    let mut log_file = std::fs::File::create(format!("cairo-executor.{pid}.log"))?;

    let server = args.nth(1).unwrap();
    let (sv, name) = IpcOneShotServer::<WrappedMessage>::new()?;
    println!("{name}"); // print to let know
    let sender = IpcSender::connect(server.clone())?;
    sender.send(Message::Ping.wrap()?)?;
    writeln!(log_file, "connected to {server:?}")?;
    let (receiver, msg) = sv.accept()?;
    writeln!(log_file, "accepted {receiver:?}")?;
    assert_eq!(msg, Message::Ping.wrap()?);

    let native_context = NativeContext::new();
    writeln!(log_file, "initialized native context")?;
    log_file.flush()?;

    loop {
        writeln!(log_file, "waiting for message")?;

        let message: Message = receiver.recv()?.to_msg()?;
        writeln!(log_file, "got message: {:?}", message)?;

        match message {
            Message::ExecuteJIT {
                id,
                program,
                inputs,
                entry_point,
            } => {
                sender.send(Message::Ack(id).wrap()?)?;
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

                sender.send(Message::ExecutionResult { id, result }.wrap()?)?;

                writeln!(log_file, "sent result msg")?;
                log_file.flush()?;
            }
            Message::ExecutionResult { .. } => {}
            Message::Ack(_) => {}
            Message::Ping => {
                sender.send(Message::Ping.wrap()?)?;
            }
            Message::Kill => {
                break;
            }
        }
    }

    Ok(())
}
