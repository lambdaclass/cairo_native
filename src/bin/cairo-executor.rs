use std::path::PathBuf;

use cairo_native::{
    context::NativeContext,
    executor::JitNativeExecutor,
    sandbox::{Message, WrappedMessage},
    utils::find_entry_point,
};
use ipc_channel::ipc::{IpcOneShotServer, IpcSender};

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args();

    let pid = std::process::id();

    let log_dir = PathBuf::from(
        std::env::var("CAIRO_EXECUTOR_LOGDIR").unwrap_or("executor_logs/".to_string()),
    );
    let file_appender =
        tracing_appender::rolling::daily(log_dir, format!("cairo-executor.{pid}.log"));

    tracing_subscriber::fmt()
        .with_writer(file_appender)
        .with_ansi(false)
        .init();

    if args.len() < 2 {
        tracing::error!("missing server ipc name");
        std::process::exit(1);
    }

    let server = args.nth(1).unwrap();
    let (sv, name) = IpcOneShotServer::<WrappedMessage>::new()?;
    println!("{name}"); // print to let know
    let sender = IpcSender::connect(server.clone())?;
    sender.send(Message::Ping.wrap()?)?;
    tracing::info!("connected to {server:?}");
    let (receiver, msg) = sv.accept()?;
    tracing::info!("accepted {receiver:?}");
    assert_eq!(msg, Message::Ping.wrap()?);

    let native_context = NativeContext::new();
    tracing::info!("initialized native context");

    loop {
        tracing::info!("waiting for message");

        let message: Message = receiver.recv()?.to_msg()?;
        tracing::info!("got message: {:?}", message);

        match message {
            Message::ExecuteJIT {
                id,
                program,
                inputs,
                entry_point,
            } => {
                sender.send(Message::Ack(id).wrap()?)?;
                tracing::info!("sent ack: {:?}", id);
                let program = program.into_v1()?.program;
                let native_program = native_context.compile(&program)?;

                let entry_point_fn = find_entry_point(&program, &entry_point).unwrap();

                let fn_id = &entry_point_fn.id;

                let native_executor =
                    JitNativeExecutor::from_native_module(native_program, Default::default());

                let result = native_executor.invoke_dynamic(fn_id, &inputs, None, None)?;

                tracing::info!("invoked with result: {:?}", result);

                sender.send(Message::ExecutionResult { id, result }.wrap()?)?;

                tracing::info!("sent result msg");
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
