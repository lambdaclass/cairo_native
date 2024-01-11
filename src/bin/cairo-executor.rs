use std::io::{Read, Write};

use cairo_native::sandbox::Message;

pub fn main() -> Result<(), std::io::Error> {
    let mut stdin = std::io::stdin().lock();
    let mut stdout = std::io::stdout().lock();

    let mut send_msg = |msg: Message| {
        let bytes = bincode::serialize(&msg).expect("failed to serialize");
        stdout.write_all(&bytes).unwrap();
    };

    let mut buffer = Vec::new();
    loop {
        stdin.read_to_end(&mut buffer)?;

        let message: Message = bincode::deserialize(&buffer).expect("failed");

        match message {
            Message::ExecuteJIT { program, inputs } => {
                send_msg(Message::Ack);
            }
            Message::ExecutionResult(_) => todo!(),
            Message::Ping => send_msg(Message::Ack),
            Message::Ack => {}
        }
        buffer.clear();

        // handle message

        // send result
    }
}
