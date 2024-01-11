use std::io::Read;

pub fn main() -> Result<(), std::io::Error> {
    let mut stdin = std::io::stdin().lock();
    let mut stdout = std::io::stdout().lock();

    let mut buffer = Vec::new();
    loop {
        stdin.read_to_end(&mut buffer)?;

        // handle message

        // send result
    }
}
