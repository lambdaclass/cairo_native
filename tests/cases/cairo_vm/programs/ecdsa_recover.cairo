    
fn main() -> felt252 {
    let message_hash: felt252 = 0x503f4bea29baee10b22a7f10bdc82dda071c977c1f25b8f3973d34e6b03b2c;
    let signature_r: felt252 = 0xbe96d72eb4f94078192c2e84d5230cde2a70f4b45c8797e2c907acff5060bb;
    let signature_s: felt252 = 0x677ae6bba6daf00d2631fab14c8acf24be6579f9d9e98f67aa7f2770e57a1f5;
    core::ecdsa::recover_public_key(:message_hash, :signature_r, :signature_s, y_parity: false).unwrap()      
}
