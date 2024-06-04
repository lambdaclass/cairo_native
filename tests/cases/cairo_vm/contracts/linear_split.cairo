#[contract]
mod LinearSplit {
    use integer::u16_try_from_felt252;

     #[external]
    fn cast(a: felt252) -> Option<u16> {
        u16_try_from_felt252(a)
    }   
}
