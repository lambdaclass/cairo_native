#[contract]
mod TestLessThan {
use integer::upcast;
use integer::downcast;
use option::OptionTrait;

    // tests whether the input (u128) can be downcast to an u8
    #[external]
    fn test_less_than_with_downcast(number: u128) -> bool {
        let downcast_test: Option<u8> = downcast(number);

        match downcast_test {
            Option::Some(_) => { return true; },
            Option::None(_) => { return false; }
        }
    }
}
