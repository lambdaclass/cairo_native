// primitive_types2.cairo
// Fill in the rest of the line that has code missing!
// No hints, there's no tricks, just get used to typing these :)


use debug::PrintTrait;

fn main() {
    // A short string is a string whose length is at most 31 characters, and therefore can fit into a single field element.
    // Short strings are actually felts, they are not a real string.
    // Note the _single_ quotes that are used with short strings.

    let mut my_first_initial = 'C';
    if is_alphabetic(
        ref my_first_initial
    ) {
        ('Alphabetical!').print();
    } else if is_numeric(
        ref my_first_initial
    ) {
        ('Numerical!').print();
    } else {
        ('Neither alphabetic nor numeric!').print();
    }

    let mut my_fav_string = '$';
    if is_alphabetic(
        ref my_fav_string
    ) {
        ('Alphabetical!').print();
    } else if is_numeric(
        ref my_fav_string
    ) {
        ('Numerical!').print();
    } else {
        ('Neither alphabetic nor numeric!').print();
    }
}

fn is_alphabetic(ref char: felt252) -> bool {
    if char >= 'a' {
        if char <= 'z' {
            return true;
        }
    }
    if char >= 'A' {
        if char <= 'Z' {
            return true;
        }
    }
    false
}

fn is_numeric(ref char: felt252) -> bool {
    if char >= '0' {
        if char <= '9' {
            return true;
        }
    }
    false
}

// Note: the following code is not part of the challenge, it's just here to make the code above work.
// Direct felt252 comparisons have been removed from the core library, so we need to implement them ourselves.
// There will probably be a string / short string type in the future
impl PartialOrdFelt of PartialOrd<felt252> {
    #[inline(always)]
    fn le(lhs: felt252, rhs: felt252) -> bool {
        !(rhs < lhs)
    }
    #[inline(always)]
    fn ge(lhs: felt252, rhs: felt252) -> bool {
        !(lhs < rhs)
    }
    #[inline(always)]
    fn lt(lhs: felt252, rhs: felt252) -> bool {
        integer::u256_from_felt252(lhs) < integer::u256_from_felt252(rhs)
    }
    #[inline(always)]
    fn gt(lhs: felt252, rhs: felt252) -> bool {
        rhs < lhs
    }
}
