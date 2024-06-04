#[contract]
mod UintDivMod {

use integer::{u512, u512_safe_div_rem_by_u256};

    #[external]
    fn div_mod() -> () {
        let zero = u512 { limb0: 0, limb1: 0, limb2: 0, limb3: 0 };
        let one = u512 { limb0: 1, limb1: 0, limb2: 0, limb3: 0 };

        let (q, r) = u512_safe_div_rem_by_u256(zero, integer::u256_as_non_zero(1));
        assert(q == zero, '0 / 1 != 0');
        assert(r == 0, '0 % 1 != 0');

        let (q, r) = u512_safe_div_rem_by_u256(one, integer::u256_as_non_zero(1));
        assert(q == one, '1 / 1 != 1');
        assert(r == 0, '1 % 1 != 0');

        let two = u512 {limb0: 0, limb1: 0, limb2: 0, limb3: 2};
        let (q, r) = u512_safe_div_rem_by_u256(two, integer::u256_as_non_zero(1));
        assert(q == two, '2/1 != 2');
        assert(r == 0, '2/1 != 0');

        return ();
    }
}
