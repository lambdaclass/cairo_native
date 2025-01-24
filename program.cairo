fn run_test(lhs: u256, rhs: u256) -> (u256, u256) {
         let q = lhs / rhs;
         let r = lhs % rhs;

         (q, r)
     }
