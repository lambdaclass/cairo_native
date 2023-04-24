func.func @helper_felt_pow(%0 : i256, %1 : i256) -> i256 {
    %2 = arith.constant 1 : i256
    %27, %28, %29 = scf.while (%3 = %0, %4 = %1, %5 = %2) : (i256, i256, i256) -> (i256, i256, i256) {
        %6 = arith.constant 0 : i256
        %7 = arith.cmpi eq, %4, %6 : i256
        scf.condition(%7) %3, %4, %5 : i256, i256, i256
    } do {
      ^0(%8 : i256, %9 : i256, %10 : i256):

        %11 = arith.trunci %9 : i256 to i1
        %12 = scf.if %11 -> i256 {
            %13 = arith.extui %10 : i256 to i512
            %14 = arith.extui %8 : i256 to i512
            %15 = arith.muli %13, %14 : i512
            %16 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512
            %17 = arith.remui %15, %16 : i512
            %18 = arith.trunci %17 : i512 to i256
            scf.yield %18 : i256
        } else {
            scf.yield %10 : i256
        }

        %19 = arith.constant 1 : i256
        %20 = arith.shrui %9, %19 : i256

        %21 = arith.extui %8 : i256 to i512
        %22 = arith.muli %21, %21 : i512
        %24 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512
        %25 = arith.remui %22, %24 : i512
        %26 = arith.trunci %25 : i512 to i256

        scf.yield %26, %20, %12 : i256, i256, i256
    }

    func.return %29 : i256
}
