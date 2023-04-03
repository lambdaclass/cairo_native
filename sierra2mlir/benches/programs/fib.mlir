module attributes {llvm.data_layout = ""} {
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"dup<felt252>"(%arg0: i256) -> !llvm.struct<(i256, i256)> {
    %0 = llvm.mlir.undef : !llvm.struct<(i256, i256)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i256, i256)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(i256, i256)> 
    llvm.return %2 : !llvm.struct<(i256, i256)>
  }
  llvm.func internal @"store_temp<felt252>"(%arg0: i256) -> i256 {
    llvm.return %arg0 : i256
  }
  llvm.func internal @felt252_add(%arg0: i256, %arg1: i256) -> i256 {
    %0 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %1 = llvm.add %arg0, %arg1  : i256
    %2 = llvm.icmp "uge" %1, %0 : i256
    llvm.cond_br %2, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.return %1 : i256
  ^bb2:  // pred: ^bb0
    %3 = llvm.sub %1, %0  : i256
    llvm.return %3 : i256
  }
  llvm.func internal @felt252_sub(%arg0: i256, %arg1: i256) -> i256 {
    %0 = llvm.mlir.constant(0 : i256) : i256
    %1 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %2 = llvm.sub %arg0, %arg1  : i256
    %3 = llvm.icmp "slt" %2, %0 : i256
    llvm.cond_br %3, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.return %2 : i256
  ^bb2:  // pred: ^bb0
    %4 = llvm.add %2, %1  : i256
    llvm.return %4 : i256
  }
  llvm.func internal @"rename<felt252>"(%arg0: i256) -> i256 {
    llvm.return %arg0 : i256
  }
  llvm.func internal @"struct_construct<Unit>"() -> !llvm.struct<()> {
    %0 = llvm.mlir.undef : !llvm.struct<()>
    llvm.return %0 : !llvm.struct<()>
  }
  llvm.func internal @"store_temp<Unit>"(%arg0: !llvm.struct<()>) -> !llvm.struct<()> {
    llvm.return %arg0 : !llvm.struct<()>
  }
  llvm.func @"fib::fib::fib"(%arg0: i256, %arg1: i256, %arg2: i256) -> i256 attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(0 : i256) : i256
    %1 = llvm.mlir.constant(1 : i256) : i256
    %2 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %3 = llvm.icmp "eq" %arg2, %0 : i256
    llvm.cond_br %3, ^bb4(%arg0 : i256), ^bb1(%arg0, %arg1, %arg2 : i256, i256, i256)
  ^bb1(%4: i256, %5: i256, %6: i256):  // pred: ^bb0
    %7 = llvm.add %4, %5  : i256
    %8 = llvm.icmp "uge" %7, %2 : i256
    llvm.cond_br %8, ^bb2, ^bb3(%7 : i256)
  ^bb2:  // pred: ^bb1
    %9 = llvm.sub %7, %2  : i256
    llvm.br ^bb3(%9 : i256)
  ^bb3(%10: i256):  // 2 preds: ^bb1, ^bb2
    %11 = llvm.sub %6, %1  : i256
    %12 = llvm.icmp "slt" %11, %0 : i256
    %13 = llvm.select %12, %6, %11 : i1, i256
    %14 = llvm.call @"fib::fib::fib"(%5, %10, %13) : (i256, i256, i256) -> i256
    llvm.br ^bb4(%14 : i256)
  ^bb4(%15: i256):  // 2 preds: ^bb0, ^bb3
    llvm.return %15 : i256
  }
  llvm.func @"_mlir_ciface_fib::fib::fib"(%arg0: i256, %arg1: i256, %arg2: i256) -> i256 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @"fib::fib::fib"(%arg0, %arg1, %arg2) : (i256, i256, i256) -> i256
    llvm.return %0 : i256
  }
  llvm.func @"fib::fib::fib_mid"(%arg0: i256) -> !llvm.struct<()> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(496 : i256) : i256
    %1 = llvm.mlir.constant(5 : i256) : i256
    %2 = llvm.mlir.constant(false) : i1
    %3 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %4 = llvm.mlir.constant(0 : i256) : i256
    %5 = llvm.mlir.constant(1 : i256) : i256
    %6 = llvm.mlir.constant(2 : i256) : i256
    %7 = llvm.mlir.constant(3 : i256) : i256
    %8 = llvm.mlir.constant(497 : i256) : i256
    %9 = llvm.icmp "eq" %arg0, %4 : i256
    llvm.cond_br %9, ^bb7, ^bb1(%arg0 : i256)
  ^bb1(%10: i256):  // pred: ^bb0
    llvm.cond_br %2, ^bb5(%6 : i256), ^bb2(%6, %7, %8 : i256, i256, i256)
  ^bb2(%11: i256, %12: i256, %13: i256):  // pred: ^bb1
    llvm.cond_br %2, ^bb3, ^bb4(%1 : i256)
  ^bb3:  // pred: ^bb2
    %14 = llvm.mlir.constant(-3618502788666131213697322783095070105623107215331596699973092056135872020476 : i256) : i256
    llvm.br ^bb4(%14 : i256)
  ^bb4(%15: i256):  // 2 preds: ^bb2, ^bb3
    %16 = llvm.call @"fib::fib::fib"(%7, %1, %0) : (i256, i256, i256) -> i256
    llvm.br ^bb5(%16 : i256)
  ^bb5(%17: i256):  // 2 preds: ^bb1, ^bb4
    llvm.br ^bb6(%17 : i256)
  ^bb6(%18: i256):  // pred: ^bb5
    %19 = llvm.sub %10, %5  : i256
    %20 = llvm.icmp "slt" %19, %4 : i256
    %21 = llvm.select %20, %10, %19 : i1, i256
    %22 = llvm.call @"fib::fib::fib_mid"(%21) : (i256) -> !llvm.struct<()>
    llvm.br ^bb7
  ^bb7:  // 2 preds: ^bb0, ^bb6
    %23 = llvm.mlir.undef : !llvm.struct<()>
    llvm.return %23 : !llvm.struct<()>
  }
  llvm.func @"_mlir_ciface_fib::fib::fib_mid"(%arg0: !llvm.ptr<struct<()>>, %arg1: i256) attributes {llvm.emit_c_interface} {
    %0 = llvm.call @"fib::fib::fib_mid"(%arg1) : (i256) -> !llvm.struct<()>
    llvm.store %0, %arg0 : !llvm.ptr<struct<()>>
    llvm.return
  }
  llvm.func @"fib::fib::main"() -> !llvm.struct<()> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(96 : i256) : i256
    %1 = llvm.mlir.constant(496 : i256) : i256
    %2 = llvm.mlir.constant(2 : i256) : i256
    %3 = llvm.mlir.constant(3 : i256) : i256
    %4 = llvm.mlir.constant(497 : i256) : i256
    %5 = llvm.mlir.constant(492 : i256) : i256
    %6 = llvm.mlir.constant(34 : i256) : i256
    %7 = llvm.mlir.constant(false) : i1
    %8 = llvm.mlir.constant(0 : i256) : i256
    %9 = llvm.mlir.constant(1 : i256) : i256
    %10 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %11 = llvm.mlir.constant(13 : i256) : i256
    %12 = llvm.mlir.constant(21 : i256) : i256
    %13 = llvm.mlir.constant(493 : i256) : i256
    %14 = llvm.mlir.constant(494 : i256) : i256
    %15 = llvm.mlir.constant(5 : i256) : i256
    %16 = llvm.mlir.constant(495 : i256) : i256
    %17 = llvm.mlir.constant(97 : i256) : i256
    %18 = llvm.mlir.constant(8 : i256) : i256
    llvm.cond_br %7, ^bb4(%11 : i256), ^bb1(%11, %12, %13 : i256, i256, i256)
  ^bb1(%19: i256, %20: i256, %21: i256):  // pred: ^bb0
    llvm.cond_br %7, ^bb2, ^bb3(%6 : i256)
  ^bb2:  // pred: ^bb1
    %22 = llvm.mlir.constant(-3618502788666131213697322783095070105623107215331596699973092056135872020447 : i256) : i256
    llvm.br ^bb3(%22 : i256)
  ^bb3(%23: i256):  // 2 preds: ^bb1, ^bb2
    %24 = llvm.call @"fib::fib::fib"(%12, %6, %5) : (i256, i256, i256) -> i256
    llvm.br ^bb4(%24 : i256)
  ^bb4(%25: i256):  // 2 preds: ^bb0, ^bb3
    llvm.br ^bb5(%25 : i256)
  ^bb5(%26: i256):  // pred: ^bb4
    llvm.cond_br %7, ^bb9(%18 : i256), ^bb6(%18, %11, %14 : i256, i256, i256)
  ^bb6(%27: i256, %28: i256, %29: i256):  // pred: ^bb5
    llvm.cond_br %7, ^bb7, ^bb8(%12 : i256)
  ^bb7:  // pred: ^bb6
    %30 = llvm.mlir.constant(-3618502788666131213697322783095070105623107215331596699973092056135872020460 : i256) : i256
    llvm.br ^bb8(%30 : i256)
  ^bb8(%31: i256):  // 2 preds: ^bb6, ^bb7
    %32 = llvm.call @"fib::fib::fib"(%11, %12, %13) : (i256, i256, i256) -> i256
    llvm.br ^bb9(%32 : i256)
  ^bb9(%33: i256):  // 2 preds: ^bb5, ^bb8
    llvm.br ^bb10(%33 : i256)
  ^bb10(%34: i256):  // pred: ^bb9
    llvm.cond_br %7, ^bb14(%15 : i256), ^bb11(%15, %18, %16 : i256, i256, i256)
  ^bb11(%35: i256, %36: i256, %37: i256):  // pred: ^bb10
    llvm.cond_br %7, ^bb12, ^bb13(%11 : i256)
  ^bb12:  // pred: ^bb11
    %38 = llvm.mlir.constant(-3618502788666131213697322783095070105623107215331596699973092056135872020468 : i256) : i256
    llvm.br ^bb13(%38 : i256)
  ^bb13(%39: i256):  // 2 preds: ^bb11, ^bb12
    %40 = llvm.call @"fib::fib::fib"(%18, %11, %14) : (i256, i256, i256) -> i256
    llvm.br ^bb14(%40 : i256)
  ^bb14(%41: i256):  // 2 preds: ^bb10, ^bb13
    llvm.br ^bb15(%41 : i256)
  ^bb15(%42: i256):  // pred: ^bb14
    llvm.cond_br %7, ^bb27, ^bb16(%17 : i256)
  ^bb16(%43: i256):  // pred: ^bb15
    llvm.cond_br %7, ^bb25(%2 : i256), ^bb17(%2, %3, %4 : i256, i256, i256)
  ^bb17(%44: i256, %45: i256, %46: i256):  // pred: ^bb16
    llvm.cond_br %7, ^bb18, ^bb19(%15 : i256)
  ^bb18:  // pred: ^bb17
    %47 = llvm.mlir.constant(-3618502788666131213697322783095070105623107215331596699973092056135872020476 : i256) : i256
    llvm.br ^bb19(%47 : i256)
  ^bb19(%48: i256):  // 2 preds: ^bb17, ^bb18
    %49 = llvm.icmp "eq" %1, %8 : i256
    llvm.cond_br %49, ^bb23(%3 : i256), ^bb20(%3, %15, %1 : i256, i256, i256)
  ^bb20(%50: i256, %51: i256, %52: i256):  // pred: ^bb19
    llvm.cond_br %7, ^bb21, ^bb22(%18 : i256)
  ^bb21:  // pred: ^bb20
    %53 = llvm.mlir.constant(-3618502788666131213697322783095070105623107215331596699973092056135872020473 : i256) : i256
    llvm.br ^bb22(%53 : i256)
  ^bb22(%54: i256):  // 2 preds: ^bb20, ^bb21
    %55 = llvm.sub %52, %9  : i256
    %56 = llvm.icmp "slt" %55, %8 : i256
    %57 = llvm.select %56, %52, %55 : i1, i256
    %58 = llvm.call @"fib::fib::fib"(%15, %18, %57) : (i256, i256, i256) -> i256
    llvm.br ^bb23(%58 : i256)
  ^bb23(%59: i256):  // 2 preds: ^bb19, ^bb22
    llvm.br ^bb24(%59 : i256)
  ^bb24(%60: i256):  // pred: ^bb23
    llvm.br ^bb25(%60 : i256)
  ^bb25(%61: i256):  // 2 preds: ^bb16, ^bb24
    llvm.br ^bb26(%61 : i256)
  ^bb26(%62: i256):  // pred: ^bb25
    %63 = llvm.call @"fib::fib::fib_mid"(%0) : (i256) -> !llvm.struct<()>
    llvm.br ^bb27
  ^bb27:  // 2 preds: ^bb15, ^bb26
    %64 = llvm.mlir.undef : !llvm.struct<()>
    llvm.br ^bb28(%64 : !llvm.struct<()>)
  ^bb28(%65: !llvm.struct<()>):  // pred: ^bb27
    llvm.return %64 : !llvm.struct<()>
  }
  llvm.func @"_mlir_ciface_fib::fib::main"(%arg0: !llvm.ptr<struct<()>>) attributes {llvm.emit_c_interface} {
    %0 = llvm.call @"fib::fib::main"() : () -> !llvm.struct<()>
    llvm.store %0, %arg0 : !llvm.ptr<struct<()>>
    llvm.return
  }
}
