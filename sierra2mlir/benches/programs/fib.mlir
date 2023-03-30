module attributes {llvm.data_layout = ""} {
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
    llvm.cond_br %2, ^bb2, ^bb1(%1 : i256)
  ^bb1(%3: i256):  // 2 preds: ^bb0, ^bb2
    llvm.return %3 : i256
  ^bb2:  // pred: ^bb0
    %4 = llvm.sub %1, %0  : i256
    llvm.br ^bb1(%4 : i256)
  }
  llvm.func internal @felt252_sub(%arg0: i256, %arg1: i256) -> i256 {
    %0 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %1 = llvm.sub %arg0, %arg1  : i256
    %2 = llvm.icmp "ult" %arg0, %arg1 : i256
    llvm.cond_br %2, ^bb2, ^bb1(%1 : i256)
  ^bb1(%3: i256):  // 2 preds: ^bb0, ^bb2
    llvm.return %3 : i256
  ^bb2:  // pred: ^bb0
    %4 = llvm.sub %1, %0  : i256
    llvm.br ^bb1(%4 : i256)
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
  llvm.func @fib_fib_fib(%arg0: i256, %arg1: i256, %arg2: i256) -> i256 attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(0 : i256) : i256
    %1 = llvm.mlir.constant(1 : i256) : i256
    %2 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %3 = llvm.mlir.constant(2 : i256) : i256
    %4 = llvm.icmp "eq" %arg2, %0 : i256
    llvm.cond_br %4, ^bb6(%arg0 : i256), ^bb1(%arg0, %arg1, %arg2 : i256, i256, i256)
  ^bb1(%5: i256, %6: i256, %7: i256):  // pred: ^bb0
    %8 = llvm.add %5, %6  : i256
    %9 = llvm.icmp "uge" %8, %2 : i256
    llvm.cond_br %9, ^bb2, ^bb3(%8 : i256)
  ^bb2:  // pred: ^bb1
    %10 = llvm.sub %8, %2  : i256
    llvm.br ^bb3(%10 : i256)
  ^bb3(%11: i256):  // 2 preds: ^bb1, ^bb2
    %12 = llvm.sub %7, %1  : i256
    %13 = llvm.icmp "ult" %7, %1 : i256
    llvm.cond_br %13, ^bb4, ^bb5(%12 : i256)
  ^bb4:  // pred: ^bb3
    %14 = llvm.sub %7, %3  : i256
    llvm.br ^bb5(%14 : i256)
  ^bb5(%15: i256):  // 2 preds: ^bb3, ^bb4
    %16 = llvm.call @fib_fib_fib(%6, %11, %15) : (i256, i256, i256) -> i256
    llvm.br ^bb6(%16 : i256)
  ^bb6(%17: i256):  // 2 preds: ^bb0, ^bb5
    llvm.return %17 : i256
  }
  llvm.func @_mlir_ciface_fib_fib_fib(%arg0: i256, %arg1: i256, %arg2: i256) -> i256 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @fib_fib_fib(%arg0, %arg1, %arg2) : (i256, i256, i256) -> i256
    llvm.return %0 : i256
  }
  llvm.func @fib_fib_fib_mid(%arg0: i256) -> !llvm.struct<()> attributes {llvm.emit_c_interface} {
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
    llvm.cond_br %9, ^bb11, ^bb1(%arg0 : i256)
  ^bb1(%10: i256):  // pred: ^bb0
    llvm.cond_br %2, ^bb7(%6 : i256), ^bb2(%6, %7, %8 : i256, i256, i256)
  ^bb2(%11: i256, %12: i256, %13: i256):  // pred: ^bb1
    llvm.cond_br %2, ^bb3, ^bb4(%1 : i256)
  ^bb3:  // pred: ^bb2
    %14 = llvm.mlir.constant(-3618502788666131213697322783095070105623107215331596699973092056135872020476 : i256) : i256
    llvm.br ^bb4(%14 : i256)
  ^bb4(%15: i256):  // 2 preds: ^bb2, ^bb3
    llvm.cond_br %2, ^bb5, ^bb6(%0 : i256)
  ^bb5:  // pred: ^bb4
    %16 = llvm.mlir.constant(495 : i256) : i256
    llvm.br ^bb6(%16 : i256)
  ^bb6(%17: i256):  // 2 preds: ^bb4, ^bb5
    %18 = llvm.call @fib_fib_fib(%7, %1, %0) : (i256, i256, i256) -> i256
    llvm.br ^bb7(%18 : i256)
  ^bb7(%19: i256):  // 2 preds: ^bb1, ^bb6
    llvm.br ^bb8(%19 : i256)
  ^bb8(%20: i256):  // pred: ^bb7
    %21 = llvm.sub %10, %5  : i256
    %22 = llvm.icmp "ult" %10, %5 : i256
    llvm.cond_br %22, ^bb9, ^bb10(%21 : i256)
  ^bb9:  // pred: ^bb8
    %23 = llvm.sub %10, %6  : i256
    llvm.br ^bb10(%23 : i256)
  ^bb10(%24: i256):  // 2 preds: ^bb8, ^bb9
    %25 = llvm.call @fib_fib_fib_mid(%24) : (i256) -> !llvm.struct<()>
    llvm.br ^bb11
  ^bb11:  // 2 preds: ^bb0, ^bb10
    %26 = llvm.mlir.undef : !llvm.struct<()>
    llvm.return %26 : !llvm.struct<()>
  }
  llvm.func @_mlir_ciface_fib_fib_fib_mid(%arg0: !llvm.ptr<struct<()>>, %arg1: i256) attributes {llvm.emit_c_interface} {
    %0 = llvm.call @fib_fib_fib_mid(%arg1) : (i256) -> !llvm.struct<()>
    llvm.store %0, %arg0 : !llvm.ptr<struct<()>>
    llvm.return
  }
  llvm.func @fib_fib_main() -> !llvm.struct<()> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(96 : i256) : i256
    %1 = llvm.mlir.constant(496 : i256) : i256
    %2 = llvm.mlir.constant(3 : i256) : i256
    %3 = llvm.mlir.constant(497 : i256) : i256
    %4 = llvm.mlir.constant(492 : i256) : i256
    %5 = llvm.mlir.constant(34 : i256) : i256
    %6 = llvm.mlir.constant(false) : i1
    %7 = llvm.mlir.constant(0 : i256) : i256
    %8 = llvm.mlir.constant(1 : i256) : i256
    %9 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %10 = llvm.mlir.constant(2 : i256) : i256
    %11 = llvm.mlir.constant(13 : i256) : i256
    %12 = llvm.mlir.constant(21 : i256) : i256
    %13 = llvm.mlir.constant(493 : i256) : i256
    %14 = llvm.mlir.constant(494 : i256) : i256
    %15 = llvm.mlir.constant(5 : i256) : i256
    %16 = llvm.mlir.constant(97 : i256) : i256
    %17 = llvm.mlir.constant(8 : i256) : i256
    %18 = llvm.mlir.constant(495 : i256) : i256
    llvm.cond_br %6, ^bb6(%11 : i256), ^bb1(%11, %12, %13 : i256, i256, i256)
  ^bb1(%19: i256, %20: i256, %21: i256):  // pred: ^bb0
    llvm.cond_br %6, ^bb2, ^bb3(%5 : i256)
  ^bb2:  // pred: ^bb1
    %22 = llvm.mlir.constant(-3618502788666131213697322783095070105623107215331596699973092056135872020447 : i256) : i256
    llvm.br ^bb3(%22 : i256)
  ^bb3(%23: i256):  // 2 preds: ^bb1, ^bb2
    llvm.cond_br %6, ^bb4, ^bb5(%4 : i256)
  ^bb4:  // pred: ^bb3
    %24 = llvm.mlir.constant(491 : i256) : i256
    llvm.br ^bb5(%24 : i256)
  ^bb5(%25: i256):  // 2 preds: ^bb3, ^bb4
    %26 = llvm.call @fib_fib_fib(%12, %5, %4) : (i256, i256, i256) -> i256
    llvm.br ^bb6(%26 : i256)
  ^bb6(%27: i256):  // 2 preds: ^bb0, ^bb5
    llvm.br ^bb7(%27 : i256)
  ^bb7(%28: i256):  // pred: ^bb6
    llvm.cond_br %6, ^bb13(%17 : i256), ^bb8(%17, %11, %14 : i256, i256, i256)
  ^bb8(%29: i256, %30: i256, %31: i256):  // pred: ^bb7
    llvm.cond_br %6, ^bb9, ^bb10(%12 : i256)
  ^bb9:  // pred: ^bb8
    %32 = llvm.mlir.constant(-3618502788666131213697322783095070105623107215331596699973092056135872020460 : i256) : i256
    llvm.br ^bb10(%32 : i256)
  ^bb10(%33: i256):  // 2 preds: ^bb8, ^bb9
    llvm.cond_br %6, ^bb11, ^bb12(%13 : i256)
  ^bb11:  // pred: ^bb10
    %34 = llvm.mlir.constant(492 : i256) : i256
    llvm.br ^bb12(%34 : i256)
  ^bb12(%35: i256):  // 2 preds: ^bb10, ^bb11
    %36 = llvm.call @fib_fib_fib(%11, %12, %13) : (i256, i256, i256) -> i256
    llvm.br ^bb13(%36 : i256)
  ^bb13(%37: i256):  // 2 preds: ^bb7, ^bb12
    llvm.br ^bb14(%37 : i256)
  ^bb14(%38: i256):  // pred: ^bb13
    llvm.cond_br %6, ^bb20(%15 : i256), ^bb15(%15, %17, %18 : i256, i256, i256)
  ^bb15(%39: i256, %40: i256, %41: i256):  // pred: ^bb14
    llvm.cond_br %6, ^bb16, ^bb17(%11 : i256)
  ^bb16:  // pred: ^bb15
    %42 = llvm.mlir.constant(-3618502788666131213697322783095070105623107215331596699973092056135872020468 : i256) : i256
    llvm.br ^bb17(%42 : i256)
  ^bb17(%43: i256):  // 2 preds: ^bb15, ^bb16
    llvm.cond_br %6, ^bb18, ^bb19(%14 : i256)
  ^bb18:  // pred: ^bb17
    %44 = llvm.mlir.constant(493 : i256) : i256
    llvm.br ^bb19(%44 : i256)
  ^bb19(%45: i256):  // 2 preds: ^bb17, ^bb18
    %46 = llvm.call @fib_fib_fib(%17, %11, %14) : (i256, i256, i256) -> i256
    llvm.br ^bb20(%46 : i256)
  ^bb20(%47: i256):  // 2 preds: ^bb14, ^bb19
    llvm.br ^bb21(%47 : i256)
  ^bb21(%48: i256):  // pred: ^bb20
    llvm.cond_br %6, ^bb39, ^bb22(%16 : i256)
  ^bb22(%49: i256):  // pred: ^bb21
    llvm.cond_br %6, ^bb35(%10 : i256), ^bb23(%10, %2, %3 : i256, i256, i256)
  ^bb23(%50: i256, %51: i256, %52: i256):  // pred: ^bb22
    llvm.cond_br %6, ^bb24, ^bb25(%15 : i256)
  ^bb24:  // pred: ^bb23
    %53 = llvm.mlir.constant(-3618502788666131213697322783095070105623107215331596699973092056135872020476 : i256) : i256
    llvm.br ^bb25(%53 : i256)
  ^bb25(%54: i256):  // 2 preds: ^bb23, ^bb24
    llvm.cond_br %6, ^bb26, ^bb27(%1 : i256)
  ^bb26:  // pred: ^bb25
    %55 = llvm.mlir.constant(495 : i256) : i256
    llvm.br ^bb27(%55 : i256)
  ^bb27(%56: i256):  // 2 preds: ^bb25, ^bb26
    llvm.cond_br %6, ^bb33(%2 : i256), ^bb28(%2, %15, %1 : i256, i256, i256)
  ^bb28(%57: i256, %58: i256, %59: i256):  // pred: ^bb27
    llvm.cond_br %6, ^bb29, ^bb30(%17 : i256)
  ^bb29:  // pred: ^bb28
    %60 = llvm.mlir.constant(-3618502788666131213697322783095070105623107215331596699973092056135872020473 : i256) : i256
    llvm.br ^bb30(%60 : i256)
  ^bb30(%61: i256):  // 2 preds: ^bb28, ^bb29
    llvm.cond_br %6, ^bb31, ^bb32(%18 : i256)
  ^bb31:  // pred: ^bb30
    %62 = llvm.mlir.constant(494 : i256) : i256
    llvm.br ^bb32(%62 : i256)
  ^bb32(%63: i256):  // 2 preds: ^bb30, ^bb31
    %64 = llvm.call @fib_fib_fib(%15, %17, %18) : (i256, i256, i256) -> i256
    llvm.br ^bb33(%64 : i256)
  ^bb33(%65: i256):  // 2 preds: ^bb27, ^bb32
    llvm.br ^bb34(%65 : i256)
  ^bb34(%66: i256):  // pred: ^bb33
    llvm.br ^bb35(%66 : i256)
  ^bb35(%67: i256):  // 2 preds: ^bb22, ^bb34
    llvm.br ^bb36(%67 : i256)
  ^bb36(%68: i256):  // pred: ^bb35
    llvm.cond_br %6, ^bb37, ^bb38(%0 : i256)
  ^bb37:  // pred: ^bb36
    %69 = llvm.mlir.constant(95 : i256) : i256
    llvm.br ^bb38(%69 : i256)
  ^bb38(%70: i256):  // 2 preds: ^bb36, ^bb37
    %71 = llvm.call @fib_fib_fib_mid(%0) : (i256) -> !llvm.struct<()>
    llvm.br ^bb39
  ^bb39:  // 2 preds: ^bb21, ^bb38
    %72 = llvm.mlir.undef : !llvm.struct<()>
    llvm.br ^bb40(%72 : !llvm.struct<()>)
  ^bb40(%73: !llvm.struct<()>):  // pred: ^bb39
    llvm.return %72 : !llvm.struct<()>
  }
  llvm.func @_mlir_ciface_fib_fib_main(%arg0: !llvm.ptr<struct<()>>) attributes {llvm.emit_c_interface} {
    %0 = llvm.call @fib_fib_main() : () -> !llvm.struct<()>
    llvm.store %0, %arg0 : !llvm.ptr<struct<()>>
    llvm.return
  }
}
