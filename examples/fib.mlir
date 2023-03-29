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
    %0 = llvm.add %arg0, %arg1  : i256
    %1 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %2 = llvm.icmp "uge" %0, %1 : i256
    llvm.cond_br %2, ^bb2, ^bb1(%0 : i256)
  ^bb1(%3: i256):  // 2 preds: ^bb0, ^bb2
    llvm.return %3 : i256
  ^bb2:  // pred: ^bb0
    %4 = llvm.sub %0, %1  : i256
    llvm.br ^bb1(%4 : i256)
  }
  llvm.func internal @felt252_sub(%arg0: i256, %arg1: i256) -> i256 {
    %0 = llvm.sub %arg0, %arg1  : i256
    %1 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %2 = llvm.icmp "ult" %arg0, %arg1 : i256
    llvm.cond_br %2, ^bb2, ^bb1(%0 : i256)
  ^bb1(%3: i256):  // 2 preds: ^bb0, ^bb2
    llvm.return %3 : i256
  ^bb2:  // pred: ^bb0
    %4 = llvm.sub %0, %1  : i256
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
  llvm.func internal @print_Unit(%arg0: !llvm.struct<()>) {
    llvm.return
  }
  llvm.func @main(%arg0: i256) attributes {llvm.emit_c_interface} {
    %0 = llvm.call @fib_fib_main(%arg0) : (i256) -> !llvm.struct<()>
    llvm.call @print_Unit(%0) : (!llvm.struct<()>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main(%arg0: i256) attributes {llvm.emit_c_interface} {
    llvm.call @main(%arg0) : (i256) -> ()
    llvm.return
  }
  llvm.func internal @fib_fib_fib(%arg0: i256, %arg1: i256, %arg2: i256) -> i256 {
    llvm.br ^bb1(%arg0, %arg1, %arg2 : i256, i256, i256)
  ^bb1(%0: i256, %1: i256, %2: i256):  // pred: ^bb0
    %3 = llvm.call @"dup<felt252>"(%2) : (i256) -> !llvm.struct<(i256, i256)>
    %4 = llvm.extractvalue %3[0] : !llvm.struct<(i256, i256)> 
    %5 = llvm.extractvalue %3[1] : !llvm.struct<(i256, i256)> 
    %6 = llvm.mlir.constant(0 : i256) : i256
    %7 = llvm.icmp "eq" %5, %6 : i256
    llvm.cond_br %7, ^bb2(%0 : i256), ^bb3(%0, %1, %4 : i256, i256, i256)
  ^bb2(%8: i256):  // pred: ^bb1
    %9 = llvm.call @"store_temp<felt252>"(%8) : (i256) -> i256
    llvm.br ^bb4(%9 : i256)
  ^bb3(%10: i256, %11: i256, %12: i256):  // pred: ^bb1
    %13 = llvm.call @"dup<felt252>"(%11) : (i256) -> !llvm.struct<(i256, i256)>
    %14 = llvm.extractvalue %13[0] : !llvm.struct<(i256, i256)> 
    %15 = llvm.extractvalue %13[1] : !llvm.struct<(i256, i256)> 
    %16 = llvm.call @felt252_add(%10, %15) : (i256, i256) -> i256
    %17 = llvm.mlir.constant(1 : i256) : i256
    %18 = llvm.call @felt252_sub(%12, %17) : (i256, i256) -> i256
    %19 = llvm.call @"store_temp<felt252>"(%14) : (i256) -> i256
    %20 = llvm.call @"store_temp<felt252>"(%16) : (i256) -> i256
    %21 = llvm.call @"store_temp<felt252>"(%18) : (i256) -> i256
    %22 = llvm.call @fib_fib_fib(%19, %20, %21) : (i256, i256, i256) -> i256
    %23 = llvm.call @"rename<felt252>"(%22) : (i256) -> i256
    llvm.br ^bb4(%23 : i256)
  ^bb4(%24: i256):  // 2 preds: ^bb2, ^bb3
    %25 = llvm.call @"rename<felt252>"(%24) : (i256) -> i256
    llvm.return %25 : i256
  }
  llvm.func internal @fib_fib_fib_mid(%arg0: i256) -> !llvm.struct<()> {
    llvm.br ^bb1(%arg0 : i256)
  ^bb1(%0: i256):  // pred: ^bb0
    %1 = llvm.call @"dup<felt252>"(%0) : (i256) -> !llvm.struct<(i256, i256)>
    %2 = llvm.extractvalue %1[0] : !llvm.struct<(i256, i256)> 
    %3 = llvm.extractvalue %1[1] : !llvm.struct<(i256, i256)> 
    %4 = llvm.mlir.constant(0 : i256) : i256
    %5 = llvm.icmp "eq" %3, %4 : i256
    llvm.cond_br %5, ^bb2, ^bb3(%2 : i256)
  ^bb2:  // pred: ^bb1
    llvm.br ^bb4
  ^bb3(%6: i256):  // pred: ^bb1
    %7 = llvm.mlir.constant(0 : i256) : i256
    %8 = llvm.mlir.constant(1 : i256) : i256
    %9 = llvm.mlir.constant(500 : i256) : i256
    %10 = llvm.call @"store_temp<felt252>"(%7) : (i256) -> i256
    %11 = llvm.call @"store_temp<felt252>"(%8) : (i256) -> i256
    %12 = llvm.call @"store_temp<felt252>"(%9) : (i256) -> i256
    %13 = llvm.call @fib_fib_fib(%10, %11, %12) : (i256, i256, i256) -> i256
    %14 = llvm.mlir.constant(1 : i256) : i256
    %15 = llvm.call @felt252_sub(%6, %14) : (i256, i256) -> i256
    %16 = llvm.call @"store_temp<felt252>"(%15) : (i256) -> i256
    %17 = llvm.call @fib_fib_fib_mid(%16) : (i256) -> !llvm.struct<()>
    llvm.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %18 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    %19 = llvm.call @"store_temp<Unit>"(%18) : (!llvm.struct<()>) -> !llvm.struct<()>
    llvm.return %19 : !llvm.struct<()>
  }
  llvm.func internal @fib_fib_main(%arg0: i256) -> !llvm.struct<()> {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.mlir.constant(100 : i256) : i256
    %1 = llvm.call @"store_temp<felt252>"(%0) : (i256) -> i256
    %2 = llvm.call @fib_fib_fib_mid(%1) : (i256) -> !llvm.struct<()>
    %3 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    %4 = llvm.call @"store_temp<Unit>"(%3) : (!llvm.struct<()>) -> !llvm.struct<()>
    llvm.return %4 : !llvm.struct<()>
  }
}
