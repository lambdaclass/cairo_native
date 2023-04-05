module attributes {llvm.data_layout = ""} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"dup<felt252>"(%arg0: i256) -> !llvm.struct<(i256, i256)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i256, i256)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i256, i256)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(i256, i256)> 
    llvm.return %2 : !llvm.struct<(i256, i256)>
  }
  llvm.func internal @felt252_add(%arg0: i256, %arg1: i256) -> i256 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.add %arg0, %arg1  : i256
    %1 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %2 = llvm.icmp "uge" %0, %1 : i256
    llvm.cond_br %2, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.return %0 : i256
  ^bb2:  // pred: ^bb0
    %3 = llvm.sub %0, %1  : i256
    llvm.return %3 : i256
  }
  llvm.func internal @felt252_sub(%arg0: i256, %arg1: i256) -> i256 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.sub %arg0, %arg1  : i256
    %1 = llvm.mlir.constant(0 : i256) : i256
    %2 = llvm.icmp "slt" %0, %1 : i256
    llvm.cond_br %2, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.return %0 : i256
  ^bb2:  // pred: ^bb0
    %3 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %4 = llvm.add %0, %3  : i256
    llvm.return %4 : i256
  }
  llvm.func internal @"struct_construct<Tuple<felt252, felt252>>"(%arg0: i256, %arg1: i256) -> !llvm.struct<(i256, i256)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i256, i256)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i256, i256)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i256, i256)> 
    llvm.return %2 : !llvm.struct<(i256, i256)>
  }
  llvm.func internal @"store_temp<Tuple<felt252, felt252>>"(%arg0: !llvm.struct<(i256, i256)>) -> !llvm.struct<(i256, i256)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    llvm.return %arg0 : !llvm.struct<(i256, i256)>
  }
  llvm.func @"simple::simple::something"(%arg0: i256) -> !llvm.struct<(i256, i256)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb1(%arg0 : i256)
  ^bb1(%0: i256):  // pred: ^bb0
    %1 = llvm.mlir.constant(2 : i256) : i256
    %2 = llvm.call @"dup<felt252>"(%0) : (i256) -> !llvm.struct<(i256, i256)>
    %3 = llvm.extractvalue %2[0] : !llvm.struct<(i256, i256)> 
    %4 = llvm.extractvalue %2[1] : !llvm.struct<(i256, i256)> 
    %5 = llvm.call @felt252_add(%4, %1) : (i256, i256) -> i256
    %6 = llvm.mlir.constant(2 : i256) : i256
    %7 = llvm.call @felt252_sub(%3, %6) : (i256, i256) -> i256
    %8 = llvm.call @"struct_construct<Tuple<felt252, felt252>>"(%5, %7) : (i256, i256) -> !llvm.struct<(i256, i256)>
    %9 = llvm.call @"store_temp<Tuple<felt252, felt252>>"(%8) : (!llvm.struct<(i256, i256)>) -> !llvm.struct<(i256, i256)>
    llvm.return %9 : !llvm.struct<(i256, i256)>
  }
  llvm.func @"_mlir_ciface_simple::simple::something"(%arg0: !llvm.ptr<struct<(i256, i256)>>, %arg1: i256) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"simple::simple::something"(%arg1) : (i256) -> !llvm.struct<(i256, i256)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i256, i256)>>
    llvm.return
  }
}
