module attributes {llvm.data_layout = ""} {
  llvm.func internal @"dup<felt252>"(%arg0: i256) -> !llvm.struct<(i256, i256)> {
    %0 = llvm.mlir.undef : !llvm.struct<(i256, i256)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i256, i256)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(i256, i256)> 
    llvm.return %2 : !llvm.struct<(i256, i256)>
  }
  llvm.func internal @felt252_add(%arg0: i256, %arg1: i256) -> i256 {
    %0 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512) : i512
    %1 = llvm.sext %arg0 : i256 to i512
    %2 = llvm.add %1, %1  : i512
    %3 = llvm.srem %2, %0  : i512
    %4 = llvm.trunc %3 : i512 to i256
    llvm.return %4 : i256
  }
  llvm.func internal @felt252_sub(%arg0: i256, %arg1: i256) -> i256 {
    %0 = llvm.mlir.constant(0 : i256) : i256
    llvm.return %0 : i256
  }
  llvm.func internal @"struct_construct<Tuple<felt252, felt252>>"(%arg0: i256, %arg1: i256) -> !llvm.struct<(i256, i256)> {
    %0 = llvm.mlir.undef : !llvm.struct<(i256, i256)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i256, i256)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i256, i256)> 
    llvm.return %2 : !llvm.struct<(i256, i256)>
  }
  llvm.func internal @"store_temp<Tuple<felt252, felt252>>"(%arg0: !llvm.struct<(i256, i256)>) -> !llvm.struct<(i256, i256)> {
    llvm.return %arg0 : !llvm.struct<(i256, i256)>
  }
  llvm.func @simple_simple_something(%arg0: i256) -> !llvm.struct<(i256, i256)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(2 : i256) : i256
    %1 = llvm.call @"dup<felt252>"(%arg0) : (i256) -> !llvm.struct<(i256, i256)>
    %2 = llvm.extractvalue %1[0] : !llvm.struct<(i256, i256)> 
    %3 = llvm.extractvalue %1[1] : !llvm.struct<(i256, i256)> 
    %4 = llvm.call @felt252_add(%3, %0) : (i256, i256) -> i256
    %5 = llvm.call @felt252_sub(%2, %0) : (i256, i256) -> i256
    %6 = llvm.call @"struct_construct<Tuple<felt252, felt252>>"(%4, %5) : (i256, i256) -> !llvm.struct<(i256, i256)>
    %7 = llvm.call @"store_temp<Tuple<felt252, felt252>>"(%6) : (!llvm.struct<(i256, i256)>) -> !llvm.struct<(i256, i256)>
    llvm.return %7 : !llvm.struct<(i256, i256)>
  }
  llvm.func @_mlir_ciface_simple_simple_something(%arg0: !llvm.ptr<struct<(i256, i256)>>, %arg1: i256) attributes {llvm.emit_c_interface} {
    %0 = llvm.call @simple_simple_something(%arg1) : (i256) -> !llvm.struct<(i256, i256)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i256, i256)>>
    llvm.return
  }
}
