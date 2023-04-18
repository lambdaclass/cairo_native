module attributes {llvm.data_layout = ""} {
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @memmove(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @bitwise(%arg0: i128, %arg1: i128) -> !llvm.struct<(i128, i128, i128)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.and %arg0, %arg1  : i128
    %1 = llvm.xor %arg0, %arg1  : i128
    %2 = llvm.or %arg0, %arg1  : i128
    %3 = llvm.mlir.undef : !llvm.struct<(i128, i128, i128)>
    %4 = llvm.insertvalue %0, %3[0] : !llvm.struct<(i128, i128, i128)> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<(i128, i128, i128)> 
    %6 = llvm.insertvalue %2, %5[2] : !llvm.struct<(i128, i128, i128)> 
    llvm.return %6 : !llvm.struct<(i128, i128, i128)>
  }
  llvm.func internal @"struct_construct<Unit>"() -> !llvm.struct<()> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<()>
    llvm.return %0 : !llvm.struct<()>
  }
  llvm.func @"bitwise::bitwise::main"() -> !llvm.struct<()> attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(5678 : i128) : i128
    %1 = llvm.mlir.constant(1234 : i128) : i128
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %2 = llvm.call @bitwise(%1, %0) : (i128, i128) -> !llvm.struct<(i128, i128, i128)>
    %3 = llvm.call @bitwise(%1, %0) : (i128, i128) -> !llvm.struct<(i128, i128, i128)>
    %4 = llvm.call @bitwise(%1, %0) : (i128, i128) -> !llvm.struct<(i128, i128, i128)>
    %5 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    llvm.return %5 : !llvm.struct<()>
  }
  llvm.func @"_mlir_ciface_bitwise::bitwise::main"(%arg0: !llvm.ptr<struct<()>>) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(5678 : i128) : i128
    %1 = llvm.mlir.constant(1234 : i128) : i128
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %2 = llvm.call @bitwise(%1, %0) : (i128, i128) -> !llvm.struct<(i128, i128, i128)>
    %3 = llvm.call @bitwise(%1, %0) : (i128, i128) -> !llvm.struct<(i128, i128, i128)>
    %4 = llvm.call @bitwise(%1, %0) : (i128, i128) -> !llvm.struct<(i128, i128, i128)>
    %5 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    llvm.br ^bb2(%5 : !llvm.struct<()>)
  ^bb2(%6: !llvm.struct<()>):  // pred: ^bb1
    llvm.store %6, %arg0 : !llvm.ptr<struct<()>>
    llvm.return
  }
}
