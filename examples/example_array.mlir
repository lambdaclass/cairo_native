module attributes {llvm.data_layout = ""} {
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"array_new<u64>"() -> !llvm.struct<(i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i32, i32, ptr)>
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(8 : i32) : i32
    %3 = llvm.insertvalue %1, %0[0] : !llvm.struct<(i32, i32, ptr)> 
    %4 = llvm.insertvalue %2, %3[1] : !llvm.struct<(i32, i32, ptr)> 
    %5 = llvm.mlir.constant(64 : i64) : i64
    %6 = llvm.mlir.null : !llvm.ptr
    %7 = llvm.call @realloc(%6, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %4[2] : !llvm.struct<(i32, i32, ptr)> 
    llvm.return %8 : !llvm.struct<(i32, i32, ptr)>
  }
  llvm.func internal @"array_append<u64>"(%arg0: !llvm.struct<(i32, i32, ptr)>, %arg1: i64) -> !llvm.struct<(i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i32, i32, ptr)> 
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(i32, i32, ptr)> 
    %2 = llvm.icmp "slt" %0, %1 : i32
    llvm.cond_br %2, ^bb2(%arg0 : !llvm.struct<(i32, i32, ptr)>), ^bb1
  ^bb1:  // pred: ^bb0
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mul %1, %3  : i32
    %5 = llvm.zext %4 : i32 to i64
    %6 = llvm.extractvalue %arg0[2] : !llvm.struct<(i32, i32, ptr)> 
    %7 = llvm.call @realloc(%6, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %arg0[2] : !llvm.struct<(i32, i32, ptr)> 
    %9 = llvm.insertvalue %4, %8[1] : !llvm.struct<(i32, i32, ptr)> 
    llvm.br ^bb2(%9 : !llvm.struct<(i32, i32, ptr)>)
  ^bb2(%10: !llvm.struct<(i32, i32, ptr)>):  // 2 preds: ^bb0, ^bb1
    %11 = llvm.extractvalue %10[2] : !llvm.struct<(i32, i32, ptr)> 
    %12 = llvm.getelementptr %11[%0] : (!llvm.ptr, i32) -> !llvm.ptr, i64
    llvm.store %arg1, %12 : i64, !llvm.ptr
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.add %0, %13  : i32
    %15 = llvm.insertvalue %14, %10[0] : !llvm.struct<(i32, i32, ptr)> 
    llvm.return %15 : !llvm.struct<(i32, i32, ptr)>
  }
  llvm.func internal @"struct_construct<Unit>"() -> !llvm.struct<()> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<()>
    llvm.return %0 : !llvm.struct<()>
  }
  llvm.func @"example_array::example_array::main"() -> !llvm.struct<(i32, i32, ptr)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.call @"array_new<u64>"() : () -> !llvm.struct<(i32, i32, ptr)>
    %1 = llvm.mlir.constant(4 : i64) : i64
    %2 = llvm.call @"array_append<u64>"(%0, %1) : (!llvm.struct<(i32, i32, ptr)>, i64) -> !llvm.struct<(i32, i32, ptr)>
    %3 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    llvm.return %2 : !llvm.struct<(i32, i32, ptr)>
  }
  llvm.func @"_mlir_ciface_example_array::example_array::main"(%arg0: !llvm.ptr<struct<(i32, i32, ptr)>>) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"example_array::example_array::main"() : () -> !llvm.struct<(i32, i32, ptr)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i32, i32, ptr)>>
    llvm.return
  }
}
