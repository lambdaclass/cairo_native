module attributes {llvm.data_layout = ""} {
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @memmove(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"struct_construct<Unit>"() -> !llvm.struct<packed ()> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed ()>
    llvm.return %0 : !llvm.struct<packed ()>
  }
  llvm.func internal @"enum_init<core::bool, 1>"(%arg0: !llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<0 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<0 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<0 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed ()>)>
    llvm.store %arg0, %4 : !llvm.struct<packed ()>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<0 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<0 x i8>)>
  }
  llvm.func internal @"enum_init<core::bool, 0>"(%arg0: !llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<0 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<0 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<0 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed ()>)>
    llvm.store %arg0, %4 : !llvm.struct<packed ()>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<0 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<0 x i8>)>
  }
  llvm.func internal @bool_or_impl(%arg0: !llvm.struct<packed (i16, array<0 x i8>)>, %arg1: !llvm.struct<packed (i16, array<0 x i8>)>) -> !llvm.struct<packed (i16, array<0 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i16, array<0 x i8>)> 
    %1 = llvm.extractvalue %arg1[0] : !llvm.struct<packed (i16, array<0 x i8>)> 
    %2 = llvm.or %0, %1  : i16
    %3 = llvm.mlir.undef : !llvm.struct<packed (i16, array<0 x i8>)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<packed (i16, array<0 x i8>)> 
    llvm.return %4 : !llvm.struct<packed (i16, array<0 x i8>)>
  }
  llvm.func internal @bool_not_impl(%arg0: !llvm.struct<packed (i16, array<0 x i8>)>) -> !llvm.struct<packed (i16, array<0 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i16, array<0 x i8>)> 
    %1 = llvm.mlir.constant(1 : i16) : i16
    %2 = llvm.xor %0, %1  : i16
    %3 = llvm.mlir.undef : !llvm.struct<packed (i16, array<0 x i8>)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<packed (i16, array<0 x i8>)> 
    llvm.return %4 : !llvm.struct<packed (i16, array<0 x i8>)>
  }
  llvm.func internal @bool_and_impl(%arg0: !llvm.struct<packed (i16, array<0 x i8>)>, %arg1: !llvm.struct<packed (i16, array<0 x i8>)>) -> !llvm.struct<packed (i16, array<0 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i16, array<0 x i8>)> 
    %1 = llvm.extractvalue %arg1[0] : !llvm.struct<packed (i16, array<0 x i8>)> 
    %2 = llvm.and %0, %1  : i16
    %3 = llvm.mlir.undef : !llvm.struct<packed (i16, array<0 x i8>)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<packed (i16, array<0 x i8>)> 
    llvm.return %4 : !llvm.struct<packed (i16, array<0 x i8>)>
  }
  llvm.func internal @bool_xor_impl(%arg0: !llvm.struct<packed (i16, array<0 x i8>)>, %arg1: !llvm.struct<packed (i16, array<0 x i8>)>) -> !llvm.struct<packed (i16, array<0 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i16, array<0 x i8>)> 
    %1 = llvm.extractvalue %arg1[0] : !llvm.struct<packed (i16, array<0 x i8>)> 
    %2 = llvm.xor %0, %1  : i16
    %3 = llvm.mlir.undef : !llvm.struct<packed (i16, array<0 x i8>)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<packed (i16, array<0 x i8>)> 
    llvm.return %4 : !llvm.struct<packed (i16, array<0 x i8>)>
  }
  llvm.func internal @bool_to_felt252(%arg0: !llvm.struct<packed (i16, array<0 x i8>)>) -> i256 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i16, array<0 x i8>)> 
    %1 = llvm.zext %0 : i16 to i256
    llvm.return %1 : i256
  }
  llvm.func @"boolean::boolean::main"() -> i256 attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    %1 = llvm.call @"enum_init<core::bool, 1>"(%0) : (!llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<0 x i8>)>
    %2 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    %3 = llvm.call @"enum_init<core::bool, 0>"(%2) : (!llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<0 x i8>)>
    %4 = llvm.call @bool_or_impl(%1, %3) : (!llvm.struct<packed (i16, array<0 x i8>)>, !llvm.struct<packed (i16, array<0 x i8>)>) -> !llvm.struct<packed (i16, array<0 x i8>)>
    %5 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    %6 = llvm.call @"enum_init<core::bool, 0>"(%5) : (!llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<0 x i8>)>
    %7 = llvm.call @bool_not_impl(%6) : (!llvm.struct<packed (i16, array<0 x i8>)>) -> !llvm.struct<packed (i16, array<0 x i8>)>
    %8 = llvm.call @bool_not_impl(%7) : (!llvm.struct<packed (i16, array<0 x i8>)>) -> !llvm.struct<packed (i16, array<0 x i8>)>
    %9 = llvm.call @bool_and_impl(%4, %8) : (!llvm.struct<packed (i16, array<0 x i8>)>, !llvm.struct<packed (i16, array<0 x i8>)>) -> !llvm.struct<packed (i16, array<0 x i8>)>
    %10 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    %11 = llvm.call @"enum_init<core::bool, 0>"(%10) : (!llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<0 x i8>)>
    %12 = llvm.call @bool_xor_impl(%9, %11) : (!llvm.struct<packed (i16, array<0 x i8>)>, !llvm.struct<packed (i16, array<0 x i8>)>) -> !llvm.struct<packed (i16, array<0 x i8>)>
    %13 = llvm.call @bool_to_felt252(%12) : (!llvm.struct<packed (i16, array<0 x i8>)>) -> i256
    llvm.return %13 : i256
  }
  llvm.func @"_mlir_ciface_boolean::boolean::main"() -> i256 attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"boolean::boolean::main"() : () -> i256
    llvm.return %0 : i256
  }
}
