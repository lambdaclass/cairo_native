module attributes {llvm.data_layout = ""} {
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @memmove(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"array_new<u32>"() -> !llvm.struct<(i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i32, i32, ptr)>
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(8 : i32) : i32
    %3 = llvm.insertvalue %1, %0[0] : !llvm.struct<(i32, i32, ptr)> 
    %4 = llvm.insertvalue %2, %3[1] : !llvm.struct<(i32, i32, ptr)> 
    %5 = llvm.mlir.constant(32 : i64) : i64
    %6 = llvm.mlir.null : !llvm.ptr
    %7 = llvm.call @realloc(%6, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %4[2] : !llvm.struct<(i32, i32, ptr)> 
    llvm.return %8 : !llvm.struct<(i32, i32, ptr)>
  }
  llvm.func internal @"array_append<u32>"(%arg0: !llvm.struct<(i32, i32, ptr)>, %arg1: i32) -> !llvm.struct<(i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i32, i32, ptr)> 
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(i32, i32, ptr)> 
    %2 = llvm.icmp "ult" %0, %1 : i32
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
    %12 = llvm.getelementptr %11[%0] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    llvm.store %arg1, %12 : i32, !llvm.ptr
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.add %0, %13  : i32
    %15 = llvm.insertvalue %14, %10[0] : !llvm.struct<(i32, i32, ptr)> 
    llvm.return %15 : !llvm.struct<(i32, i32, ptr)>
  }
  llvm.func internal @"enum_init<core::option::Option::<core::integer::u32>, 0>"(%arg0: i32) -> !llvm.struct<(i16, array<4 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<4 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<4 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<4 x i8>)>
    llvm.store %arg0, %4 : i32, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<4 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<4 x i8>)>
  }
  llvm.func internal @"struct_construct<Unit>"() -> !llvm.struct<()> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<()>
    llvm.return %0 : !llvm.struct<()>
  }
  llvm.func internal @"enum_init<core::option::Option::<core::integer::u32>, 1>"(%arg0: !llvm.struct<()>) -> !llvm.struct<(i16, array<4 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<4 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<4 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<4 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<()>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<4 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<4 x i8>)>
  }
  llvm.func internal @"array_len<u32>"(%arg0: !llvm.struct<(i32, i32, ptr)>) -> i32 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i32, i32, ptr)> 
    llvm.return %0 : i32
  }
  llvm.func internal @"struct_deconstruct<Tuple<u32>>"(%arg0: !llvm.struct<(i32)>) -> i32 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i32)> 
    llvm.return %0 : i32
  }
  llvm.func internal @"struct_construct<Tuple<u32, u32, u32, u32, u32>>"(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> !llvm.struct<(i32, i32, i32, i32, i32)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32, i32)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i32, i32, i32, i32, i32)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i32, i32, i32, i32, i32)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(i32, i32, i32, i32, i32)> 
    %4 = llvm.insertvalue %arg3, %3[3] : !llvm.struct<(i32, i32, i32, i32, i32)> 
    %5 = llvm.insertvalue %arg4, %4[4] : !llvm.struct<(i32, i32, i32, i32, i32)> 
    llvm.return %5 : !llvm.struct<(i32, i32, i32, i32, i32)>
  }
  llvm.func internal @"struct_construct<Tuple<Tuple<u32, u32, u32, u32, u32>>>"(%arg0: !llvm.struct<(i32, i32, i32, i32, i32)>) -> !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)> 
    llvm.return %1 : !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 0>"(%arg0: !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>) -> !llvm.struct<(i16, array<20 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<20 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<20 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<20 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<20 x i8>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(%arg0: !llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<20 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<20 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<20 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<20 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<(i32, i32, ptr)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<20 x i8>)>
  }
  llvm.func internal @"struct_deconstruct<Tuple<Box<u32>>>"(%arg0: !llvm.struct<(i32)>) -> i32 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i32)> 
    llvm.return %0 : i32
  }
  llvm.func internal @"struct_construct<Tuple<u32>>"(%arg0: i32) -> !llvm.struct<(i32)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i32)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i32)> 
    llvm.return %1 : !llvm.struct<(i32)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(@core::integer::u32)>, 0>"(%arg0: !llvm.struct<(i32)>) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<(i32)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(@core::integer::u32)>, 1>"(%arg0: !llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<(i32, i32, ptr)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func internal @"struct_construct<Tuple<Box<u32>>>"(%arg0: i32) -> !llvm.struct<(i32)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i32)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i32)> 
    llvm.return %1 : !llvm.struct<(i32)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 0>"(%arg0: !llvm.struct<(i32)>) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<(i32)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func internal @"array_new<felt252>"() -> !llvm.struct<(i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i32, i32, ptr)>
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(8 : i32) : i32
    %3 = llvm.insertvalue %1, %0[0] : !llvm.struct<(i32, i32, ptr)> 
    %4 = llvm.insertvalue %2, %3[1] : !llvm.struct<(i32, i32, ptr)> 
    %5 = llvm.mlir.constant(256 : i64) : i64
    %6 = llvm.mlir.null : !llvm.ptr
    %7 = llvm.call @realloc(%6, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %4[2] : !llvm.struct<(i32, i32, ptr)> 
    llvm.return %8 : !llvm.struct<(i32, i32, ptr)>
  }
  llvm.func internal @"array_append<felt252>"(%arg0: !llvm.struct<(i32, i32, ptr)>, %arg1: i256) -> !llvm.struct<(i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i32, i32, ptr)> 
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(i32, i32, ptr)> 
    %2 = llvm.icmp "ult" %0, %1 : i32
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
    %12 = llvm.getelementptr %11[%0] : (!llvm.ptr, i32) -> !llvm.ptr, i256
    llvm.store %arg1, %12 : i256, !llvm.ptr
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.add %0, %13  : i32
    %15 = llvm.insertvalue %14, %10[0] : !llvm.struct<(i32, i32, ptr)> 
    llvm.return %15 : !llvm.struct<(i32, i32, ptr)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 1>"(%arg0: !llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<(i32, i32, ptr)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func internal @print_u32(%arg0: i32) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    %0 = llvm.mlir.constant(4 : i64) : i64
    %1 = llvm.alloca %0 x i8 : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(dense<[37, 88, 10, 0]> : tensor<4xi8>) : !llvm.array<4 x i8>
    llvm.store %2, %1 : !llvm.array<4 x i8>, !llvm.ptr
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.call @dprintf(%3, %1, %arg0) : (i32, !llvm.ptr, i32) -> i32
    llvm.return
  }
  llvm.func internal @"print_Tuple<u32, u32, u32, u32, u32>"(%arg0: !llvm.struct<(i32, i32, i32, i32, i32)>) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i32, i32, i32, i32, i32)> 
    llvm.call @print_u32(%0) : (i32) -> ()
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(i32, i32, i32, i32, i32)> 
    llvm.call @print_u32(%1) : (i32) -> ()
    %2 = llvm.extractvalue %arg0[2] : !llvm.struct<(i32, i32, i32, i32, i32)> 
    llvm.call @print_u32(%2) : (i32) -> ()
    %3 = llvm.extractvalue %arg0[3] : !llvm.struct<(i32, i32, i32, i32, i32)> 
    llvm.call @print_u32(%3) : (i32) -> ()
    %4 = llvm.extractvalue %arg0[4] : !llvm.struct<(i32, i32, i32, i32, i32)> 
    llvm.call @print_u32(%4) : (i32) -> ()
    llvm.return
  }
  llvm.func internal @"print_Tuple<Tuple<u32, u32, u32, u32, u32>>"(%arg0: !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)> 
    llvm.call @"print_Tuple<u32, u32, u32, u32, u32>"(%0) : (!llvm.struct<(i32, i32, i32, i32, i32)>) -> ()
    llvm.return
  }
  llvm.func internal @print_felt252(%arg0: i256) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    %0 = llvm.mlir.constant(224 : i256) : i256
    %1 = llvm.ashr %arg0, %0  : i256
    %2 = llvm.trunc %1 : i256 to i32
    %3 = llvm.mlir.constant(5 : i64) : i64
    %4 = llvm.alloca %3 x i8 : (i64) -> !llvm.ptr
    %5 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %5, %4 : !llvm.array<5 x i8>, !llvm.ptr
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.call @dprintf(%6, %4, %2) : (i32, !llvm.ptr, i32) -> i32
    %8 = llvm.mlir.constant(192 : i256) : i256
    %9 = llvm.ashr %arg0, %8  : i256
    %10 = llvm.trunc %9 : i256 to i32
    %11 = llvm.mlir.constant(5 : i64) : i64
    %12 = llvm.alloca %11 x i8 : (i64) -> !llvm.ptr
    %13 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %13, %12 : !llvm.array<5 x i8>, !llvm.ptr
    %14 = llvm.mlir.constant(1 : i32) : i32
    %15 = llvm.call @dprintf(%14, %12, %10) : (i32, !llvm.ptr, i32) -> i32
    %16 = llvm.mlir.constant(160 : i256) : i256
    %17 = llvm.ashr %arg0, %16  : i256
    %18 = llvm.trunc %17 : i256 to i32
    %19 = llvm.mlir.constant(5 : i64) : i64
    %20 = llvm.alloca %19 x i8 : (i64) -> !llvm.ptr
    %21 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %21, %20 : !llvm.array<5 x i8>, !llvm.ptr
    %22 = llvm.mlir.constant(1 : i32) : i32
    %23 = llvm.call @dprintf(%22, %20, %18) : (i32, !llvm.ptr, i32) -> i32
    %24 = llvm.mlir.constant(128 : i256) : i256
    %25 = llvm.ashr %arg0, %24  : i256
    %26 = llvm.trunc %25 : i256 to i32
    %27 = llvm.mlir.constant(5 : i64) : i64
    %28 = llvm.alloca %27 x i8 : (i64) -> !llvm.ptr
    %29 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %29, %28 : !llvm.array<5 x i8>, !llvm.ptr
    %30 = llvm.mlir.constant(1 : i32) : i32
    %31 = llvm.call @dprintf(%30, %28, %26) : (i32, !llvm.ptr, i32) -> i32
    %32 = llvm.mlir.constant(96 : i256) : i256
    %33 = llvm.ashr %arg0, %32  : i256
    %34 = llvm.trunc %33 : i256 to i32
    %35 = llvm.mlir.constant(5 : i64) : i64
    %36 = llvm.alloca %35 x i8 : (i64) -> !llvm.ptr
    %37 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %37, %36 : !llvm.array<5 x i8>, !llvm.ptr
    %38 = llvm.mlir.constant(1 : i32) : i32
    %39 = llvm.call @dprintf(%38, %36, %34) : (i32, !llvm.ptr, i32) -> i32
    %40 = llvm.mlir.constant(64 : i256) : i256
    %41 = llvm.ashr %arg0, %40  : i256
    %42 = llvm.trunc %41 : i256 to i32
    %43 = llvm.mlir.constant(5 : i64) : i64
    %44 = llvm.alloca %43 x i8 : (i64) -> !llvm.ptr
    %45 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %45, %44 : !llvm.array<5 x i8>, !llvm.ptr
    %46 = llvm.mlir.constant(1 : i32) : i32
    %47 = llvm.call @dprintf(%46, %44, %42) : (i32, !llvm.ptr, i32) -> i32
    %48 = llvm.mlir.constant(32 : i256) : i256
    %49 = llvm.ashr %arg0, %48  : i256
    %50 = llvm.trunc %49 : i256 to i32
    %51 = llvm.mlir.constant(5 : i64) : i64
    %52 = llvm.alloca %51 x i8 : (i64) -> !llvm.ptr
    %53 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %53, %52 : !llvm.array<5 x i8>, !llvm.ptr
    %54 = llvm.mlir.constant(1 : i32) : i32
    %55 = llvm.call @dprintf(%54, %52, %50) : (i32, !llvm.ptr, i32) -> i32
    %56 = llvm.mlir.constant(0 : i256) : i256
    %57 = llvm.trunc %arg0 : i256 to i32
    %58 = llvm.mlir.constant(5 : i64) : i64
    %59 = llvm.alloca %58 x i8 : (i64) -> !llvm.ptr
    %60 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %60, %59 : !llvm.array<5 x i8>, !llvm.ptr
    %61 = llvm.mlir.constant(1 : i32) : i32
    %62 = llvm.call @dprintf(%61, %59, %57) : (i32, !llvm.ptr, i32) -> i32
    %63 = llvm.mlir.constant(2 : i64) : i64
    %64 = llvm.alloca %63 x i8 : (i64) -> !llvm.ptr
    %65 = llvm.mlir.constant(dense<[10, 0]> : tensor<2xi8>) : !llvm.array<2 x i8>
    llvm.store %65, %64 : !llvm.array<2 x i8>, !llvm.ptr
    %66 = llvm.mlir.constant(1 : i32) : i32
    %67 = llvm.call @dprintf(%66, %64) : (i32, !llvm.ptr) -> i32
    llvm.return
  }
  llvm.func internal @"print_Array<felt252>"(%arg0: !llvm.struct<(i32, i32, ptr)>) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[2] : !llvm.struct<(i32, i32, ptr)> 
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.extractvalue %arg0[0] : !llvm.struct<(i32, i32, ptr)> 
    %3 = llvm.mlir.constant(1 : i32) : i32
    llvm.br ^bb1(%1 : i32)
  ^bb1(%4: i32):  // 2 preds: ^bb0, ^bb2
    %5 = llvm.icmp "ult" %4, %2 : i32
    llvm.cond_br %5, ^bb2(%4 : i32), ^bb3
  ^bb2(%6: i32):  // pred: ^bb1
    %7 = llvm.getelementptr %0[%6] : (!llvm.ptr, i32) -> !llvm.ptr, i256
    %8 = llvm.load %7 : !llvm.ptr -> i256
    llvm.call @print_felt252(%8) : (i256) -> ()
    %9 = llvm.add %6, %3  : i32
    llvm.br ^bb1(%9 : i32)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func internal @"print_core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>"(%arg0: !llvm.struct<(i16, array<20 x i8>)>) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i16, array<20 x i8>)> 
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x !llvm.array<20 x i8> : (i64) -> !llvm.ptr
    %3 = llvm.extractvalue %arg0[1] : !llvm.struct<(i16, array<20 x i8>)> 
    llvm.store %3, %2 : !llvm.array<20 x i8>, !llvm.ptr
    llvm.switch %0 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb1:  // pred: ^bb0
    %4 = llvm.load %2 : !llvm.ptr -> !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>
    llvm.call @"print_Tuple<Tuple<u32, u32, u32, u32, u32>>"(%4) : (!llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>) -> ()
    llvm.return
  ^bb2:  // pred: ^bb0
    %5 = llvm.load %2 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.call @"print_Array<felt252>"(%5) : (!llvm.struct<(i32, i32, ptr)>) -> ()
    llvm.return
  ^bb3:  // pred: ^bb0
    llvm.return
  }
  llvm.func @main() attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"example_array::example_array::main"() : () -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.call @"print_core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>"(%0) : (!llvm.struct<(i16, array<20 x i8>)>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
  llvm.func @"example_array::example_array::main"() -> !llvm.struct<(i16, array<20 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb17
  ^bb1:  // pred: ^bb17
    %0 = llvm.extractvalue %27[2] : !llvm.struct<(i32, i32, ptr)> 
    %1 = llvm.getelementptr %0[%29] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %2 = llvm.load %1 : !llvm.ptr -> i32
    %3 = llvm.sub %30, %28  : i32
    %4 = llvm.getelementptr %0[%28] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %5 = llvm.zext %3 : i32 to i64
    %6 = llvm.call @memmove(%0, %4, %5) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %7 = llvm.insertvalue %3, %27[0] : !llvm.struct<(i32, i32, ptr)> 
    %8 = llvm.insertvalue %6, %7[2] : !llvm.struct<(i32, i32, ptr)> 
    llvm.br ^bb18(%8, %2 : !llvm.struct<(i32, i32, ptr)>, i32)
  ^bb2:  // pred: ^bb20
    %9 = llvm.load %50 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb21(%44, %9 : !llvm.struct<(i32, i32, ptr)>, !llvm.struct<(i32)>)
  ^bb3:  // pred: ^bb20
    %10 = llvm.load %50 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb30(%10 : !llvm.struct<(i32, i32, ptr)>)
  ^bb4:  // pred: ^bb20
    llvm.unreachable
  ^bb5:  // pred: ^bb21
    %11 = llvm.load %59 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb22(%53, %51, %11 : i32, !llvm.struct<(i32, i32, ptr)>, !llvm.struct<(i32)>)
  ^bb6:  // pred: ^bb21
    %12 = llvm.load %59 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb29(%12 : !llvm.struct<(i32, i32, ptr)>)
  ^bb7:  // pred: ^bb21
    llvm.unreachable
  ^bb8:  // pred: ^bb22
    %13 = llvm.load %69 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb23(%60, %63, %61, %13 : i32, i32, !llvm.struct<(i32, i32, ptr)>, !llvm.struct<(i32)>)
  ^bb9:  // pred: ^bb22
    %14 = llvm.load %69 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb28(%14 : !llvm.struct<(i32, i32, ptr)>)
  ^bb10:  // pred: ^bb22
    llvm.unreachable
  ^bb11:  // pred: ^bb23
    %15 = llvm.load %80 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb24(%70, %71, %74, %72, %15 : i32, i32, i32, !llvm.struct<(i32, i32, ptr)>, !llvm.struct<(i32)>)
  ^bb12:  // pred: ^bb23
    %16 = llvm.load %80 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb27(%16 : !llvm.struct<(i32, i32, ptr)>)
  ^bb13:  // pred: ^bb23
    llvm.unreachable
  ^bb14:  // pred: ^bb24
    %17 = llvm.load %92 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb25(%81, %82, %83, %86, %17 : i32, i32, i32, i32, !llvm.struct<(i32)>)
  ^bb15:  // pred: ^bb24
    %18 = llvm.load %92 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb26(%18 : !llvm.struct<(i32, i32, ptr)>)
  ^bb16:  // pred: ^bb24
    llvm.unreachable
  ^bb17:  // pred: ^bb0
    %19 = llvm.call @"array_new<u32>"() : () -> !llvm.struct<(i32, i32, ptr)>
    %20 = llvm.mlir.constant(1 : i32) : i32
    %21 = llvm.call @"array_append<u32>"(%19, %20) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i32, i32, ptr)>
    %22 = llvm.mlir.constant(2 : i32) : i32
    %23 = llvm.call @"array_append<u32>"(%21, %22) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i32, i32, ptr)>
    %24 = llvm.mlir.constant(3 : i32) : i32
    %25 = llvm.call @"array_append<u32>"(%23, %24) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i32, i32, ptr)>
    %26 = llvm.mlir.constant(8 : i32) : i32
    %27 = llvm.call @"array_append<u32>"(%25, %26) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i32, i32, ptr)>
    %28 = llvm.mlir.constant(1 : i32) : i32
    %29 = llvm.mlir.constant(0 : i32) : i32
    %30 = llvm.extractvalue %27[0] : !llvm.struct<(i32, i32, ptr)> 
    %31 = llvm.icmp "uge" %30, %28 : i32
    llvm.cond_br %31, ^bb1, ^bb19(%27 : !llvm.struct<(i32, i32, ptr)>)
  ^bb18(%32: !llvm.struct<(i32, i32, ptr)>, %33: i32):  // pred: ^bb1
    %34 = llvm.call @"enum_init<core::option::Option::<core::integer::u32>, 0>"(%33) : (i32) -> !llvm.struct<(i16, array<4 x i8>)>
    llvm.br ^bb20(%32 : !llvm.struct<(i32, i32, ptr)>)
  ^bb19(%35: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb17
    %36 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    %37 = llvm.call @"enum_init<core::option::Option::<core::integer::u32>, 1>"(%36) : (!llvm.struct<()>) -> !llvm.struct<(i16, array<4 x i8>)>
    llvm.br ^bb20(%35 : !llvm.struct<(i32, i32, ptr)>)
  ^bb20(%38: !llvm.struct<(i32, i32, ptr)>):  // 2 preds: ^bb18, ^bb19
    %39 = llvm.mlir.constant(4 : i32) : i32
    %40 = llvm.call @"array_append<u32>"(%38, %39) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i32, i32, ptr)>
    %41 = llvm.mlir.constant(5 : i32) : i32
    %42 = llvm.call @"array_append<u32>"(%40, %41) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i32, i32, ptr)>
    %43 = llvm.call @"array_len<u32>"(%42) : (!llvm.struct<(i32, i32, ptr)>) -> i32
    %44 = llvm.call @"array_append<u32>"(%42, %43) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i32, i32, ptr)>
    %45 = llvm.mlir.constant(0 : i32) : i32
    %46 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%44, %45) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<16 x i8>)>
    %47 = llvm.extractvalue %46[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %48 = llvm.extractvalue %46[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %49 = llvm.mlir.constant(1 : i64) : i64
    %50 = llvm.alloca %49 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %48, %50 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %47 : i16, ^bb4 [
      0: ^bb2,
      1: ^bb3
    ]
  ^bb21(%51: !llvm.struct<(i32, i32, ptr)>, %52: !llvm.struct<(i32)>):  // pred: ^bb2
    %53 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%52) : (!llvm.struct<(i32)>) -> i32
    %54 = llvm.mlir.constant(1 : i32) : i32
    %55 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%51, %54) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<16 x i8>)>
    %56 = llvm.extractvalue %55[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %57 = llvm.extractvalue %55[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %58 = llvm.mlir.constant(1 : i64) : i64
    %59 = llvm.alloca %58 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %57, %59 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %56 : i16, ^bb7 [
      0: ^bb5,
      1: ^bb6
    ]
  ^bb22(%60: i32, %61: !llvm.struct<(i32, i32, ptr)>, %62: !llvm.struct<(i32)>):  // pred: ^bb5
    %63 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%62) : (!llvm.struct<(i32)>) -> i32
    %64 = llvm.mlir.constant(2 : i32) : i32
    %65 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%61, %64) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<16 x i8>)>
    %66 = llvm.extractvalue %65[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %67 = llvm.extractvalue %65[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %68 = llvm.mlir.constant(1 : i64) : i64
    %69 = llvm.alloca %68 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %67, %69 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %66 : i16, ^bb10 [
      0: ^bb8,
      1: ^bb9
    ]
  ^bb23(%70: i32, %71: i32, %72: !llvm.struct<(i32, i32, ptr)>, %73: !llvm.struct<(i32)>):  // pred: ^bb8
    %74 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%73) : (!llvm.struct<(i32)>) -> i32
    %75 = llvm.mlir.constant(3 : i32) : i32
    %76 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%72, %75) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<16 x i8>)>
    %77 = llvm.extractvalue %76[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %78 = llvm.extractvalue %76[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %79 = llvm.mlir.constant(1 : i64) : i64
    %80 = llvm.alloca %79 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %78, %80 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %77 : i16, ^bb13 [
      0: ^bb11,
      1: ^bb12
    ]
  ^bb24(%81: i32, %82: i32, %83: i32, %84: !llvm.struct<(i32, i32, ptr)>, %85: !llvm.struct<(i32)>):  // pred: ^bb11
    %86 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%85) : (!llvm.struct<(i32)>) -> i32
    %87 = llvm.mlir.constant(4 : i32) : i32
    %88 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%84, %87) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<16 x i8>)>
    %89 = llvm.extractvalue %88[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %90 = llvm.extractvalue %88[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %91 = llvm.mlir.constant(1 : i64) : i64
    %92 = llvm.alloca %91 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %90, %92 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %89 : i16, ^bb16 [
      0: ^bb14,
      1: ^bb15
    ]
  ^bb25(%93: i32, %94: i32, %95: i32, %96: i32, %97: !llvm.struct<(i32)>):  // pred: ^bb14
    %98 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%97) : (!llvm.struct<(i32)>) -> i32
    %99 = llvm.call @"struct_construct<Tuple<u32, u32, u32, u32, u32>>"(%93, %94, %95, %96, %98) : (i32, i32, i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32, i32)>
    %100 = llvm.call @"struct_construct<Tuple<Tuple<u32, u32, u32, u32, u32>>>"(%99) : (!llvm.struct<(i32, i32, i32, i32, i32)>) -> !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>
    %101 = llvm.call @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 0>"(%100) : (!llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>) -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.return %101 : !llvm.struct<(i16, array<20 x i8>)>
  ^bb26(%102: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb15
    %103 = llvm.call @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(%102) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.return %103 : !llvm.struct<(i16, array<20 x i8>)>
  ^bb27(%104: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb12
    %105 = llvm.call @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(%104) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.return %105 : !llvm.struct<(i16, array<20 x i8>)>
  ^bb28(%106: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb9
    %107 = llvm.call @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(%106) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.return %107 : !llvm.struct<(i16, array<20 x i8>)>
  ^bb29(%108: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb6
    %109 = llvm.call @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(%108) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.return %109 : !llvm.struct<(i16, array<20 x i8>)>
  ^bb30(%110: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb3
    %111 = llvm.call @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(%110) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.return %111 : !llvm.struct<(i16, array<20 x i8>)>
  }
  llvm.func @"_mlir_ciface_example_array::example_array::main"(%arg0: !llvm.ptr<struct<(i16, array<20 x i8>)>>) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"example_array::example_array::main"() : () -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<20 x i8>)>>
    llvm.return
  }
  llvm.func @"core::array::ArrayIndex::<core::integer::u32>::index"(%arg0: !llvm.struct<(i32, i32, ptr)>, %arg1: i32) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4(%arg0, %arg1 : !llvm.struct<(i32, i32, ptr)>, i32)
  ^bb1:  // pred: ^bb4
    %0 = llvm.load %8 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb5(%0 : !llvm.struct<(i32)>)
  ^bb2:  // pred: ^bb4
    %1 = llvm.load %8 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb6(%1 : !llvm.struct<(i32, i32, ptr)>)
  ^bb3:  // pred: ^bb4
    llvm.unreachable
  ^bb4(%2: !llvm.struct<(i32, i32, ptr)>, %3: i32):  // pred: ^bb0
    %4 = llvm.call @"core::array::array_at::<core::integer::u32>"(%2, %3) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<16 x i8>)>
    %5 = llvm.extractvalue %4[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %6 = llvm.extractvalue %4[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.alloca %7 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %6, %8 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %5 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb5(%9: !llvm.struct<(i32)>):  // pred: ^bb1
    %10 = llvm.call @"struct_deconstruct<Tuple<Box<u32>>>"(%9) : (!llvm.struct<(i32)>) -> i32
    %11 = llvm.call @"struct_construct<Tuple<u32>>"(%10) : (i32) -> !llvm.struct<(i32)>
    %12 = llvm.call @"enum_init<core::PanicResult::<(@core::integer::u32)>, 0>"(%11) : (!llvm.struct<(i32)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %12 : !llvm.struct<(i16, array<16 x i8>)>
  ^bb6(%13: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb2
    %14 = llvm.call @"enum_init<core::PanicResult::<(@core::integer::u32)>, 1>"(%13) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %14 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::array::ArrayIndex::<core::integer::u32>::index"(%arg0: !llvm.ptr<struct<(i16, array<16 x i8>)>>, %arg1: !llvm.struct<(i32, i32, ptr)>, %arg2: i32) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%arg1, %arg2) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<16 x i8>)>>
    llvm.return
  }
  llvm.func @"core::array::array_at::<core::integer::u32>"(%arg0: !llvm.struct<(i32, i32, ptr)>, %arg1: i32) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb2(%arg0, %arg1 : !llvm.struct<(i32, i32, ptr)>, i32)
  ^bb1:  // pred: ^bb2
    %0 = llvm.extractvalue %3[2] : !llvm.struct<(i32, i32, ptr)> 
    %1 = llvm.getelementptr %0[%4] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %2 = llvm.load %1 : !llvm.ptr -> i32
    llvm.br ^bb3(%2 : i32)
  ^bb2(%3: !llvm.struct<(i32, i32, ptr)>, %4: i32):  // pred: ^bb0
    %5 = llvm.extractvalue %3[0] : !llvm.struct<(i32, i32, ptr)> 
    %6 = llvm.icmp "ult" %4, %5 : i32
    llvm.cond_br %6, ^bb1, ^bb4
  ^bb3(%7: i32):  // pred: ^bb1
    %8 = llvm.call @"struct_construct<Tuple<Box<u32>>>"(%7) : (i32) -> !llvm.struct<(i32)>
    %9 = llvm.call @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 0>"(%8) : (!llvm.struct<(i32)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %9 : !llvm.struct<(i16, array<16 x i8>)>
  ^bb4:  // pred: ^bb2
    %10 = llvm.call @"array_new<felt252>"() : () -> !llvm.struct<(i32, i32, ptr)>
    %11 = llvm.mlir.constant(1637570914057682275393755530660268060279989363 : i256) : i256
    %12 = llvm.call @"array_append<felt252>"(%10, %11) : (!llvm.struct<(i32, i32, ptr)>, i256) -> !llvm.struct<(i32, i32, ptr)>
    %13 = llvm.call @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 1>"(%12) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %13 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::array::array_at::<core::integer::u32>"(%arg0: !llvm.ptr<struct<(i16, array<16 x i8>)>>, %arg1: !llvm.struct<(i32, i32, ptr)>, %arg2: i32) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::array::array_at::<core::integer::u32>"(%arg1, %arg2) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<16 x i8>)>>
    llvm.return
  }
}
