module attributes {llvm.data_layout = ""} {
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @memmove(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"array_new<u32>"() -> !llvm.struct<packed (i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (i32, i32, ptr)>
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(8 : i32) : i32
    %3 = llvm.insertvalue %1, %0[0] : !llvm.struct<packed (i32, i32, ptr)> 
    %4 = llvm.insertvalue %2, %3[1] : !llvm.struct<packed (i32, i32, ptr)> 
    %5 = llvm.mlir.constant(32 : i64) : i64
    %6 = llvm.mlir.null : !llvm.ptr
    %7 = llvm.call @realloc(%6, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %4[2] : !llvm.struct<packed (i32, i32, ptr)> 
    llvm.return %8 : !llvm.struct<packed (i32, i32, ptr)>
  }
  llvm.func internal @"array_append<u32>"(%arg0: !llvm.struct<packed (i32, i32, ptr)>, %arg1: i32) -> !llvm.struct<packed (i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i32, i32, ptr)> 
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<packed (i32, i32, ptr)> 
    %2 = llvm.icmp "ult" %0, %1 : i32
    llvm.cond_br %2, ^bb2(%arg0 : !llvm.struct<packed (i32, i32, ptr)>), ^bb1
  ^bb1:  // pred: ^bb0
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mul %1, %3  : i32
    %5 = llvm.zext %4 : i32 to i64
    %6 = llvm.extractvalue %arg0[2] : !llvm.struct<packed (i32, i32, ptr)> 
    %7 = llvm.call @realloc(%6, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %arg0[2] : !llvm.struct<packed (i32, i32, ptr)> 
    %9 = llvm.insertvalue %4, %8[1] : !llvm.struct<packed (i32, i32, ptr)> 
    llvm.br ^bb2(%9 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb2(%10: !llvm.struct<packed (i32, i32, ptr)>):  // 2 preds: ^bb0, ^bb1
    %11 = llvm.extractvalue %10[2] : !llvm.struct<packed (i32, i32, ptr)> 
    %12 = llvm.getelementptr %11[%0] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    llvm.store %arg1, %12 : i32, !llvm.ptr
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.add %0, %13  : i32
    %15 = llvm.insertvalue %14, %10[0] : !llvm.struct<packed (i32, i32, ptr)> 
    llvm.return %15 : !llvm.struct<packed (i32, i32, ptr)>
  }
  llvm.func internal @"enum_init<core::option::Option::<core::integer::u32>, 0>"(%arg0: i32) -> !llvm.struct<packed (i16, array<4 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<4 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<4 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i32)>
    llvm.store %arg0, %4 : i32, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<4 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<4 x i8>)>
  }
  llvm.func internal @"struct_construct<Unit>"() -> !llvm.struct<packed ()> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed ()>
    llvm.return %0 : !llvm.struct<packed ()>
  }
  llvm.func internal @"enum_init<core::option::Option::<core::integer::u32>, 1>"(%arg0: !llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<4 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<4 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<4 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed ()>)>
    llvm.store %arg0, %4 : !llvm.struct<packed ()>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<4 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<4 x i8>)>
  }
  llvm.func internal @"array_len<u32>"(%arg0: !llvm.struct<packed (i32, i32, ptr)>) -> i32 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i32, i32, ptr)> 
    llvm.return %0 : i32
  }
  llvm.func internal @"struct_deconstruct<Tuple<u32>>"(%arg0: !llvm.struct<packed (i32)>) -> i32 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i32)> 
    llvm.return %0 : i32
  }
  llvm.func internal @"struct_construct<Tuple<u32, u32, u32, u32, u32>>"(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> !llvm.struct<packed (i32, i32, i32, i32, i32)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (i32, i32, i32, i32, i32)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<packed (i32, i32, i32, i32, i32)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<packed (i32, i32, i32, i32, i32)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<packed (i32, i32, i32, i32, i32)> 
    %4 = llvm.insertvalue %arg3, %3[3] : !llvm.struct<packed (i32, i32, i32, i32, i32)> 
    %5 = llvm.insertvalue %arg4, %4[4] : !llvm.struct<packed (i32, i32, i32, i32, i32)> 
    llvm.return %5 : !llvm.struct<packed (i32, i32, i32, i32, i32)>
  }
  llvm.func internal @"struct_construct<Tuple<Tuple<u32, u32, u32, u32, u32>>>"(%arg0: !llvm.struct<packed (i32, i32, i32, i32, i32)>) -> !llvm.struct<packed (struct<packed (i32, i32, i32, i32, i32)>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (struct<packed (i32, i32, i32, i32, i32)>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<packed (struct<packed (i32, i32, i32, i32, i32)>)> 
    llvm.return %1 : !llvm.struct<packed (struct<packed (i32, i32, i32, i32, i32)>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 0>"(%arg0: !llvm.struct<packed (struct<packed (i32, i32, i32, i32, i32)>)>) -> !llvm.struct<packed (i16, array<20 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<20 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<20 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (struct<packed (i32, i32, i32, i32, i32)>)>)>
    llvm.store %arg0, %4 : !llvm.struct<packed (struct<packed (i32, i32, i32, i32, i32)>)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<20 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<20 x i8>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(%arg0: !llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<20 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    llvm.call_intrinsic "llvm.trap"() : () -> ()
    llvm.unreachable
  }
  llvm.func internal @"struct_deconstruct<Tuple<Box<u32>>>"(%arg0: !llvm.struct<packed (i32)>) -> i32 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i32)> 
    llvm.return %0 : i32
  }
  llvm.func internal @"struct_construct<Tuple<u32>>"(%arg0: i32) -> !llvm.struct<packed (i32)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (i32)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<packed (i32)> 
    llvm.return %1 : !llvm.struct<packed (i32)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(@core::integer::u32)>, 0>"(%arg0: !llvm.struct<packed (i32)>) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32)>)>
    llvm.store %arg0, %4 : !llvm.struct<packed (i32)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(@core::integer::u32)>, 1>"(%arg0: !llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    llvm.call_intrinsic "llvm.trap"() : () -> ()
    llvm.unreachable
  }
  llvm.func internal @"struct_construct<Tuple<Box<u32>>>"(%arg0: i32) -> !llvm.struct<packed (i32)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (i32)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<packed (i32)> 
    llvm.return %1 : !llvm.struct<packed (i32)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 0>"(%arg0: !llvm.struct<packed (i32)>) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32)>)>
    llvm.store %arg0, %4 : !llvm.struct<packed (i32)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func internal @"array_new<felt252>"() -> !llvm.struct<packed (i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (i32, i32, ptr)>
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(8 : i32) : i32
    %3 = llvm.insertvalue %1, %0[0] : !llvm.struct<packed (i32, i32, ptr)> 
    %4 = llvm.insertvalue %2, %3[1] : !llvm.struct<packed (i32, i32, ptr)> 
    %5 = llvm.mlir.constant(256 : i64) : i64
    %6 = llvm.mlir.null : !llvm.ptr
    %7 = llvm.call @realloc(%6, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %4[2] : !llvm.struct<packed (i32, i32, ptr)> 
    llvm.return %8 : !llvm.struct<packed (i32, i32, ptr)>
  }
  llvm.func internal @"array_append<felt252>"(%arg0: !llvm.struct<packed (i32, i32, ptr)>, %arg1: i256) -> !llvm.struct<packed (i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i32, i32, ptr)> 
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<packed (i32, i32, ptr)> 
    %2 = llvm.icmp "ult" %0, %1 : i32
    llvm.cond_br %2, ^bb2(%arg0 : !llvm.struct<packed (i32, i32, ptr)>), ^bb1
  ^bb1:  // pred: ^bb0
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mul %1, %3  : i32
    %5 = llvm.zext %4 : i32 to i64
    %6 = llvm.extractvalue %arg0[2] : !llvm.struct<packed (i32, i32, ptr)> 
    %7 = llvm.call @realloc(%6, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %arg0[2] : !llvm.struct<packed (i32, i32, ptr)> 
    %9 = llvm.insertvalue %4, %8[1] : !llvm.struct<packed (i32, i32, ptr)> 
    llvm.br ^bb2(%9 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb2(%10: !llvm.struct<packed (i32, i32, ptr)>):  // 2 preds: ^bb0, ^bb1
    %11 = llvm.extractvalue %10[2] : !llvm.struct<packed (i32, i32, ptr)> 
    %12 = llvm.getelementptr %11[%0] : (!llvm.ptr, i32) -> !llvm.ptr, i256
    llvm.store %arg1, %12 : i256, !llvm.ptr
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.add %0, %13  : i32
    %15 = llvm.insertvalue %14, %10[0] : !llvm.struct<packed (i32, i32, ptr)> 
    llvm.return %15 : !llvm.struct<packed (i32, i32, ptr)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 1>"(%arg0: !llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    llvm.call_intrinsic "llvm.trap"() : () -> ()
    llvm.unreachable
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
  llvm.func internal @"print_Tuple<u32, u32, u32, u32, u32>"(%arg0: !llvm.struct<packed (i32, i32, i32, i32, i32)>) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i32, i32, i32, i32, i32)> 
    llvm.call @print_u32(%0) : (i32) -> ()
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<packed (i32, i32, i32, i32, i32)> 
    llvm.call @print_u32(%1) : (i32) -> ()
    %2 = llvm.extractvalue %arg0[2] : !llvm.struct<packed (i32, i32, i32, i32, i32)> 
    llvm.call @print_u32(%2) : (i32) -> ()
    %3 = llvm.extractvalue %arg0[3] : !llvm.struct<packed (i32, i32, i32, i32, i32)> 
    llvm.call @print_u32(%3) : (i32) -> ()
    %4 = llvm.extractvalue %arg0[4] : !llvm.struct<packed (i32, i32, i32, i32, i32)> 
    llvm.call @print_u32(%4) : (i32) -> ()
    llvm.return
  }
  llvm.func internal @"print_Tuple<Tuple<u32, u32, u32, u32, u32>>"(%arg0: !llvm.struct<packed (struct<packed (i32, i32, i32, i32, i32)>)>) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (struct<packed (i32, i32, i32, i32, i32)>)> 
    llvm.call @"print_Tuple<u32, u32, u32, u32, u32>"(%0) : (!llvm.struct<packed (i32, i32, i32, i32, i32)>) -> ()
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
  llvm.func internal @"print_Array<felt252>"(%arg0: !llvm.struct<packed (i32, i32, ptr)>) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[2] : !llvm.struct<packed (i32, i32, ptr)> 
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i32, i32, ptr)> 
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
  llvm.func internal @"print_core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>"(%arg0: !llvm.struct<packed (i16, array<20 x i8>)>) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<20 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %arg0, %1 : !llvm.struct<packed (i16, array<20 x i8>)>, !llvm.ptr
    %2 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<20 x i8>)>
    %3 = llvm.load %2 : !llvm.ptr -> i16
    llvm.switch %3 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb1:  // pred: ^bb0
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (struct<packed (i32, i32, i32, i32, i32)>)>)>
    %5 = llvm.load %4 : !llvm.ptr -> !llvm.struct<packed (struct<packed (i32, i32, i32, i32, i32)>)>
    llvm.call @"print_Tuple<Tuple<u32, u32, u32, u32, u32>>"(%5) : (!llvm.struct<packed (struct<packed (i32, i32, i32, i32, i32)>)>) -> ()
    llvm.return
  ^bb2:  // pred: ^bb0
    %6 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %7 = llvm.load %6 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.call @"print_Array<felt252>"(%7) : (!llvm.struct<packed (i32, i32, ptr)>) -> ()
    llvm.return
  ^bb3:  // pred: ^bb0
    llvm.return
  }
  llvm.func @main() attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"example_array::example_array::main"() : () -> !llvm.struct<packed (i16, array<20 x i8>)>
    llvm.call @"print_core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>"(%0) : (!llvm.struct<packed (i16, array<20 x i8>)>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
  llvm.func @"example_array::example_array::main"() -> !llvm.struct<packed (i16, array<20 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb17
  ^bb1:  // pred: ^bb17
    %0 = llvm.extractvalue %36[2] : !llvm.struct<packed (i32, i32, ptr)> 
    %1 = llvm.load %0 : !llvm.ptr -> i32
    %2 = llvm.sub %39, %37  : i32
    %3 = llvm.getelementptr %0[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %4 = llvm.zext %2 : i32 to i64
    %5 = llvm.mlir.constant(4 : i64) : i64
    %6 = llvm.mul %4, %5  : i64
    %7 = llvm.call @memmove(%0, %3, %6) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %2, %36[0] : !llvm.struct<packed (i32, i32, ptr)> 
    %9 = llvm.insertvalue %7, %8[2] : !llvm.struct<packed (i32, i32, ptr)> 
    llvm.br ^bb18(%9, %1 : !llvm.struct<packed (i32, i32, ptr)>, i32)
  ^bb2:  // pred: ^bb20
    %10 = llvm.getelementptr inbounds %57[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32)>)>
    %11 = llvm.load %10 : !llvm.ptr -> !llvm.struct<packed (i32)>
    llvm.br ^bb21(%53, %11 : !llvm.struct<packed (i32, i32, ptr)>, !llvm.struct<packed (i32)>)
  ^bb3:  // pred: ^bb20
    %12 = llvm.getelementptr inbounds %57[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %13 = llvm.load %12 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb30(%13 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb4:  // pred: ^bb20
    llvm.unreachable
  ^bb5:  // pred: ^bb21
    %14 = llvm.getelementptr inbounds %66[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32)>)>
    %15 = llvm.load %14 : !llvm.ptr -> !llvm.struct<packed (i32)>
    llvm.br ^bb22(%62, %60, %15 : i32, !llvm.struct<packed (i32, i32, ptr)>, !llvm.struct<packed (i32)>)
  ^bb6:  // pred: ^bb21
    %16 = llvm.getelementptr inbounds %66[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %17 = llvm.load %16 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb29(%17 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb7:  // pred: ^bb21
    llvm.unreachable
  ^bb8:  // pred: ^bb22
    %18 = llvm.getelementptr inbounds %76[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32)>)>
    %19 = llvm.load %18 : !llvm.ptr -> !llvm.struct<packed (i32)>
    llvm.br ^bb23(%69, %72, %70, %19 : i32, i32, !llvm.struct<packed (i32, i32, ptr)>, !llvm.struct<packed (i32)>)
  ^bb9:  // pred: ^bb22
    %20 = llvm.getelementptr inbounds %76[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %21 = llvm.load %20 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb28(%21 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb10:  // pred: ^bb22
    llvm.unreachable
  ^bb11:  // pred: ^bb23
    %22 = llvm.getelementptr inbounds %87[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32)>)>
    %23 = llvm.load %22 : !llvm.ptr -> !llvm.struct<packed (i32)>
    llvm.br ^bb24(%79, %80, %83, %81, %23 : i32, i32, i32, !llvm.struct<packed (i32, i32, ptr)>, !llvm.struct<packed (i32)>)
  ^bb12:  // pred: ^bb23
    %24 = llvm.getelementptr inbounds %87[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %25 = llvm.load %24 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb27(%25 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb13:  // pred: ^bb23
    llvm.unreachable
  ^bb14:  // pred: ^bb24
    %26 = llvm.getelementptr inbounds %99[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32)>)>
    %27 = llvm.load %26 : !llvm.ptr -> !llvm.struct<packed (i32)>
    llvm.br ^bb25(%90, %91, %92, %95, %27 : i32, i32, i32, i32, !llvm.struct<packed (i32)>)
  ^bb15:  // pred: ^bb24
    %28 = llvm.getelementptr inbounds %99[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %29 = llvm.load %28 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb26(%29 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb16:  // pred: ^bb24
    llvm.unreachable
  ^bb17:  // pred: ^bb0
    %30 = llvm.call @"array_new<u32>"() : () -> !llvm.struct<packed (i32, i32, ptr)>
    %31 = llvm.mlir.constant(1 : i32) : i32
    %32 = llvm.call @"array_append<u32>"(%30, %31) : (!llvm.struct<packed (i32, i32, ptr)>, i32) -> !llvm.struct<packed (i32, i32, ptr)>
    %33 = llvm.mlir.constant(2 : i32) : i32
    %34 = llvm.call @"array_append<u32>"(%32, %33) : (!llvm.struct<packed (i32, i32, ptr)>, i32) -> !llvm.struct<packed (i32, i32, ptr)>
    %35 = llvm.mlir.constant(3 : i32) : i32
    %36 = llvm.call @"array_append<u32>"(%34, %35) : (!llvm.struct<packed (i32, i32, ptr)>, i32) -> !llvm.struct<packed (i32, i32, ptr)>
    %37 = llvm.mlir.constant(1 : i32) : i32
    %38 = llvm.mlir.constant(0 : i32) : i32
    %39 = llvm.extractvalue %36[0] : !llvm.struct<packed (i32, i32, ptr)> 
    %40 = llvm.icmp "uge" %39, %37 : i32
    llvm.cond_br %40, ^bb1, ^bb19(%36 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb18(%41: !llvm.struct<packed (i32, i32, ptr)>, %42: i32):  // pred: ^bb1
    %43 = llvm.call @"enum_init<core::option::Option::<core::integer::u32>, 0>"(%42) : (i32) -> !llvm.struct<packed (i16, array<4 x i8>)>
    llvm.br ^bb20(%41 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb19(%44: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb17
    %45 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    %46 = llvm.call @"enum_init<core::option::Option::<core::integer::u32>, 1>"(%45) : (!llvm.struct<packed ()>) -> !llvm.struct<packed (i16, array<4 x i8>)>
    llvm.br ^bb20(%44 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb20(%47: !llvm.struct<packed (i32, i32, ptr)>):  // 2 preds: ^bb18, ^bb19
    %48 = llvm.mlir.constant(7 : i32) : i32
    %49 = llvm.call @"array_append<u32>"(%47, %48) : (!llvm.struct<packed (i32, i32, ptr)>, i32) -> !llvm.struct<packed (i32, i32, ptr)>
    %50 = llvm.mlir.constant(5 : i32) : i32
    %51 = llvm.call @"array_append<u32>"(%49, %50) : (!llvm.struct<packed (i32, i32, ptr)>, i32) -> !llvm.struct<packed (i32, i32, ptr)>
    %52 = llvm.call @"array_len<u32>"(%51) : (!llvm.struct<packed (i32, i32, ptr)>) -> i32
    %53 = llvm.call @"array_append<u32>"(%51, %52) : (!llvm.struct<packed (i32, i32, ptr)>, i32) -> !llvm.struct<packed (i32, i32, ptr)>
    %54 = llvm.mlir.constant(0 : i32) : i32
    %55 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%53, %54) : (!llvm.struct<packed (i32, i32, ptr)>, i32) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %56 = llvm.mlir.constant(1 : i64) : i64
    %57 = llvm.alloca %56 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %55, %57 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %58 = llvm.getelementptr inbounds %57[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %59 = llvm.load %58 : !llvm.ptr -> i16
    llvm.switch %59 : i16, ^bb4 [
      0: ^bb2,
      1: ^bb3
    ]
  ^bb21(%60: !llvm.struct<packed (i32, i32, ptr)>, %61: !llvm.struct<packed (i32)>):  // pred: ^bb2
    %62 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%61) : (!llvm.struct<packed (i32)>) -> i32
    %63 = llvm.mlir.constant(1 : i32) : i32
    %64 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%60, %63) : (!llvm.struct<packed (i32, i32, ptr)>, i32) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %65 = llvm.mlir.constant(1 : i64) : i64
    %66 = llvm.alloca %65 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %64, %66 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %67 = llvm.getelementptr inbounds %66[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %68 = llvm.load %67 : !llvm.ptr -> i16
    llvm.switch %68 : i16, ^bb7 [
      0: ^bb5,
      1: ^bb6
    ]
  ^bb22(%69: i32, %70: !llvm.struct<packed (i32, i32, ptr)>, %71: !llvm.struct<packed (i32)>):  // pred: ^bb5
    %72 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%71) : (!llvm.struct<packed (i32)>) -> i32
    %73 = llvm.mlir.constant(2 : i32) : i32
    %74 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%70, %73) : (!llvm.struct<packed (i32, i32, ptr)>, i32) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %75 = llvm.mlir.constant(1 : i64) : i64
    %76 = llvm.alloca %75 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %74, %76 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %77 = llvm.getelementptr inbounds %76[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %78 = llvm.load %77 : !llvm.ptr -> i16
    llvm.switch %78 : i16, ^bb10 [
      0: ^bb8,
      1: ^bb9
    ]
  ^bb23(%79: i32, %80: i32, %81: !llvm.struct<packed (i32, i32, ptr)>, %82: !llvm.struct<packed (i32)>):  // pred: ^bb8
    %83 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%82) : (!llvm.struct<packed (i32)>) -> i32
    %84 = llvm.mlir.constant(3 : i32) : i32
    %85 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%81, %84) : (!llvm.struct<packed (i32, i32, ptr)>, i32) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %86 = llvm.mlir.constant(1 : i64) : i64
    %87 = llvm.alloca %86 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %85, %87 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %88 = llvm.getelementptr inbounds %87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %89 = llvm.load %88 : !llvm.ptr -> i16
    llvm.switch %89 : i16, ^bb13 [
      0: ^bb11,
      1: ^bb12
    ]
  ^bb24(%90: i32, %91: i32, %92: i32, %93: !llvm.struct<packed (i32, i32, ptr)>, %94: !llvm.struct<packed (i32)>):  // pred: ^bb11
    %95 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%94) : (!llvm.struct<packed (i32)>) -> i32
    %96 = llvm.mlir.constant(4 : i32) : i32
    %97 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%93, %96) : (!llvm.struct<packed (i32, i32, ptr)>, i32) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %98 = llvm.mlir.constant(1 : i64) : i64
    %99 = llvm.alloca %98 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %97, %99 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %100 = llvm.getelementptr inbounds %99[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %101 = llvm.load %100 : !llvm.ptr -> i16
    llvm.switch %101 : i16, ^bb16 [
      0: ^bb14,
      1: ^bb15
    ]
  ^bb25(%102: i32, %103: i32, %104: i32, %105: i32, %106: !llvm.struct<packed (i32)>):  // pred: ^bb14
    %107 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%106) : (!llvm.struct<packed (i32)>) -> i32
    %108 = llvm.call @"struct_construct<Tuple<u32, u32, u32, u32, u32>>"(%102, %103, %104, %105, %107) : (i32, i32, i32, i32, i32) -> !llvm.struct<packed (i32, i32, i32, i32, i32)>
    %109 = llvm.call @"struct_construct<Tuple<Tuple<u32, u32, u32, u32, u32>>>"(%108) : (!llvm.struct<packed (i32, i32, i32, i32, i32)>) -> !llvm.struct<packed (struct<packed (i32, i32, i32, i32, i32)>)>
    %110 = llvm.call @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 0>"(%109) : (!llvm.struct<packed (struct<packed (i32, i32, i32, i32, i32)>)>) -> !llvm.struct<packed (i16, array<20 x i8>)>
    llvm.return %110 : !llvm.struct<packed (i16, array<20 x i8>)>
  ^bb26(%111: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb15
    %112 = llvm.call @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(%111) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<20 x i8>)>
    llvm.return %112 : !llvm.struct<packed (i16, array<20 x i8>)>
  ^bb27(%113: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb12
    %114 = llvm.call @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(%113) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<20 x i8>)>
    llvm.return %114 : !llvm.struct<packed (i16, array<20 x i8>)>
  ^bb28(%115: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb9
    %116 = llvm.call @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(%115) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<20 x i8>)>
    llvm.return %116 : !llvm.struct<packed (i16, array<20 x i8>)>
  ^bb29(%117: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb6
    %118 = llvm.call @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(%117) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<20 x i8>)>
    llvm.return %118 : !llvm.struct<packed (i16, array<20 x i8>)>
  ^bb30(%119: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb3
    %120 = llvm.call @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(%119) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<20 x i8>)>
    llvm.return %120 : !llvm.struct<packed (i16, array<20 x i8>)>
  }
  llvm.func @"_mlir_ciface_example_array::example_array::main"(%arg0: !llvm.ptr<struct<packed (i16, array<20 x i8>)>>) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"example_array::example_array::main"() : () -> !llvm.struct<packed (i16, array<20 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed (i16, array<20 x i8>)>>
    llvm.return
  }
  llvm.func @"core::array::ArrayIndex::<core::integer::u32>::index"(%arg0: !llvm.struct<packed (i32, i32, ptr)>, %arg1: i32) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4(%arg0, %arg1 : !llvm.struct<packed (i32, i32, ptr)>, i32)
  ^bb1:  // pred: ^bb4
    %0 = llvm.getelementptr inbounds %8[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32)>)>
    %1 = llvm.load %0 : !llvm.ptr -> !llvm.struct<packed (i32)>
    llvm.br ^bb5(%1 : !llvm.struct<packed (i32)>)
  ^bb2:  // pred: ^bb4
    %2 = llvm.getelementptr inbounds %8[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %3 = llvm.load %2 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb6(%3 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb3:  // pred: ^bb4
    llvm.unreachable
  ^bb4(%4: !llvm.struct<packed (i32, i32, ptr)>, %5: i32):  // pred: ^bb0
    %6 = llvm.call @"core::array::array_at::<core::integer::u32>"(%4, %5) : (!llvm.struct<packed (i32, i32, ptr)>, i32) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.alloca %7 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %6, %8 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %9 = llvm.getelementptr inbounds %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %10 = llvm.load %9 : !llvm.ptr -> i16
    llvm.switch %10 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb5(%11: !llvm.struct<packed (i32)>):  // pred: ^bb1
    %12 = llvm.call @"struct_deconstruct<Tuple<Box<u32>>>"(%11) : (!llvm.struct<packed (i32)>) -> i32
    %13 = llvm.call @"struct_construct<Tuple<u32>>"(%12) : (i32) -> !llvm.struct<packed (i32)>
    %14 = llvm.call @"enum_init<core::PanicResult::<(@core::integer::u32)>, 0>"(%13) : (!llvm.struct<packed (i32)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %14 : !llvm.struct<packed (i16, array<16 x i8>)>
  ^bb6(%15: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb2
    %16 = llvm.call @"enum_init<core::PanicResult::<(@core::integer::u32)>, 1>"(%15) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %16 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::array::ArrayIndex::<core::integer::u32>::index"(%arg0: !llvm.ptr<struct<packed (i16, array<16 x i8>)>>, %arg1: !llvm.struct<packed (i32, i32, ptr)>, %arg2: i32) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%arg1, %arg2) : (!llvm.struct<packed (i32, i32, ptr)>, i32) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed (i16, array<16 x i8>)>>
    llvm.return
  }
  llvm.func @"core::array::array_at::<core::integer::u32>"(%arg0: !llvm.struct<packed (i32, i32, ptr)>, %arg1: i32) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb2(%arg0, %arg1 : !llvm.struct<packed (i32, i32, ptr)>, i32)
  ^bb1:  // pred: ^bb2
    %0 = llvm.extractvalue %3[2] : !llvm.struct<packed (i32, i32, ptr)> 
    %1 = llvm.getelementptr %0[%4] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %2 = llvm.load %1 : !llvm.ptr -> i32
    llvm.br ^bb3(%2 : i32)
  ^bb2(%3: !llvm.struct<packed (i32, i32, ptr)>, %4: i32):  // pred: ^bb0
    %5 = llvm.extractvalue %3[0] : !llvm.struct<packed (i32, i32, ptr)> 
    %6 = llvm.icmp "ult" %4, %5 : i32
    llvm.cond_br %6, ^bb1, ^bb4
  ^bb3(%7: i32):  // pred: ^bb1
    %8 = llvm.call @"struct_construct<Tuple<Box<u32>>>"(%7) : (i32) -> !llvm.struct<packed (i32)>
    %9 = llvm.call @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 0>"(%8) : (!llvm.struct<packed (i32)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %9 : !llvm.struct<packed (i16, array<16 x i8>)>
  ^bb4:  // pred: ^bb2
    %10 = llvm.call @"array_new<felt252>"() : () -> !llvm.struct<packed (i32, i32, ptr)>
    %11 = llvm.mlir.constant(1637570914057682275393755530660268060279989363 : i256) : i256
    %12 = llvm.call @"array_append<felt252>"(%10, %11) : (!llvm.struct<packed (i32, i32, ptr)>, i256) -> !llvm.struct<packed (i32, i32, ptr)>
    %13 = llvm.call @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 1>"(%12) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %13 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::array::array_at::<core::integer::u32>"(%arg0: !llvm.ptr<struct<packed (i16, array<16 x i8>)>>, %arg1: !llvm.struct<packed (i32, i32, ptr)>, %arg2: i32) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::array::array_at::<core::integer::u32>"(%arg1, %arg2) : (!llvm.struct<packed (i32, i32, ptr)>, i32) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed (i16, array<16 x i8>)>>
    llvm.return
  }
}
