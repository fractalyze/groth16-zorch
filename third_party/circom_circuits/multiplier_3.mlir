// Multiplier3 circuit: out = a * b * c
//
// Uses sub-component Multiplier for a*b, then multiplies by c.
//
// Signal layout (8 total):
//   0: constant (= 1)
//   1: out     (Multiplier3, output)
//   2: a       (Multiplier3, input)
//   3: b       (Multiplier3, input)
//   4: c       (Multiplier3, input)
//   5: out     (Multiplier, sub-component output = a*b)
//   6: a       (Multiplier, sub-component input copy)
//   7: b       (Multiplier, sub-component input copy)
//
// Witness (6 values, w2s = [0,1,2,3,4,5]):
//   [1, 60, 3, 4, 5, 12]  for inputs a=3, b=4, c=5

// --- Template: Multiplier (id=0) ---

  func.func @Multiplier_0_create(%signalOffset: index, %componentIdx: index,
      %signals: memref<?x!F>, %subcmps: memref<?xindex>) {
    %numInputs = arith.constant 2 : index
    %templateId = arith.constant 0 : index
    func.call @"circom.initComponent"(%subcmps, %componentIdx, %signalOffset,
        %numInputs, %templateId)
        : (memref<?xindex>, index, index, index, index) -> ()
    func.return
  }

  func.func @Multiplier_0_run(%ctx_index: index, %signals: memref<?x!F>,
      %subcmps: memref<?xindex>) {
    %lvar_static = memref.alloca() : memref<1x!F>
    %lvar = memref.cast %lvar_static : memref<1x!F> to memref<?x!F>
    %my_subcmps_static = memref.alloca() : memref<1xindex>
    %my_subcmps = memref.cast %my_subcmps_static : memref<1xindex> to memref<?xindex>
    %mySignalStart = func.call @"circom.getSignalStart"(%subcmps, %ctx_index)
        : (memref<?xindex>, index) -> index
    // out = input_a * input_b
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %off_a = arith.addi %c1, %mySignalStart : index
    %a = memref.load %signals[%off_a] : memref<?x!F>
    %off_b = arith.addi %c2, %mySignalStart : index
    %b = memref.load %signals[%off_b] : memref<?x!F>
    %out = field.mul %a, %b : !F
    %off_out = arith.addi %c0, %mySignalStart : index
    memref.store %out, %signals[%off_out] : memref<?x!F>
    func.return
  }

// --- Template: Multiplier3 (id=1) ---

  func.func @Multiplier3_1_create(%signalOffset: index, %componentIdx: index,
      %signals: memref<?x!F>, %subcmps: memref<?xindex>) {
    %numInputs = arith.constant 3 : index
    %templateId = arith.constant 1 : index
    func.call @"circom.initComponent"(%subcmps, %componentIdx, %signalOffset,
        %numInputs, %templateId)
        : (memref<?xindex>, index, index, index, index) -> ()
    func.return
  }

  func.func @Multiplier3_1_run(%ctx_index: index, %signals: memref<?x!F>,
      %subcmps: memref<?xindex>) {
    %lvar_static = memref.alloca() : memref<1x!F>
    %lvar = memref.cast %lvar_static : memref<1x!F> to memref<?x!F>
    %my_subcmps_static = memref.alloca() : memref<1xindex>
    %my_subcmps = memref.cast %my_subcmps_static : memref<1xindex> to memref<?xindex>
    %mySignalStart = func.call @"circom.getSignalStart"(%subcmps, %ctx_index)
        : (memref<?xindex>, index) -> index

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    // Create sub-component Multiplier (cmp_idx=1) at signal offset mySignalStart+4
    %subcmp_offset = arith.addi %c4, %mySignalStart : index
    %cmp1 = arith.constant 1 : index
    func.call @Multiplier_0_create(%subcmp_offset, %cmp1, %signals, %subcmps)
        : (index, index, memref<?x!F>, memref<?xindex>) -> ()

    // Write a to sub-component input: signals[subcmp_offset+1] = signals[mySignalStart+1]
    %src_a = arith.addi %c1, %mySignalStart : index
    %a = memref.load %signals[%src_a] : memref<?x!F>
    %dst_a = arith.addi %c1, %subcmp_offset : index
    memref.store %a, %signals[%dst_a] : memref<?x!F>
    func.call @"circom.decrementInputCounter"(%subcmps, %cmp1)
        : (memref<?xindex>, index) -> ()

    // Write b to sub-component input: signals[subcmp_offset+2] = signals[mySignalStart+2]
    %src_b = arith.addi %c2, %mySignalStart : index
    %b = memref.load %signals[%src_b] : memref<?x!F>
    %dst_b = arith.addi %c2, %subcmp_offset : index
    memref.store %b, %signals[%dst_b] : memref<?x!F>
    func.call @"circom.decrementInputCounterAndRun"(%subcmps, %cmp1)
        : (memref<?xindex>, index) -> ()

    // Run sub-component: computes signals[subcmp_offset+0] = a * b
    func.call @Multiplier_0_run(%cmp1, %signals, %subcmps)
        : (index, memref<?x!F>, memref<?xindex>) -> ()

    // Read sub-component output (a*b)
    %src_ab = arith.addi %c0, %subcmp_offset : index
    %ab = memref.load %signals[%src_ab] : memref<?x!F>

    // Read c
    %src_c = arith.addi %c3, %mySignalStart : index
    %c_val = memref.load %signals[%src_c] : memref<?x!F>

    // out = ab * c
    %out = field.mul %ab, %c_val : !F
    %dst_out = arith.addi %c0, %mySignalStart : index
    memref.store %out, %signals[%dst_out] : memref<?x!F>

    func.return
  }

// --- Dispatch and IO mapping ---

func.func @"circom.getSubcmpMappedOffset"(%subcmps: memref<?xindex>,
    %cmp_index: index, %signal_code: index) -> index {
  %c0 = arith.constant 0 : index
  return %c0 : index
}

func.func @"circom.dispatchRun"(%ctx_index: index, %signals: memref<?x!F>,
    %subcmps: memref<?xindex>) {
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %base = arith.muli %ctx_index, %c3 : index
  %slot2 = arith.addi %base, %c2 : index
  %template_id = memref.load %subcmps[%slot2] : memref<?xindex>
  %tid_0 = arith.constant 0 : index
  %is_tid_0 = arith.cmpi eq, %template_id, %tid_0 : index
  scf.if %is_tid_0 {
    func.call @Multiplier_0_run(%ctx_index, %signals, %subcmps)
        : (index, memref<?x!F>, memref<?xindex>) -> ()
  }
  %tid_1 = arith.constant 1 : index
  %is_tid_1 = arith.cmpi eq, %template_id, %tid_1 : index
  scf.if %is_tid_1 {
    func.call @Multiplier3_1_run(%ctx_index, %signals, %subcmps)
        : (index, memref<?x!F>, memref<?xindex>) -> ()
  }
  return
}

// --- Entry point ---

func.func @circuit_main(%signals: memref<?x!F>, %subcmps: memref<?xindex>)
    attributes {llvm.emit_c_interface} {
  %c1_offset = arith.constant 1 : index
  %c0_cmp = arith.constant 0 : index
  func.call @Multiplier3_1_create(%c1_offset, %c0_cmp, %signals, %subcmps)
      : (index, index, memref<?x!F>, memref<?xindex>) -> ()
  func.call @Multiplier3_1_run(%c0_cmp, %signals, %subcmps)
      : (index, memref<?x!F>, memref<?xindex>) -> ()
  func.return
}

// --- Metadata ---

func.func @circom_get_total_signals() -> index attributes {llvm.emit_c_interface} {
  %c = arith.constant 8 : index
  return %c : index
}
func.func @circom_get_num_components() -> index attributes {llvm.emit_c_interface} {
  %c = arith.constant 2 : index
  return %c : index
}
func.func @circom_get_num_outputs() -> index attributes {llvm.emit_c_interface} {
  %c = arith.constant 1 : index
  return %c : index
}
func.func @circom_get_witness_size() -> index attributes {llvm.emit_c_interface} {
  %c = arith.constant 6 : index
  return %c : index
}
func.func @circom_get_num_inputs() -> index attributes {llvm.emit_c_interface} {
  %c = arith.constant 3 : index
  return %c : index
}
