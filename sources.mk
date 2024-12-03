#
# This file is generated!  Editing it directly is a bad idea.
#
# Generated on: Sat 21 Aug 2021 06:39:39 AM CEST
#
###############################################################################

# Library Object files - only used for LIB
OBJ := \
  Target/Subst.o  \
  Target/BufferObject.o  \
  Target/Pretty.o  \
  Target/instr/ALUOp.o  \
  Target/instr/RegOrImm.o  \
  Target/instr/Conditions.o  \
  Target/instr/Reg.o  \
  Target/instr/ALUInstruction.o  \
  Target/instr/Instr.o  \
  Target/instr/Label.o  \
  Target/instr/Imm.o  \
  Target/instr/Mnemonics.o  \
  Target/SmallLiteral.o  \
  Target/EmuSupport.o  \
  Target/Emulator.o  \
  Target/Satisfy.o  \
  BaseKernel.o  \
  Source/Lang.o  \
  Source/Cond.o  \
  Source/OpItems.o  \
  Source/Interpreter.o  \
  Source/Translate.o  \
  Source/Ptr.o  \
  Source/Pretty.o  \
  Source/BExpr.o  \
  Source/Int.o  \
  Source/Functions.o  \
  Source/gather.o  \
  Source/Op.o  \
  Source/Expr.o  \
  Source/StmtStack.o  \
  Source/CExpr.o  \
  Source/Float.o  \
  Source/Complex.o  \
  Source/Var.o  \
  Source/Stmt.o  \
  Support/debug.o  \
  Support/Timer.o  \
  Support/InstructionComment.o  \
  Support/basics.o  \
  Support/RegIdSet.o  \
  Support/pgm.o  \
  Support/Helpers.o  \
  Support/Platform.o  \
  Support/HeapManager.o  \
  SourceTranslate.o  \
  Common/SharedArray.o  \
  Common/BufferObject.o  \
  Common/CompileData.o  \
  Kernels/DotVector.o  \
  Kernels/Cursor.o  \
  Kernels/Rot3D.o  \
  Kernels/ComplexDotVector.o  \
  Kernels/Matrix.o  \
  Liveness/Range.o  \
  Liveness/LiveSet.o  \
  Liveness/UseDef.o  \
  Liveness/Optimizations.o  \
  Liveness/RegUsage.o  \
  Liveness/Liveness.o  \
  Liveness/CFG.o  \
  LibSettings.o  \
  v3d/PerformanceCounters.o  \
  v3d/v3d.o  \
  v3d/BufferObject.o  \
  v3d/SourceTranslate.o  \
  v3d/instr/Source.o  \
  v3d/instr/RFAddress.o  \
  v3d/instr/OpItems.o  \
  v3d/instr/Snippets.o  \
  v3d/instr/SmallImm.o  \
  v3d/instr/Encode.o  \
  v3d/instr/Register.o  \
  v3d/instr/Instr.o  \
  v3d/instr/Mnemonics.o  \
  v3d/Driver.o  \
  v3d/RegisterMapping.o  \
  v3d/KernelDriver.o  \
  vc4/PerformanceCounters.o  \
  vc4/Mailbox.o  \
  vc4/BufferObject.o  \
  vc4/SourceTranslate.o  \
  vc4/RegAlloc.o  \
  vc4/Invoke.o  \
  vc4/RegisterMap.o  \
  vc4/Instr.o  \
  vc4/DMA/Helpers.o  \
  vc4/DMA/DMA.o  \
  vc4/DMA/LoadStore.o  \
  vc4/DMA/Operations.o  \
  vc4/vc4.o  \
  vc4/KernelDriver.o  \
  KernelDriver.o  \
  v3d/instr/v3d_api.o  \
  vc4/dump_instr.o  \

# All programs in the Examples *and Tools* directory
EXAMPLES := \
  HeatMap  \
  OET  \
  Tri  \
  ID  \
  ReqRecv  \
  GCD  \
  Hello  \
  Mandelbrot  \
  DMA  \
  Rot3D  \
  Matrix  \
  detectPlatform  \
  Hello2

# support files for examples
EXAMPLES_EXTRA := \
  Examples/Support/Settings.o  \
# support files for tests
TESTS_FILES := \
  Tests/testRegMap.o  \
  Tests/testImmediates.o  \
  Tests/testBO.o  \
  Tests/testMatrix.o  \
  Tests/testFFT.o  \
  Tests/testV3d.o  \
  Tests/testRot3D.o  \
  Tests/testPrefetch.o  \
  Tests/testFunctions.o  \
  Tests/support/ProfileOutput.o  \
  Tests/support/disasm_kernel.o  \
  Tests/support/rotate_kernel.o  \
  Tests/support/support.o  \
  Tests/support/matrix_support.o  \
  Tests/support/summation_kernel.o  \
  Tests/support/dft_support.o  \
  Tests/testSFU.o  \
  Tests/testDFT.o  \
  Tests/testConditionCodes.o  \
  Tests/testMain.o  \
  Tests/testDSL.o  \
  Tests/testCmdLine.o  \
  Tests/support/qpu_disasm.o  \

