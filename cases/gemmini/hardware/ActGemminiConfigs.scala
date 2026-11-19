package chipyard

import org.chipsalliance.cde.config.Config
import freechips.rocketchip.tile.OpcodeSet
import gemmini.{DefaultGemminiConfig, GemminiConfigs}
import gemmini.Arithmetic.SIntArithmetic

// Identical datapaths, independent local SRAMs, distinct command destinations.
// Both DMA clients join the ordinary shared system-memory fabric. No new ISA
// opcode implements DoubleGEMM: the software lowering issues to custom3/custom2.
class ActDualGemminiRocketConfig extends Config(
  new DefaultGemminiConfig(GemminiConfigs.defaultConfig.copy(
    opcodes = OpcodeSet.custom3,
    headerFileName = "gemmini_lane0_params.h",
    use_shared_ext_mem = false,
    clock_gate = false)) ++
  new DefaultGemminiConfig(GemminiConfigs.defaultConfig.copy(
    opcodes = OpcodeSet.custom2,
    headerFileName = "gemmini_lane1_params.h",
    use_shared_ext_mem = false,
    clock_gate = false)) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)
