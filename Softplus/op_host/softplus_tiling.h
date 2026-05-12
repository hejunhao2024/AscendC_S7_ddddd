
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SoftplusTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, coreNum);
  TILING_DATA_FIELD_DEF(uint32_t, tiling_block_num);


  TILING_DATA_FIELD_DEF(uint32_t, block_length);
  TILING_DATA_FIELD_DEF(uint32_t, small_core_block_num);
  TILING_DATA_FIELD_DEF(uint32_t, big_core_num);


  TILING_DATA_FIELD_DEF(float, beta);
  TILING_DATA_FIELD_DEF(float, threshold);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Softplus, SoftplusTilingData)
}
