#include <vector>
#include <algorithm>
#include <cstdio>
#include "softplus_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint32_t BLOCK_SIZE = 32; 
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    SoftplusTilingData tiling;
    float beta = *(context->GetAttrs()->GetFloat(0));
    float threshold = *(context->GetAttrs()->GetFloat(1));

    // 平台信息
    uint64_t ub_size = 0;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    uint32_t coreNum = platform.GetCoreNum();



    /* ==================== 3. 输入信息 ==================== */
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize(); //元素个数
    auto dt = context->GetInputTensor(0)->GetDataType();
     

    uint32_t sizeofType = 0;
    uint32_t dataNum = 0; //要申请几个空间
    constexpr uint32_t BUFFER_NUM = 2;

    if (dt == ge::DT_FLOAT16) {
        sizeofType = 2;
        dataNum = 5;
        if (beta != 1) {
            context->SetTilingKey(0);
        } else {
            context->SetTilingKey(10);   // FP16
        }
        
    } else if (dt == ge::DT_FLOAT) {
        sizeofType = 4;
        dataNum = 4;
        if (beta != 1) {
            context->SetTilingKey(1);
        } else {
            context->SetTilingKey(11);   // FP32
        }   // FP32
    } else if (dt == ge::DT_BF16) {
        sizeofType = 2;
        dataNum = 5;
        if (beta != 1) {
            context->SetTilingKey(2);
        } else {
            context->SetTilingKey(12);   // BF16
        }   // BF16
    } else {
        return ge::GRAPH_FAILED;
    }
    /* ==================== 4. Core 切分 ==================== */
    uint32_t block_length = BLOCK_SIZE / sizeofType;  // 一个block几个数据

    uint32_t totalLengthAlign32 = (totalLength + block_length - 1) / block_length * block_length; 
    uint32_t total_block_num = totalLengthAlign32 / block_length;



    uint32_t tiling_block_num = ((ub_size / BLOCK_SIZE) / 2) / dataNum; 
    
    coreNum = coreNum < total_block_num ? coreNum : total_block_num;
    coreNum = coreNum >= 1 ? coreNum : 1;
    
    
    tiling_block_num = tiling_block_num <= 8 ? tiling_block_num : tiling_block_num / 8 * 8; //一个tile几个block
    
    uint32_t small_core_block_num = total_block_num / coreNum; //一个小核多少个Block
    uint32_t big_core_num = total_block_num - small_core_block_num * coreNum;


    // // uint32_t tiling_length = tiling_block_num * block_length; //一个tiling几个数据
    // uint32_t core_length = totalLength / (block_length * 8) * 8; //有多少整的数据
    // uint32_t core_remain_length = totalLength - core_length; //零散的数据



    tiling.set_totalLength(totalLength); //总长度
    tiling.set_coreNum(coreNum); //几个核
    tiling.set_tiling_block_num(tiling_block_num); //一个tiling几个block
    tiling.set_block_length(block_length); //一个block几个元素
    tiling.set_small_core_block_num(small_core_block_num); //一个小核几个block
    tiling.set_big_core_num(big_core_num); //几个大核
    tiling.set_beta(beta);
    tiling.set_threshold(threshold);

    /* ==================== 7. 提交 ==================== */
    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(
        tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class Softplus : public OpDef {
public:
    explicit Softplus(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("beta").AttrType(OPTIONAL).Float(1.0);
        this->Attr("threshold").AttrType(OPTIONAL).Float(20.0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Softplus);
}
