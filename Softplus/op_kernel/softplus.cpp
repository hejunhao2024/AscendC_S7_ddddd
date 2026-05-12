#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelSoftplusFP16 {
public:
    __aicore__ inline KernelSoftplusFP16() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t coreNum, uint32_t tiling_block_num, uint32_t block_length, uint32_t small_core_block_num, uint32_t big_core_num)
    {
        uint32_t total_length = totalLength;
        
        this->tiling_length = tiling_block_num * block_length;
        int64_t core_idx = GetBlockIdx();
        uint32_t start = core_idx <= big_core_num ? core_idx * (small_core_block_num + 1) : big_core_num * (small_core_block_num + 1) + (core_idx - big_core_num) * small_core_block_num;
        start = start * block_length; //偏移地址
        
        this->core_length = core_idx >= big_core_num ? small_core_block_num * block_length : (small_core_block_num + 1) * block_length;
        
        xGM.SetGlobalBuffer((__gm__ half*)x + start, this->core_length);
        yGM.SetGlobalBuffer((__gm__ half*)y + start, this->core_length);

        this->tile_num = this->core_length / this->tiling_length + (this->core_length % this->tiling_length > 0); //一共多少个tile
       
        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tiling_length * sizeof(half));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tiling_length * sizeof(half));
        pipe.InitBuffer(tmpbxBuffer, this->tiling_length * sizeof(float));
        pipe.InitBuffer(selMaskBuffer, this->tiling_length * sizeof(uint8_t));  
        

    }

    __aicore__ inline void Process(float beta, float threshold) { 
        for (int32_t i = 0; i < this->tile_num - 1; ++i) {       
            CopyIn(i, this->tiling_length);
            Compute(i, this->tiling_length, beta, threshold);  
            CopyOut(i, this->tiling_length);  
        }
        uint32_t length = this->core_length - this->tiling_length * (this->tile_num - 1);
        CopyIn(this->tile_num - 1, length);
        Compute(this->tile_num - 1, length, beta, threshold); 
        CopyOut(this->tile_num - 1, length);  
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<half> x = Q_x.AllocTensor<half>();    
        DataCopy(x, xGM[progress * this->tiling_length], length);  
        Q_x.EnQue(x);      
    } 

    __aicore__ inline void Compute(int32_t progress, uint32_t length, float beta, float threshold) {
        LocalTensor<half> x = Q_x.DeQue<half>();
        LocalTensor<half> y = Q_y.AllocTensor<half>();
        
        //if (β * x > threshold):
        //    y = x
        //else:
        //    y = (1/β) * log(1 + exp(β * x))

        // tmp buffers (FP32)
        LocalTensor<float> tmpBX  = tmpbxBuffer.Get<float>();
        LocalTensor<uint8_t> selMask = selMaskBuffer.Get<uint8_t>();

        // ---------------------------------------
        // Cast x -> float
        // ---------------------------------------
        Cast(tmpBX, x, RoundMode::CAST_NONE, length);

        // ---------------------------------------
        // tmpBX = beta * x
        // ---------------------------------------
        Muls(tmpBX, tmpBX, beta, length);

        // ---------------------------------------
        // selMask = (beta * x > threshold)
        // ---------------------------------------

        CompareScalar(selMask, tmpBX, threshold, CMPMODE::GT, length);

        // ---------------------------------------
        // exp(beta * x)
        // ---------------------------------------
        Exp(tmpBX, tmpBX, length);

        // ---------------------------------------
        // 1 + exp(beta * x)
        // ---------------------------------------
        Adds(tmpBX, tmpBX, static_cast<float>(1), length);

        // ---------------------------------------
        // log(1 + exp(beta * x))
        // ---------------------------------------
        Ln(tmpBX, tmpBX, length);

        // ---------------------------------------
        // (1 / beta) * log(...)
        // ---------------------------------------
        Muls(tmpBX, tmpBX, static_cast<float>(1.0f / beta), length);

        // ---------------------------------------
        // Cast back to FP16
        // ---------------------------------------
        Cast(y, tmpBX, RoundMode::CAST_ROUND, length);

        // ---------------------------------------
        // Select
        // if beta*x > threshold: y = x
        // else:                 y = (1/beta)*log(1+exp(beta*x))
        // ---------------------------------------
        Select(
            y,
            selMask,
            x,      // src0: x
            y,      // src1: softplus result
            SELMODE::VSEL_TENSOR_TENSOR_MODE,
            length);


        Q_x.FreeTensor(x);
        Q_y.EnQue<half>(y);

    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<half> y = Q_y.DeQue<half>();
        DataCopy(yGM[progress * this->tiling_length], y, length);
        Q_y.FreeTensor(y);
    } 


private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<TPosition::VECCALC> tmpbxBuffer, selMaskBuffer;


    GlobalTensor<half> xGM, yGM;
    uint32_t core_length, tiling_length, tile_num;

};



class KernelSoftplusFP32 {
public:
    __aicore__ inline KernelSoftplusFP32() {}


    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t coreNum, uint32_t tiling_block_num, uint32_t block_length, uint32_t small_core_block_num, uint32_t big_core_num)
    {
        uint32_t total_length = totalLength;
        
        this->tiling_length = tiling_block_num * block_length;
        int64_t core_idx = GetBlockIdx();
        uint32_t start = core_idx <= big_core_num ? core_idx * (small_core_block_num + 1) : big_core_num * (small_core_block_num + 1) + (core_idx - big_core_num) * small_core_block_num;
        start = start * block_length; //偏移地址
        
        this->core_length = core_idx >= big_core_num ? small_core_block_num * block_length : (small_core_block_num + 1) * block_length;
        
        xGM.SetGlobalBuffer((__gm__ float*)x + start, this->core_length);
        yGM.SetGlobalBuffer((__gm__ float*)y + start, this->core_length);

        this->tile_num = this->core_length / this->tiling_length + (this->core_length % this->tiling_length > 0); //一共多少个tile
       
        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tiling_length * sizeof(float));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tiling_length * sizeof(float));
        pipe.InitBuffer(tmpbxBuffer, this->tiling_length * sizeof(float));
        pipe.InitBuffer(selMaskBuffer, this->tiling_length * sizeof(uint8_t));  
        

    }

    __aicore__ inline void Process(float beta, float threshold) { 
        for (int32_t i = 0; i < this->tile_num - 1; ++i) {       
            CopyIn(i, this->tiling_length);
            Compute(i, this->tiling_length, beta, threshold);  
            CopyOut(i, this->tiling_length);  
        }
        uint32_t length = this->core_length - this->tiling_length * (this->tile_num - 1);
        CopyIn(this->tile_num - 1, length);
        Compute(this->tile_num - 1, length, beta, threshold); 
        CopyOut(this->tile_num - 1, length);  
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<float> x = Q_x.AllocTensor<float>();    
        DataCopy(x, xGM[progress * this->tiling_length], length);  
        Q_x.EnQue(x);      
    } 

    __aicore__ inline void Compute(int32_t progress, uint32_t length, float beta, float threshold) {
        LocalTensor<float> x = Q_x.DeQue<float>();
        LocalTensor<float> y = Q_y.AllocTensor<float>();
        
        
        //if (β * x > threshold):
        //    y = x
        //else:
        //    y = (1/β) * log(1 + exp(β * x))

        // tmp buffers (FP32)
        LocalTensor<float> tmpBX  = tmpbxBuffer.Get<float>();
        LocalTensor<uint8_t> selMask = selMaskBuffer.Get<uint8_t>();


        // ---------------------------------------
        // tmpBX = beta * x
        // ---------------------------------------


        Muls(tmpBX, x, beta, length);

        // ---------------------------------------
        // selMask = (beta * x > threshold)
        // ---------------------------------------
        CompareScalar(selMask, tmpBX, threshold, CMPMODE::GT, length);
        // ---------------------------------------
        // exp(beta * x)
        // ---------------------------------------
        Exp(tmpBX, tmpBX, length);

        // ---------------------------------------
        // 1 + exp(beta * x)
        // ---------------------------------------
        Adds(tmpBX, tmpBX, static_cast<float>(1), length);

        // ---------------------------------------
        // log(1 + exp(beta * x))
        // ---------------------------------------
        Ln(tmpBX, tmpBX, length);

        // ---------------------------------------
        // (1 / beta) * log(...)
        // ---------------------------------------
        Muls(tmpBX, tmpBX, static_cast<float>(1.0f / beta), length);
        // Cast(y, tmpBX, RoundMode::CAST_ROUND, length);
        // ---------------------------------------
        // Select
        // if beta*x > threshold: y = x
        // else:                 y = (1/beta)*log(1+exp(beta*x))
        // ---------------------------------------
        Select(
            y,
            selMask,
            x,      // src0: x
            tmpBX,      // src1: softplus result
            SELMODE::VSEL_TENSOR_TENSOR_MODE,
            length);


        Q_x.FreeTensor(x);
        Q_y.EnQue<float>(y);

    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<float> y = Q_y.DeQue<float>();
        DataCopy(yGM[progress * this->tiling_length], y, length);
        Q_y.FreeTensor(y);
    } 


private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<TPosition::VECCALC> tmpbxBuffer, selMaskBuffer;



    GlobalTensor<float> xGM, yGM;
    uint32_t core_length, tiling_length, tile_num;
};


class KernelSoftplusBF16 {
public:
    __aicore__ inline KernelSoftplusBF16() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t coreNum, uint32_t tiling_block_num, uint32_t block_length, uint32_t small_core_block_num, uint32_t big_core_num)
    {
        uint32_t total_length = totalLength;
        
        this->tiling_length = tiling_block_num * block_length;
        int64_t core_idx = GetBlockIdx();
        uint32_t start = core_idx <= big_core_num ? core_idx * (small_core_block_num + 1) : big_core_num * (small_core_block_num + 1) + (core_idx - big_core_num) * small_core_block_num;
        start = start * block_length; //偏移地址
        
        this->core_length = core_idx >= big_core_num ? small_core_block_num * block_length : (small_core_block_num + 1) * block_length;
        
        xGM.SetGlobalBuffer((__gm__ bfloat16_t*)x + start, this->core_length);
        yGM.SetGlobalBuffer((__gm__ bfloat16_t*)y + start, this->core_length);

        this->tile_num = this->core_length / this->tiling_length + (this->core_length % this->tiling_length > 0); //一共多少个tile
       
        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tiling_length * sizeof(bfloat16_t));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tiling_length * sizeof(bfloat16_t));
        pipe.InitBuffer(tmpbxBuffer, this->tiling_length * sizeof(float));
        pipe.InitBuffer(selMaskBuffer, this->tiling_length * sizeof(uint8_t));  
        

    }

    __aicore__ inline void Process(float beta, float threshold) { 
        for (int32_t i = 0; i < this->tile_num - 1; ++i) {       
            CopyIn(i, this->tiling_length);
            Compute(i, this->tiling_length, beta, threshold);  
            CopyOut(i, this->tiling_length);  
        }
        uint32_t length = this->core_length - this->tiling_length * (this->tile_num - 1);
        CopyIn(this->tile_num - 1, length);
        Compute(this->tile_num - 1, length, beta, threshold); 
        CopyOut(this->tile_num - 1, length);  
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<bfloat16_t> x = Q_x.AllocTensor<bfloat16_t>();    
        DataCopy(x, xGM[progress * this->tiling_length], length);  
        Q_x.EnQue(x);      
    } 

    __aicore__ inline void Compute(int32_t progress, uint32_t length, float beta, float threshold) {
        LocalTensor<bfloat16_t> x = Q_x.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> y = Q_y.AllocTensor<bfloat16_t>();
        
        //if (β * x > threshold):
        //    y = x
        //else:
        //    y = (1/β) * log(1 + exp(β * x))

        // tmp buffers (FP32)
        LocalTensor<float> tmpBX  = tmpbxBuffer.Get<float>();
        LocalTensor<uint8_t> selMask = selMaskBuffer.Get<uint8_t>();
        

        // ---------------------------------------
        // Cast x -> float
        // ---------------------------------------
        Cast(tmpBX, x, RoundMode::CAST_NONE, length);

        // ---------------------------------------
        // tmpBX = beta * x
        // ---------------------------------------
        Muls(tmpBX, tmpBX, beta, length);

        // ---------------------------------------
        // selMask = (beta * x > threshold)
        // ---------------------------------------
        CompareScalar(selMask, tmpBX, threshold, CMPMODE::GT, length);

        // ---------------------------------------
        // exp(beta * x)
        // ---------------------------------------
        Exp(tmpBX, tmpBX, length);

        // ---------------------------------------
        // 1 + exp(beta * x)
        // ---------------------------------------
        Adds(tmpBX, tmpBX, static_cast<float>(1), length);

        // ---------------------------------------
        // log(1 + exp(beta * x))
        // ---------------------------------------
        Ln(tmpBX, tmpBX, length);

        // ---------------------------------------
        // (1 / beta) * log(...)
        // ---------------------------------------
        Muls(tmpBX, tmpBX, static_cast<float>(1.0f / beta), length);

        // ---------------------------------------
        // Cast back to FP16
        // ---------------------------------------
        Cast(y, tmpBX, RoundMode::CAST_ROUND, length);

        // ---------------------------------------
        // Select
        // if beta*x > threshold: y = x
        // else:                 y = (1/beta)*log(1+exp(beta*x))
        // ---------------------------------------
        Select(
            y,
            selMask,
            x,      // src0: x
            y,      // src1: softplus result
            SELMODE::VSEL_TENSOR_TENSOR_MODE,
            length);


        Q_x.FreeTensor(x);
        Q_y.EnQue<bfloat16_t>(y);

    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<bfloat16_t> y = Q_y.DeQue<bfloat16_t>();
        DataCopy(yGM[progress * this->tiling_length], y, length);
        Q_y.FreeTensor(y);
    } 


private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<TPosition::VECCALC> tmpbxBuffer, selMaskBuffer;



    GlobalTensor<bfloat16_t> xGM, yGM;
    uint32_t core_length, tiling_length, tile_num;
};



////

class KernelSoftplusFP16b {
public:
    __aicore__ inline KernelSoftplusFP16b() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t coreNum, uint32_t tiling_block_num, uint32_t block_length, uint32_t small_core_block_num, uint32_t big_core_num)
    {
        uint32_t total_length = totalLength;
        
        this->tiling_length = tiling_block_num * block_length;
        int64_t core_idx = GetBlockIdx();
        uint32_t start = core_idx <= big_core_num ? core_idx * (small_core_block_num + 1) : big_core_num * (small_core_block_num + 1) + (core_idx - big_core_num) * small_core_block_num;
        start = start * block_length; //偏移地址
        
        this->core_length = core_idx >= big_core_num ? small_core_block_num * block_length : (small_core_block_num + 1) * block_length;
        
        xGM.SetGlobalBuffer((__gm__ half*)x + start, this->core_length);
        yGM.SetGlobalBuffer((__gm__ half*)y + start, this->core_length);

        this->tile_num = this->core_length / this->tiling_length + (this->core_length % this->tiling_length > 0); //一共多少个tile
       
        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tiling_length * sizeof(half));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tiling_length * sizeof(half));
        pipe.InitBuffer(tmpbxBuffer, this->tiling_length * sizeof(float));
        pipe.InitBuffer(selMaskBuffer, this->tiling_length * sizeof(uint8_t));  
        

    }

    __aicore__ inline void Process(float beta, float threshold) { 
        for (int32_t i = 0; i < this->tile_num - 1; ++i) {       
            CopyIn(i, this->tiling_length);
            Compute(i, this->tiling_length, beta, threshold);  
            CopyOut(i, this->tiling_length);  
        }
        uint32_t length = this->core_length - this->tiling_length * (this->tile_num - 1);
        CopyIn(this->tile_num - 1, length);
        Compute(this->tile_num - 1, length, beta, threshold); 
        CopyOut(this->tile_num - 1, length);  
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<half> x = Q_x.AllocTensor<half>();    
        DataCopy(x, xGM[progress * this->tiling_length], length);  
        Q_x.EnQue(x);      
    } 

    __aicore__ inline void Compute(int32_t progress, uint32_t length, float beta, float threshold) {
        LocalTensor<half> x = Q_x.DeQue<half>();
        LocalTensor<half> y = Q_y.AllocTensor<half>();
        
        //if (β * x > threshold):
        //    y = x
        //else:
        //    y = (1/β) * log(1 + exp(β * x))

        // tmp buffers (FP32)
        LocalTensor<float> tmpBX  = tmpbxBuffer.Get<float>();
        LocalTensor<uint8_t> selMask = selMaskBuffer.Get<uint8_t>();

        // ---------------------------------------
        // Cast x -> float
        // ---------------------------------------
        Cast(tmpBX, x, RoundMode::CAST_NONE, length);

        // ---------------------------------------
        // tmpBX = beta * x
        // ---------------------------------------
        // Muls(tmpBX, tmpBX, beta, length);

        // ---------------------------------------
        // selMask = (beta * x > threshold)
        // ---------------------------------------

        CompareScalar(selMask, tmpBX, threshold, CMPMODE::GT, length);

        // ---------------------------------------
        // exp(beta * x)
        // ---------------------------------------
        Exp(tmpBX, tmpBX, length);

        // ---------------------------------------
        // 1 + exp(beta * x)
        // ---------------------------------------
        Adds(tmpBX, tmpBX, static_cast<float>(1), length);

        // ---------------------------------------
        // log(1 + exp(beta * x))
        // ---------------------------------------
        Ln(tmpBX, tmpBX, length);



        // ---------------------------------------
        // Cast back to FP16
        // ---------------------------------------
        Cast(y, tmpBX, RoundMode::CAST_ROUND, length);

        // ---------------------------------------
        // Select
        // if beta*x > threshold: y = x
        // else:                 y = (1/beta)*log(1+exp(beta*x))
        // ---------------------------------------
        Select(
            y,
            selMask,
            x,      // src0: x
            y,      // src1: softplus result
            SELMODE::VSEL_TENSOR_TENSOR_MODE,
            length);


        Q_x.FreeTensor(x);
        Q_y.EnQue<half>(y);

    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<half> y = Q_y.DeQue<half>();
        DataCopy(yGM[progress * this->tiling_length], y, length);
        Q_y.FreeTensor(y);
    } 


private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<TPosition::VECCALC> tmpbxBuffer, selMaskBuffer;


    GlobalTensor<half> xGM, yGM;
    uint32_t core_length, tiling_length, tile_num;

};



class KernelSoftplusFP32b {
public:
    __aicore__ inline KernelSoftplusFP32b() {}


    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t coreNum, uint32_t tiling_block_num, uint32_t block_length, uint32_t small_core_block_num, uint32_t big_core_num)
    {
        uint32_t total_length = totalLength;
        
        this->tiling_length = tiling_block_num * block_length;
        int64_t core_idx = GetBlockIdx();
        uint32_t start = core_idx <= big_core_num ? core_idx * (small_core_block_num + 1) : big_core_num * (small_core_block_num + 1) + (core_idx - big_core_num) * small_core_block_num;
        start = start * block_length; //偏移地址
        
        this->core_length = core_idx >= big_core_num ? small_core_block_num * block_length : (small_core_block_num + 1) * block_length;
        
        xGM.SetGlobalBuffer((__gm__ float*)x + start, this->core_length);
        yGM.SetGlobalBuffer((__gm__ float*)y + start, this->core_length);

        this->tile_num = this->core_length / this->tiling_length + (this->core_length % this->tiling_length > 0); //一共多少个tile
       
        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tiling_length * sizeof(float));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tiling_length * sizeof(float));
        pipe.InitBuffer(tmpbxBuffer, this->tiling_length * sizeof(float));
        pipe.InitBuffer(selMaskBuffer, this->tiling_length * sizeof(uint8_t));  
        

    }

    __aicore__ inline void Process(float beta, float threshold) { 
        for (int32_t i = 0; i < this->tile_num - 1; ++i) {       
            CopyIn(i, this->tiling_length);
            Compute(i, this->tiling_length, beta, threshold);  
            CopyOut(i, this->tiling_length);  
        }
        uint32_t length = this->core_length - this->tiling_length * (this->tile_num - 1);
        CopyIn(this->tile_num - 1, length);
        Compute(this->tile_num - 1, length, beta, threshold); 
        CopyOut(this->tile_num - 1, length);  
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<float> x = Q_x.AllocTensor<float>();    
        DataCopy(x, xGM[progress * this->tiling_length], length);  
        Q_x.EnQue(x);      
    } 

    __aicore__ inline void Compute(int32_t progress, uint32_t length, float beta, float threshold) {
        LocalTensor<float> x = Q_x.DeQue<float>();
        LocalTensor<float> y = Q_y.AllocTensor<float>();
        
        
        //if (β * x > threshold):
        //    y = x
        //else:
        //    y = (1/β) * log(1 + exp(β * x))

        // tmp buffers (FP32)
        LocalTensor<float> tmpBX  = tmpbxBuffer.Get<float>();
        LocalTensor<uint8_t> selMask = selMaskBuffer.Get<uint8_t>();


        // ---------------------------------------
        // tmpBX = beta * x
        // ---------------------------------------


        //Muls(tmpBX, x, beta, length);

        // ---------------------------------------
        // selMask = (beta * x > threshold)
        // ---------------------------------------
        CompareScalar(selMask, x, threshold, CMPMODE::GT, length);
        // ---------------------------------------
        // exp(beta * x)
        // ---------------------------------------
        Exp(tmpBX, x, length);

        // ---------------------------------------
        // 1 + exp(beta * x)
        // ---------------------------------------
        Adds(tmpBX, tmpBX, static_cast<float>(1), length);

        // ---------------------------------------
        // log(1 + exp(beta * x))
        // ---------------------------------------
        Ln(tmpBX, tmpBX, length);

        // ---------------------------------------
        // (1 / beta) * log(...)
        // ---------------------------------------
        // Muls(tmpBX, tmpBX, static_cast<float>(1.0f / beta), length);
        // Cast(y, tmpBX, RoundMode::CAST_ROUND, length);
        // ---------------------------------------
        // Select
        // if beta*x > threshold: y = x
        // else:                 y = (1/beta)*log(1+exp(beta*x))
        // ---------------------------------------
        Select(
            y,
            selMask,
            x,      // src0: x
            tmpBX,      // src1: softplus result
            SELMODE::VSEL_TENSOR_TENSOR_MODE,
            length);


        Q_x.FreeTensor(x);
        Q_y.EnQue<float>(y);

    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<float> y = Q_y.DeQue<float>();
        DataCopy(yGM[progress * this->tiling_length], y, length);
        Q_y.FreeTensor(y);
    } 


private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<TPosition::VECCALC> tmpbxBuffer, selMaskBuffer;



    GlobalTensor<float> xGM, yGM;
    uint32_t core_length, tiling_length, tile_num;
};


class KernelSoftplusBF16b {
public:
    __aicore__ inline KernelSoftplusBF16b() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t coreNum, uint32_t tiling_block_num, uint32_t block_length, uint32_t small_core_block_num, uint32_t big_core_num)
    {
        uint32_t total_length = totalLength;
        
        this->tiling_length = tiling_block_num * block_length;
        int64_t core_idx = GetBlockIdx();
        uint32_t start = core_idx <= big_core_num ? core_idx * (small_core_block_num + 1) : big_core_num * (small_core_block_num + 1) + (core_idx - big_core_num) * small_core_block_num;
        start = start * block_length; //偏移地址
        
        this->core_length = core_idx >= big_core_num ? small_core_block_num * block_length : (small_core_block_num + 1) * block_length;
        
        xGM.SetGlobalBuffer((__gm__ bfloat16_t*)x + start, this->core_length);
        yGM.SetGlobalBuffer((__gm__ bfloat16_t*)y + start, this->core_length);

        this->tile_num = this->core_length / this->tiling_length + (this->core_length % this->tiling_length > 0); //一共多少个tile
       
        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tiling_length * sizeof(bfloat16_t));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tiling_length * sizeof(bfloat16_t));
        pipe.InitBuffer(tmpbxBuffer, this->tiling_length * sizeof(float));
        pipe.InitBuffer(selMaskBuffer, this->tiling_length * sizeof(uint8_t));  
        

    }

    __aicore__ inline void Process(float beta, float threshold) { 
        for (int32_t i = 0; i < this->tile_num - 1; ++i) {       
            CopyIn(i, this->tiling_length);
            Compute(i, this->tiling_length, beta, threshold);  
            CopyOut(i, this->tiling_length);  
        }
        uint32_t length = this->core_length - this->tiling_length * (this->tile_num - 1);
        CopyIn(this->tile_num - 1, length);
        Compute(this->tile_num - 1, length, beta, threshold); 
        CopyOut(this->tile_num - 1, length);  
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<bfloat16_t> x = Q_x.AllocTensor<bfloat16_t>();    
        DataCopy(x, xGM[progress * this->tiling_length], length);  
        Q_x.EnQue(x);      
    } 

    __aicore__ inline void Compute(int32_t progress, uint32_t length, float beta, float threshold) {
        LocalTensor<bfloat16_t> x = Q_x.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> y = Q_y.AllocTensor<bfloat16_t>();
        
        //if (β * x > threshold):
        //    y = x
        //else:
        //    y = (1/β) * log(1 + exp(β * x))

        // tmp buffers (FP32)
        LocalTensor<float> tmpBX  = tmpbxBuffer.Get<float>();
        LocalTensor<uint8_t> selMask = selMaskBuffer.Get<uint8_t>();
        

        // ---------------------------------------
        // Cast x -> float
        // ---------------------------------------
        Cast(tmpBX, x, RoundMode::CAST_NONE, length);



        // ---------------------------------------
        // selMask = (beta * x > threshold)
        // ---------------------------------------
        CompareScalar(selMask, tmpBX, threshold, CMPMODE::GT, length);

        // ---------------------------------------
        // exp(beta * x)
        // ---------------------------------------
        Exp(tmpBX, tmpBX, length);

        // ---------------------------------------
        // 1 + exp(beta * x)
        // ---------------------------------------
        Adds(tmpBX, tmpBX, static_cast<float>(1), length);

        // ---------------------------------------
        // log(1 + exp(beta * x))
        // ---------------------------------------
        Ln(tmpBX, tmpBX, length);



        // ---------------------------------------
        // Cast back to FP16
        // ---------------------------------------
        Cast(y, tmpBX, RoundMode::CAST_ROUND, length);

        // ---------------------------------------
        // Select
        // if beta*x > threshold: y = x
        // else:                 y = (1/beta)*log(1+exp(beta*x))
        // ---------------------------------------
        Select(
            y,
            selMask,
            x,      // src0: x
            y,      // src1: softplus result
            SELMODE::VSEL_TENSOR_TENSOR_MODE,
            length);


        Q_x.FreeTensor(x);
        Q_y.EnQue<bfloat16_t>(y);

    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<bfloat16_t> y = Q_y.DeQue<bfloat16_t>();
        DataCopy(yGM[progress * this->tiling_length], y, length);
        Q_y.FreeTensor(y);
    } 


private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<TPosition::VECCALC> tmpbxBuffer, selMaskBuffer;



    GlobalTensor<bfloat16_t> xGM, yGM;
    uint32_t core_length, tiling_length, tile_num;
};


////


extern "C" __global__ __aicore__ void softplus(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    uint32_t totalLength = tiling_data.totalLength;
    uint32_t coreNum = tiling_data.coreNum;
    uint32_t tiling_block_num = tiling_data.tiling_block_num;

    uint32_t block_length = tiling_data.block_length;
    uint32_t small_core_block_num = tiling_data.small_core_block_num;
    uint32_t big_core_num = tiling_data.big_core_num;

    if (TILING_KEY_IS(0)) {
        KernelSoftplusFP16 op;
        op.Init(x, y, totalLength, coreNum, tiling_block_num, block_length, small_core_block_num, big_core_num);
        op.Process(tiling_data.beta, tiling_data.threshold);
    } else if (TILING_KEY_IS(1)) {
        KernelSoftplusFP32 op;
        op.Init(x, y, totalLength, coreNum, tiling_block_num, block_length, small_core_block_num, big_core_num);
        op.Process(tiling_data.beta, tiling_data.threshold);
    } else if (TILING_KEY_IS(2)) {
        KernelSoftplusBF16 op;
        op.Init(x, y, totalLength, coreNum, tiling_block_num, block_length, small_core_block_num, big_core_num);
        op.Process(tiling_data.beta, tiling_data.threshold);
    } else if (TILING_KEY_IS(10)) {
        KernelSoftplusFP16b op;
        op.Init(x, y, totalLength, coreNum, tiling_block_num, block_length, small_core_block_num, big_core_num);
        op.Process(tiling_data.beta, tiling_data.threshold);
    } else if (TILING_KEY_IS(11)) {
        KernelSoftplusFP32b op;
        op.Init(x, y, totalLength, coreNum, tiling_block_num, block_length, small_core_block_num, big_core_num);
        op.Process(tiling_data.beta, tiling_data.threshold);
    } else if (TILING_KEY_IS(12)) {
        KernelSoftplusBF16b op;
        op.Init(x, y, totalLength, coreNum, tiling_block_num, block_length, small_core_block_num, big_core_num);
        op.Process(tiling_data.beta, tiling_data.threshold);
    }
}