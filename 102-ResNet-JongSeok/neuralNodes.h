#ifndef NEURAL_NODES_H
#define NEURAL_NODES_H


#include "neuralNet.h"
#include <string>

class NpzLoader;


class ConvolutionNode : public Node
{
    uint32_t C, F, K, S;   // C: input channels, F: output channels, K: kernel width, S: stride

    ComputePipeline im2col;
    ComputePipeline gemm;
    DescriptorSet im2colDescSet;
    DescriptorSet gemmDescSet;
    uint32_t gemmTileSize;

public:
    ConvolutionNode(uint32_t inChannels, uint32_t outChannels, uint32_t kernelWidth, uint32_t stride = 1);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class ReluNode : public Node
{
    ComputePipeline relu;
    DescriptorSet reluDescSet;

public:
    ReluNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class MaxPoolingNode : public Node
{
    const bool discardTail = true; // If true, discard the tail elements that don't fit into the pooling window
    uint32_t P;

    ComputePipeline maxpool;
    DescriptorSet maxpoolDescSet;

public:
    MaxPoolingNode(uint32_t poolSize);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class FlattenNode : public Node
{
    ComputePipeline copy;
    DescriptorSet copyDescSet;

public:
    FlattenNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class FullyConnectedNode : public Node
{
    uint32_t I, O; // I: input size, O: output size
    ComputePipeline gemm;
    DescriptorSet gemmDescSet;
    uint32_t gemmTileSize;

    ComputePipeline setZero;
    DescriptorSet setZeroDescSet;

public:
    FullyConnectedNode(uint32_t inDim, uint32_t outDim);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class BatchNormalizationNode : public Node
{
    uint32_t C; // number of channels
    ComputePipeline batchnorm;
    DescriptorSet batchnormDescSet;

public:
    BatchNormalizationNode(uint32_t numChannels);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class GlobalAvgPoolNode : public Node
{
    ComputePipeline globalAvgPool;
    DescriptorSet globalAvgPoolDescSet;

public:
    GlobalAvgPoolNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class AddNode : public Node
{
    ComputePipeline add;
    DescriptorSet addDescSet;

public:
    AddNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class BasicBlock : public NodeGroup
{
    ConvolutionNode conv1;
    BatchNormalizationNode bn1;
    ReluNode relu1;
    ConvolutionNode conv2;
    BatchNormalizationNode bn2;
    AddNode add;
    ReluNode relu2;

    ConvolutionNode* downsampleConv;
    BatchNormalizationNode* downsampleBN;

public:
    bool hasDownsample;

    BasicBlock(uint32_t inChannels, uint32_t outChannels, uint32_t stride = 1);
    ~BasicBlock();
    void buildGraph();
    void loadWeights(const NpzLoader& npz, const std::string& prefix);
};


extern Device netGlobalDevice; // Global device for neural network operations

void loadConvolutionWeights(ConvolutionNode& node, const NpzLoader& npz, const std::string& baseName);
void loadBatchNormalizationWeights(BatchNormalizationNode& node, const NpzLoader& npz, const std::string& baseName);


#endif // NEURAL_NODES_H
