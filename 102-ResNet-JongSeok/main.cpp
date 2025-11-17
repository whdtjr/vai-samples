#include "neuralNodes.h"
#include "vulkanApp.h"
#include "npzLoader.h"
#include <stb/stb_image.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

// Helper function to read and preprocess image
std::vector<float> readAndPreprocessImage(const char* filename, uint32_t targetWidth, uint32_t targetHeight)
{
    int w, h, c;
    uint8_t* image = stbi_load(filename, &w, &h, &c, 3);  // Force 3 channels (RGB)

    if (!image)
    {
        std::cerr << "Failed to load image: " << stbi_failure_reason() << std::endl;
        throw std::runtime_error("Image loading failed");
    }

    std::cout << "Loaded image: " << w << "x" << h << "x" << c << std::endl;

    // Simple resize (nearest neighbor) if needed
    std::vector<float> processed(targetWidth * targetHeight * 3);

    // ImageNet normalization parameters
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};

    for (uint32_t h_out = 0; h_out < targetHeight; ++h_out)
    {
        for (uint32_t w_out = 0; w_out < targetWidth; ++w_out)
        {
            // Simple nearest neighbor sampling
            int h_in = (h_out * h) / targetHeight;
            int w_in = (w_out * w) / targetWidth;

            for (int ch = 0; ch < 3; ++ch)
            {
                uint8_t pixel = image[(h_in * w + w_in) * 3 + ch];
                float normalized = (pixel / 255.0f - mean[ch]) / std[ch];
                processed[(h_out * targetWidth + w_out) * 3 + ch] = normalized;
            }
        }
    }

    stbi_image_free(image);
    std::cout << "Image preprocessed to " << targetWidth << "x" << targetHeight << std::endl;

    return processed;
}

// Helper function to create a layer with multiple BasicBlocks
std::vector<BasicBlock*> makeLayer(uint32_t inChannels, uint32_t outChannels, uint32_t numBlocks, uint32_t stride = 1)
{
    std::vector<BasicBlock*> blocks;

    // First block may have stride > 1 and channel change
    blocks.push_back(new BasicBlock(inChannels, outChannels, stride));

    // Remaining blocks have stride=1 and same channels
    for (uint32_t i = 1; i < numBlocks; ++i)
    {
        blocks.push_back(new BasicBlock(outChannels, outChannels, 1));
    }

    return blocks;
}

int main()
{
    // Initialize GLFW (required for Vulkan)
    glfwInit();

    // Create neural network using the global device
    NeuralNet net(netGlobalDevice, 1, 1);

    // Create nodes for ResNet34
    // Initial layers
    ConvolutionNode conv1(3, 64, 7, 2);        // 3x224x224 -> 64x112x112
    BatchNormalizationNode bn1(64);
    ReluNode relu;
    MaxPoolingNode maxpool(2);                  // 64x112x112 -> 64x56x56

    // ResNet layers
    auto layer1 = makeLayer(64, 64, 3, 1);      // 64x56x56 -> 64x56x56
    auto layer2 = makeLayer(64, 128, 4, 2);     // 64x56x56 -> 128x28x28
    auto layer3 = makeLayer(128, 256, 6, 2);    // 128x28x28 -> 256x14x14
    auto layer4 = makeLayer(256, 512, 3, 2);    // 256x14x14 -> 512x7x7

    // Final layers
    GlobalAvgPoolNode avgpool;                  // 512x7x7 -> 512
    FlattenNode flatten;
    FullyConnectedNode fc(512, 1000);          // 512 -> 1000 classes

    // Build graph connections
    // input -> conv1 -> bn1 -> relu -> maxpool
    net.input().slot("out0") - conv1.slot("in0");
    conv1.slot("out0") - bn1.slot("in0");
    bn1.slot("out0") - relu.slot("in0");
    relu.slot("out0") - maxpool.slot("in0");

    // Connect layer1 blocks
    NodeGroup* prevGroup = nullptr;
    Node* prevNode = &maxpool;
    std::string prevSlot = "out0";

    for (auto* block : layer1)
    {
        if (prevGroup)
        {
            prevGroup->slot(prevSlot) - block->slot("in0");
            prevGroup->slot(prevSlot) - block->slot("in0_skip");
        }
        else
        {
            prevNode->slot(prevSlot) - block->slot("in0");
            prevNode->slot(prevSlot) - block->slot("in0_skip");
        }
        prevGroup = block;
        prevNode = nullptr;
        prevSlot = "out0";
    }

    // Connect layer2 blocks
    for (auto* block : layer2)
    {
        prevGroup->slot(prevSlot) - block->slot("in0");
        prevGroup->slot(prevSlot) - block->slot("in0_skip");
        prevGroup = block;
        prevSlot = "out0";
    }

    // Connect layer3 blocks
    for (auto* block : layer3)
    {
        prevGroup->slot(prevSlot) - block->slot("in0");
        prevGroup->slot(prevSlot) - block->slot("in0_skip");
        prevGroup = block;
        prevSlot = "out0";
    }

    // Connect layer4 blocks
    for (auto* block : layer4)
    {
        prevGroup->slot(prevSlot) - block->slot("in0");
        prevGroup->slot(prevSlot) - block->slot("in0_skip");
        prevGroup = block;
        prevSlot = "out0";
    }

    // Connect final layers
    prevGroup->slot(prevSlot) - avgpool.slot("in0");
    avgpool.slot("out0") - flatten.slot("in0");
    flatten.slot("out0") - fc.slot("in0");
    fc.slot("out0") - net.output().slot("in0");

    std::cout << "ResNet34 graph structure created successfully!" << std::endl;

    // Load shaders
    void loadShaders();
    loadShaders();

    // Load and preprocess image
    std::cout << "\n=== Loading Image ===" << std::endl;
    auto imageData = readAndPreprocessImage(PROJECT_CURRENT_DIR"/dog.jpg", 224, 224);
    Tensor inputTensor = Tensor(224, 224, 3).set(imageData);

    std::cout << "\n=== Loading Weights ===" << std::endl;
    // Load NPZ file directly
    NpzLoader npz(PROJECT_CURRENT_DIR"/resnet34_weights.npz");

    loadConvolutionWeights(conv1, npz, "conv1");
    loadBatchNormalizationWeights(bn1, npz, "bn1");

    auto loadLayer = [&](const std::vector<BasicBlock*>& blocks, const std::string& baseName) {
        for (size_t i = 0; i < blocks.size(); ++i)
        {
            blocks[i]->loadWeights(npz, baseName + "." + std::to_string(i));
        }
    };

    loadLayer(layer1, "layer1");
    loadLayer(layer2, "layer2");
    loadLayer(layer3, "layer3");
    loadLayer(layer4, "layer4");

    auto toVector = [](const NpyArray& arr) -> std::vector<float> {
        return arr.data;
    };

    // Load FC layer weights
    if (npz.hasKey("fc.weight"))
    {
        const auto& w = npz["fc.weight"];
        // FC: (1000, 512) -> reshape to (512, 1000)
        fc["weight"] = Tensor(static_cast<uint32_t>(w.shape[1]),
                              static_cast<uint32_t>(w.shape[0]))
                         .set(toVector(w));
    }
    if (npz.hasKey("fc.bias"))
        fc["bias"] = Tensor(static_cast<uint32_t>(npz["fc.bias"].shape[0]))
                       .set(toVector(npz["fc.bias"]));

    std::cout << "Weights loaded successfully!" << std::endl;

    std::cout << "\n=== Running Inference ===" << std::endl;
    auto outputs = net(inputTensor);
    Tensor& result = outputs[0];

    std::cout << "\n=== Copying Results ===" << std::endl;
    // Copy results to host
    vk::Buffer outBuffer = netGlobalDevice.createBuffer({
        1000 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, result.buffer())
        .end()
        .submit()
        .wait();

    float* predictions = (float*)outBuffer.map();

    // Find top-5 predictions
    std::vector<std::pair<int, float>> scores;
    for (int i = 0; i < 1000; ++i)
    {
        scores.push_back({i, predictions[i]});
    }

    std::sort(scores.begin(), scores.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    std::cout << "\n=== Top-5 Predictions ===" << std::endl;
    for (int i = 0; i < 5; ++i)
    {
        std::cout << "Class " << scores[i].first << ": " << scores[i].second << std::endl;
    }

    // Cleanup
    for (auto* block : layer1) delete block;
    for (auto* block : layer2) delete block;
    for (auto* block : layer3) delete block;
    for (auto* block : layer4) delete block;

    std::cout << "\nResNet34 inference completed!" << std::endl;

    glfwTerminate();
    return 0;
}
