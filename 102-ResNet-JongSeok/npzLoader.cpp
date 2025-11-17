#include "npzLoader.h"
#include <iostream>
#include <sstream>
#include <cstdint>
#include <algorithm>
#include <zlib.h>

// Minimal ZIP parser for NPZ files
struct ZipLocalFileHeader
{
    uint32_t signature;
    uint16_t version;
    uint16_t flags;
    uint16_t compression;
    uint16_t modTime;
    uint16_t modDate;
    uint32_t crc32;
    uint32_t compressedSize;
    uint32_t uncompressedSize;
    uint16_t filenameLength;
    uint16_t extraLength;
};

// Decompress stored (uncompressed) data
std::vector<uint8_t> decompressStored(const uint8_t* data, size_t size)
{
    return std::vector<uint8_t>(data, data + size);
}

// Decompress deflate compressed data using zlib
std::vector<uint8_t> decompressDeflate(const uint8_t* data, size_t compressedSize, size_t uncompressedSize)
{
    std::vector<uint8_t> result(uncompressedSize);

    z_stream strm;
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = compressedSize;
    strm.next_in = const_cast<uint8_t*>(data);
    strm.avail_out = uncompressedSize;
    strm.next_out = result.data();

    // Use raw deflate (negative window bits)
    if (inflateInit2(&strm, -MAX_WBITS) != Z_OK)
        throw std::runtime_error("inflateInit2 failed");

    int ret = inflate(&strm, Z_FINISH);
    inflateEnd(&strm);

    if (ret != Z_STREAM_END)
        throw std::runtime_error("inflate failed");

    return result;
}

NpzLoader::NpzLoader(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Failed to open NPZ file: " + filename);

    // Read entire file
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> fileData(fileSize);
    file.read((char*)fileData.data(), fileSize);
    file.close();

    // Parse ZIP entries
    size_t offset = 0;
    while (offset < fileSize)
    {
        // Check for local file header signature (0x04034b50)
        if (offset + 30 > fileSize) break;

        uint32_t signature = *reinterpret_cast<const uint32_t*>(&fileData[offset]);

        if (signature == 0x04034b50)  // Local file header
        {
            ZipLocalFileHeader header;
            size_t pos = offset;

            header.signature = *reinterpret_cast<const uint32_t*>(&fileData[pos]); pos += 4;
            header.version = *reinterpret_cast<const uint16_t*>(&fileData[pos]); pos += 2;
            header.flags = *reinterpret_cast<const uint16_t*>(&fileData[pos]); pos += 2;
            header.compression = *reinterpret_cast<const uint16_t*>(&fileData[pos]); pos += 2;
            header.modTime = *reinterpret_cast<const uint16_t*>(&fileData[pos]); pos += 2;
            header.modDate = *reinterpret_cast<const uint16_t*>(&fileData[pos]); pos += 2;
            header.crc32 = *reinterpret_cast<const uint32_t*>(&fileData[pos]); pos += 4;
            header.compressedSize = *reinterpret_cast<const uint32_t*>(&fileData[pos]); pos += 4;
            header.uncompressedSize = *reinterpret_cast<const uint32_t*>(&fileData[pos]); pos += 4;
            header.filenameLength = *reinterpret_cast<const uint16_t*>(&fileData[pos]); pos += 2;
            header.extraLength = *reinterpret_cast<const uint16_t*>(&fileData[pos]); pos += 2;

            offset = pos;

            // Read filename
            std::string filename(reinterpret_cast<const char*>(&fileData[offset]), header.filenameLength);
            offset += header.filenameLength;
            offset += header.extraLength;

            // Read file data
            std::vector<uint8_t> fileContent;

            if (header.compression == 0)  // Stored (no compression)
            {
                fileContent = decompressStored(&fileData[offset], header.uncompressedSize);
            }
            else if (header.compression == 8)  // Deflate
            {
                fileContent = decompressDeflate(&fileData[offset], header.compressedSize, header.uncompressedSize);
            }
            else
            {
                throw std::runtime_error("Unsupported compression method: " + std::to_string(header.compression));
            }

            offset += header.compressedSize;

            // Parse NPY data
            if (filename.find(".npy") != std::string::npos)
            {
                std::string key = filename.substr(0, filename.find(".npy"));
                arrays[key] = parseNpy(fileContent);
                std::cout << "Loaded array '" << key << "': ";
                for (size_t i = 0; i < arrays[key].shape.size(); ++i)
                {
                    if (i > 0) std::cout << " × ";
                    std::cout << arrays[key].shape[i];
                }
                std::cout << std::endl;
            }
        }
        else if (signature == 0x02014b50)  // Central directory header
        {
            break;  // We've read all local files
        }
        else
        {
            offset++;  // Skip unknown data
        }
    }

    std::cout << "Total arrays loaded: " << arrays.size() << std::endl;
}

NpyArray NpzLoader::parseNpy(const std::vector<uint8_t>& data)
{
    NpyArray array;

    // NPY format:
    // Magic: \x93NUMPY
    // Version: major, minor (uint8 x2)
    // Header length: uint16 (v1.0) or uint32 (v2.0)
    // Header: Python dict as string
    // Data: raw binary

    if (data.size() < 10)
        throw std::runtime_error("NPY file too small");

    // Check magic
    if (data[0] != 0x93 || data[1] != 'N' || data[2] != 'U' ||
        data[3] != 'M' || data[4] != 'P' || data[5] != 'Y')
        throw std::runtime_error("Invalid NPY magic number");

    uint8_t major = data[6];
    uint8_t minor = data[7];

    size_t headerLen;
    size_t dataOffset;

    if (major == 1)
    {
        headerLen = *reinterpret_cast<const uint16_t*>(&data[8]);
        dataOffset = 10 + headerLen;
    }
    else if (major == 2)
    {
        headerLen = *reinterpret_cast<const uint32_t*>(&data[8]);
        dataOffset = 12 + headerLen;
    }
    else
    {
        throw std::runtime_error("Unsupported NPY version");
    }

    // Parse header (Python dict)
    std::string header(reinterpret_cast<const char*>(&data[10]), headerLen);

    // Extract shape
    size_t shapeStart = header.find("'shape'");
    if (shapeStart == std::string::npos)
        shapeStart = header.find("\"shape\"");

    if (shapeStart != std::string::npos)
    {
        size_t tupleStart = header.find('(', shapeStart);
        size_t tupleEnd = header.find(')', tupleStart);

        std::string shapeStr = header.substr(tupleStart + 1, tupleEnd - tupleStart - 1);
        std::istringstream ss(shapeStr);
        std::string token;

        while (std::getline(ss, token, ','))
        {
            token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
            if (!token.empty())
                array.shape.push_back(std::stoul(token));
        }
    }

    // Calculate number of elements
    size_t numElements = 1;
    for (auto dim : array.shape)
        numElements *= dim;

    // Read data (assuming float32)
    size_t dataSize = numElements * sizeof(float);
    if (dataOffset + dataSize > data.size())
        throw std::runtime_error("NPY data size mismatch");

    array.data.resize(numElements);
    memcpy(array.data.data(), &data[dataOffset], dataSize);

    return array;
}
