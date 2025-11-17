#ifndef NPZ_LOADER_H
#define NPZ_LOADER_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <cstring>
#include <cstdint>
#include <stdexcept>


struct NpyArray
{
    std::vector<float> data;
    std::vector<size_t> shape;

    size_t size() const
    {
        size_t s = 1;
        for (auto dim : shape) s *= dim;
        return s;
    }
};

class NpzLoader
{
    std::map<std::string, NpyArray> arrays;

public:
    NpzLoader(const std::string& filename);

    const NpyArray& operator[](const std::string& key) const
    {
        auto it = arrays.find(key);
        if (it == arrays.end())
            throw std::runtime_error("Key not found in NPZ: " + key);
        return it->second;
    }

    bool hasKey(const std::string& key) const
    {
        return arrays.find(key) != arrays.end();
    }

    const std::map<std::string, NpyArray>& getArrays() const
    {
        return arrays;
    }

private:
    NpyArray parseNpy(const std::vector<uint8_t>& data);
};

#endif // NPZ_LOADER_H
