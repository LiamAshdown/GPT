#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

class Vocab {
public:
    static Vocab buildBaseVocab() {
        Vocab vocab;
        for (int i = 0; i < 256; i++)
            vocab.add({ static_cast<uint8_t>(i) });
        return vocab;
    }

    int add(const std::vector<uint8_t>& bytes) {
        const int id = static_cast<int>(_idToBytes.size());

        _idToBytes.push_back(bytes);

        const std::string key(bytes.begin(), bytes.end());
        _bytesToId[key] = id;

        return id;
    }

    int size() const {
        return static_cast<int>(_idToBytes.size());
    }

    std::vector<uint8_t> get(const int id) const {
        return _idToBytes[id];
    }

private:
    std::vector<std::vector<uint8_t>> _idToBytes;
    std::unordered_map<std::string, int> _bytesToId;
};
