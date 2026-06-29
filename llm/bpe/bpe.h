#pragma once


#include <algorithm>
#include <map>
#include <utility>
#include <fstream>

#include "vocab.h"
#include "tokenizer.h"

using Pair = std::pair<int, int>;
using PairCounts = std::map<Pair, int>;

struct Merge {
    Pair pair;
    int newId;
};

class Bpe {
public:
    explicit Bpe(Vocab vocab) : _vocab(std::move(vocab)), _merges({}) {}

public:
    void train(const Tokenizer& tokenizer, const int targetVocabSize) {
        std::vector<std::vector<int>> chunks = tokenizer.tokenize();

        while (_vocab.size() < targetVocabSize) {
            PairCounts counts = countPairs(chunks);

            if (counts.empty()) {
                break;
            }

            Pair best = mostFrequentPair(counts);

            std::vector<uint8_t> leftBytes = _vocab.get(best.first);
            std::vector<uint8_t> rightBytes = _vocab.get(best.second);

            std::vector<uint8_t> newBytes;
            newBytes.insert(newBytes.end(), leftBytes.begin(), leftBytes.end());
            newBytes.insert(newBytes.end(), rightBytes.begin(), rightBytes.end());

            const int newId = _vocab.add(newBytes);

            chunks = merge(chunks, best, newId);

            _merges.push_back({best, newId});
        }
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> tokens;

        std::vector<std::pair<std::string, bool>> segments;
        std::string remaining = text;

        while (!remaining.empty()) {
            std::size_t earliestPos = std::string::npos;
            std::string earliestToken;

            for (const auto&[specialStr, id] : _specialTokens) {
                const std::size_t pos = remaining.find(specialStr);
                if (pos != std::string::npos && (earliestPos == std::string::npos || pos < earliestPos)) {
                    earliestPos = pos;
                    earliestToken = specialStr;
                }
            }

            if (earliestPos == std::string::npos) {
                segments.push_back({remaining, false});
                break;
            }

            if (earliestPos > 0) {
                segments.emplace_back(remaining.substr(0, earliestPos), false);
            }

            segments.push_back({earliestToken, true});

            remaining = remaining.substr(earliestPos + earliestToken.size());
        }

        for (const auto& [str, isSpecial] : segments) {
            if (isSpecial) {
                tokens.push_back(_specialTokens.at(str));
            } else {
                Tokenizer tokenizer(str);

                std::vector<std::vector<int>> chunks = tokenizer.tokenize();

                for (const auto& [pair, newId] : _merges) {
                    chunks = merge(chunks, pair, newId);
                }

                for (const auto& chunk : chunks) {
                    tokens.insert(tokens.end(), chunk.begin(), chunk.end());
                }
            }
        }

        return tokens;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string text;

        for (const int id : ids) {
            const auto& bytes = _vocab.get(id);
            text.append(bytes.begin(), bytes.end());
        }

        return text;
    }

    void addSpecialToken(const std::string& token) {
        const std::vector<uint8_t> bytes(token.begin(), token.end());
        const int id = _vocab.add(bytes);

        _specialTokens[token] = id;
    }

    Vocab getVocab() const {
        return _vocab;
    }

    void save(const std::string& path) const {
        std::ofstream file(path, std::ios::binary);

        const int vocabSize = _vocab.size();
        file.write(reinterpret_cast<const char*>(&vocabSize), sizeof(int));

        for (int i = 0; i < _vocab.size(); i++) {
            std::vector<uint8_t> bytes = _vocab.get(i);
            int length = bytes.size();

            file.write(reinterpret_cast<const char*>(&length), sizeof(int));
            file.write(reinterpret_cast<const char*>(bytes.data()), length);
        }

        const int mergeCount = _merges.size();
        file.write(reinterpret_cast<const char*>(&mergeCount), sizeof(int));
        for (const auto& merge : _merges) {
            file.write(reinterpret_cast<const char*>(&merge.pair.first), sizeof(int));
            file.write(reinterpret_cast<const char*>(&merge.pair.second), sizeof(int));
            file.write(reinterpret_cast<const char*>(&merge.newId), sizeof(int));
        }

        const int specialTokenCount = _specialTokens.size();
        file.write(reinterpret_cast<const char*>(&specialTokenCount), sizeof(int));
        for (const auto& [str, id] : _specialTokens) {
            int length = str.size();
            file.write(reinterpret_cast<const char*>(&length), sizeof(int));
            file.write(str.data(), length);
            file.write(reinterpret_cast<const char*>(&id), sizeof(int));
        }
    }

    static Bpe load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);

        if (!f.good()) {
            throw std::runtime_error("Couldn't load BPE: " + path + " - does the path exist?");
        }

        Vocab vocab;

        // load vocab
        int vocabSize;
        f.read(reinterpret_cast<char*>(&vocabSize), sizeof(int));
        for (int i = 0; i < vocabSize; i++) {
            int length;
            f.read(reinterpret_cast<char*>(&length), sizeof(int));
            std::vector<uint8_t> bytes(length);
            f.read(reinterpret_cast<char*>(bytes.data()), length);
            vocab.add(bytes);
        }

        Bpe bpe(vocab);

        int mergeCount;
        f.read(reinterpret_cast<char*>(&mergeCount), sizeof(int));
        for (int i = 0; i < mergeCount; i++) {
            int a, b, newId;
            f.read(reinterpret_cast<char*>(&a),     sizeof(int));
            f.read(reinterpret_cast<char*>(&b),     sizeof(int));
            f.read(reinterpret_cast<char*>(&newId), sizeof(int));
            bpe._merges.push_back({{a, b}, newId});
        }

        int specialCount;
        f.read(reinterpret_cast<char*>(&specialCount), sizeof(int));
        for (int i = 0; i < specialCount; i++) {
            int length;
            f.read(reinterpret_cast<char*>(&length), sizeof(int));
            std::string str(length, '\0');
            f.read(str.data(), length);
            int id;
            f.read(reinterpret_cast<char*>(&id), sizeof(int));
            bpe._specialTokens[str] = id;
        }

        return bpe;
    }

private:
    static PairCounts countPairs(const std::vector<std::vector<int>>& chunks) {
        PairCounts counts;
        for (const auto& tokens : chunks) {
            for (size_t i = 0; i + 1 < tokens.size(); i++) {
                counts[{tokens[i], tokens[i+1]}]++;
            }
        }

        return counts;
    }

    static Pair mostFrequentPair(const PairCounts& counts) {
        int maxCount = 0;
        Pair best = {-1, -1};
        for (const auto& p : counts) {
            if (p.second > maxCount) {
                maxCount = p.second;
                best = p.first;
            }
        }
        return best;
    }

    static std::vector<std::vector<int>> merge(const std::vector<std::vector<int>>& chunks, const Pair &pair, int newId) {
        std::vector<std::vector<int>> newChunks;
        newChunks.reserve(chunks.size());

        for (const auto& tokens : chunks) {
            std::vector<int> result;
            result.reserve(tokens.size());

            std::size_t i = 0;

            while (i < tokens.size()) {
                if (i + 1 < tokens.size() &&
                    tokens[i] == pair.first &&
                    tokens[i+1] == pair.second) {
                    result.push_back(newId);
                    i += 2;
                    } else {
                        result.push_back(tokens[i]);
                        i += 1;
                    }
            }
            newChunks.push_back(std::move(result));
        }

        return newChunks;
    }

private:
    Vocab _vocab;
    std::vector<Merge> _merges;
    std::unordered_map<std::string, int> _specialTokens;
};