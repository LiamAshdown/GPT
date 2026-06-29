#pragma once

#include <string>
#include <vector>

class Tokenizer {
public:
    explicit Tokenizer(std::string corpus) : _corpus(std::move(corpus)) {
    }

    std::vector<std::vector<int> > tokenize() const {
        std::vector<std::vector<int>> chunks;
        std::string current;

        auto flushChunk = [&chunks, &current]() {
            pushChunk(chunks, current);
            current.clear();
        };

        for (size_t i = 0; i < _corpus.size(); i++) {
            const unsigned char c = _corpus[i];

            if (c == ' ') {
                // it's a space, clear current and start with new
                if (!current.empty()) {
                    flushChunk();
                }

                // next one is a letter, so continue
                if (i + 1 < _corpus.size() && std::isalnum((unsigned char) _corpus[i + 1])) {
                    current += c;
                } else {
                    // else it's not start a new chunk
                    pushChunk(chunks, std::string(1, c));
                }
            } else if (std::isalnum(c)) {
                // previous current is not a letter and the next one is
                // so clear current and start with new
                if (!current.empty() && !std::isalnum((unsigned char) current[0])) {
                    flushChunk();
                }
                current += c;
            } else {
                // not a letter, so clear the current and start with new
                if (!current.empty() && std::isalnum((unsigned char) current[0])) {
                    flushChunk();
                }
                current += c;
            }
        }

        if (!current.empty()) {
            flushChunk();
        }

        return chunks;
    }

private:
    static void pushChunk(std::vector<std::vector<int> > &chunks, const std::string &string) {
        std::vector<int> ids;
        ids.reserve(string.size());

        for (unsigned char c: string) {
            ids.push_back(static_cast<int>(c));
        }

        chunks.push_back(ids);
    }

private:
    std::string _corpus;
};
