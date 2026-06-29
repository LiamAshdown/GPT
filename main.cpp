#include "llm/gpt/gpt.h"
#include "llm/bpe/bpe.h"
#include "llm/bpe/tokenizer.h"

#include "util/event-dispatcher/event-dispatcher.h"
#include "util/event-dispatcher/events/events.h"
#include "util/event-dispatcher/listeners/sample-window-listener.h"
#include "util/event-dispatcher/listeners/gpt-listener.h"

#include "util/logger/logger.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static constexpr int MAX_SEQ_LEN  = 64;
static constexpr int D_MODEL      = 256;
static constexpr int N_LAYERS     = 6;
static constexpr int N_HEADS      = 4;
static constexpr int BPE_VOCAB    = 4000;
static constexpr int TRAIN_STEPS  = 50000;

std::string loadText(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Could not open: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main() {
    std::cout << "=== Tiny GPT ===\n\n";

    spdlog::stdout_color_mt("console");
    auto console = spdlog::get("console");

    if (torch::mps::is_available()) {
        // globalDevice() = torch::Device("mps");
        console->info("using MPS (Apple GPU)");
    } else {
        console->info("using CPU");
    }

    Bpe bpe(Vocab::buildBaseVocab());

    if (std::ifstream("bpe.bin").good()) {
        console->info("found bpe.bin — loading...");
        bpe = Bpe::load("bpe.bin");
    } else {
        console->info("no bpe.bin found — training BPE from scratch...");

        const std::string corpus = loadText("../data/war-and-peace.txt");
        console->info("loaded corpus: {} bytes", corpus.size());

        const Tokenizer tokenizer(corpus);

        bpe.train(tokenizer, BPE_VOCAB);
        bpe.save("bpe.bin");
    }

    console->info("BPE ready, vocab size = {}", bpe.getVocab().size());

    EventDispatcher dispatcher;
    SampleWindowListener sampleListener;
    GptListener gptListener(console);

    dispatcher.subscribe<SampleWindowEvent>(&sampleListener, &SampleWindowListener::onSampleWindow);
    dispatcher.subscribe<BatchLossEvent>(&gptListener, &GptListener::onBatchLoss);
    dispatcher.subscribe<StepEvent>(&gptListener, &GptListener::onStep);

    GPT gpt(bpe.getVocab().size(), MAX_SEQ_LEN, D_MODEL, N_LAYERS, N_HEADS, &dispatcher);

    if (std::ifstream("model.pt").good()) {
        console->info("found model.pt - loading weights, skipping training...");
        gpt.load("model.pt");
    } else {
        console->info("no model.pt found — training from scratch...");

        std::vector<int> tokens;

        if (std::ifstream("tokens.bin").good()) {
            console->info("found tokens.bin — loading cached tokens...");
            std::ifstream f("tokens.bin", std::ios::binary);
            int size;
            f.read(reinterpret_cast<char*>(&size), sizeof(int));
            tokens.resize(size);
            f.read(reinterpret_cast<char*>(tokens.data()), size * sizeof(int));
            console->info("loaded {} tokens", tokens.size());
        } else {
            const std::string corpus = loadText("../data/war-and-peace.txt");
            console->info("loaded corpus: {} bytes", corpus.size());

            console->info("encoding corpus...");
            tokens = bpe.encode(corpus);
            console->info("encoded {} tokens — saving to tokens.bin...", tokens.size());

            std::ofstream f("tokens.bin", std::ios::binary);
            int size = static_cast<int>(tokens.size());
            f.write(reinterpret_cast<const char*>(&size), sizeof(int));
            f.write(reinterpret_cast<const char*>(tokens.data()), size * sizeof(int));
        }

        gpt.fit(tokens, TRAIN_STEPS);
        gpt.save("model.pt");
        console->info("model saved to model.pt");
    }

    auto start = "There was a time";

    auto encoded = bpe.encode(start);
    auto result  = gpt.predict(encoded, 100, 1.0f);

    // std::vector<int> newTokens(result.begin() + encoded.size(), result.end());
    console->info("start: {} \n prediction: {}", start, bpe.decode(result));

    return 0;
}
