#pragma once

#include "event-handler.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

class HandlerPool {
public:
    struct Slot {
        std::size_t id{0};
        std::unique_ptr<EventHandler> handler;
    };

    void add(const std::size_t id, std::unique_ptr<EventHandler> handler) {
        std::unique_lock lock(_mutex);
        _slots.push_back({ id, std::move(handler) });
    }

    void remove(std::size_t id) {
        std::unique_lock lock(_mutex);
        _slots.erase(std::remove_if(_slots.begin(), _slots.end(),
            [id](const Slot& s) { return s.id == id; }), _slots.end());
    }

    void dispatch(const Event& event) const {
        for (const auto&[id, handler] : _slots)
            handler->handle(event);
    }

private:
    std::vector<Slot> _slots;
    std::mutex _mutex;
};
