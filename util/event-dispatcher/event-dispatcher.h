#pragma once

#include "priority.h"
#include "event.h"
#include "handler-pool.h"
#include "member-handler.h"
#include "listener-handle.h"

#include <shared_mutex>
#include <typeindex>
#include <type_traits>
#include <unordered_map>
#include <array>

class EventDispatcher {
public:
    template<typename E, typename T>
    ListenerHandle subscribe(T* instance,
        void (T::*method)(const E&),
        const Priority priority = Priority::Normal) {
        static_assert(std::is_base_of_v<Event, E>, "E must derive from Event");

        auto key    = std::type_index(typeid(E));
        auto id     = _next_id.fetch_add(1, std::memory_order_relaxed);
        auto& pool  = poolFor(key, priority);
        auto handler = std::make_unique<MemberHandler<E, T>>(instance, method);
        pool.add(id, std::move(handler));
        return { key, id };
    }

    void unsubscribe(const ListenerHandle& handle,
                     const Priority priority = Priority::Normal)
    {
        auto& pool = poolFor(handle.getType(), priority);
        pool.remove(handle.getId());
    }

    template<typename E>
    void emit(const E& event) {
        static_assert(std::is_base_of_v<Event, E>, "E must derive from Event");
        const auto key = std::type_index(typeid(E));

        std::shared_lock lock(_mutex);
        const auto it = _pools.find(key);
        if (it == _pools.end()) {
            return;
        }

        auto& tier = it->second;
        for (int p = kPriorityCount - 1; p >= 0; --p)
            tier[p].dispatch(event);
    }

private:
    using PriorityTier = std::array<HandlerPool, kPriorityCount>;

    HandlerPool& poolFor(const std::type_index key, Priority priority) {
        std::unique_lock lock(_mutex);
        return _pools[key][static_cast<std::size_t>(priority)];
    }

    std::unordered_map<std::type_index, PriorityTier> _pools;
    std::shared_mutex _mutex;
    std::atomic<std::size_t> _next_id{ 1 };
};
