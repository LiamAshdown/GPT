#pragma once

#include <typeindex>

class ListenerHandle {

public:
    ListenerHandle(): _type(typeid(void)), _id(0) {

    }

    ListenerHandle(const std::type_index type, const std::size_t id): _type(type), _id(id) {}

public:
    std::type_index getType() const { return _type; }
    std::size_t getId() const { return _id; }
    bool valid() const { return _type != typeid(void); }

private:
    std::type_index _type;
    std::size_t _id;
};
