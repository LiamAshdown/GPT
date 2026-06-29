#pragma once

#include "event-handler.h"

template<typename E, typename T>
class MemberHandler final : public EventHandler {
public:
    using MethodPtr = void (T::*)(const E &);

    MemberHandler(T *instance, MethodPtr method) : _instance(instance), _method(method) {
    }

public:
    void handle(const Event &event) override {
        (_instance->*_method)(static_cast<const E &>(event));
    }

private:
    T *_instance;
    MethodPtr _method;
};
