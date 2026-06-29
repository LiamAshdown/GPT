#pragma once

#include "event.h"

class EventHandler {
public:
    virtual ~EventHandler() = default;
    virtual void handle(const Event& event) = 0;
};