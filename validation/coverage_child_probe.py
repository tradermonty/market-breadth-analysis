"""Measured only from a Python child to detect lost subprocess instrumentation."""


def child_marker(value):
    if value > 0:
        return value + 1
    return value - 1
