"""The Windows event-loop setup repair must not permit application networking."""
import socket

import pytest


def test_create_connection_still_forbidden():
    with pytest.raises(AssertionError, match="Network calls are forbidden"):
        socket.create_connection(("127.0.0.1", 9))


def test_direct_connect_still_forbidden():
    with socket.socket() as connection:
        with pytest.raises(AssertionError, match="Network calls are forbidden"):
            connection.connect(("127.0.0.1", 9))
