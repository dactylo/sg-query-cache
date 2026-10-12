import socket
import ssl
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TLS_CA = Path(__file__).parent / 'fixtures' / 'tls' / 'ca.pem'
TLS_SERVER_CERTIFICATE = Path(__file__).parent / 'fixtures' / 'tls' / 'server.pem'
# This private key is deliberately public and disposable. It is used only by local tests.
TLS_SERVER_KEY = Path(__file__).parent / 'fixtures' / 'tls' / 'server-key.pem'

_HANDSHAKE_TIMEOUT_SECONDS = 2.0


def perform_tls_handshake(
    wrap_client: Callable[[socket.socket], ssl.SSLSocket],
) -> None:
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(TLS_SERVER_CERTIFICATE, TLS_SERVER_KEY)
    client_socket, server_socket = socket.socketpair()
    client_socket.settimeout(_HANDSHAKE_TIMEOUT_SECONDS)
    server_socket.settimeout(_HANDSHAKE_TIMEOUT_SECONDS)

    def serve() -> None:
        with server_context.wrap_socket(server_socket, server_side=True) as secured:
            # Keep both endpoints open until both handshakes have completed.
            assert secured.recv(1) == b'?'
            secured.sendall(b'!')

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            server = executor.submit(serve)
            with wrap_client(client_socket) as secured:
                secured.sendall(b'?')
                assert secured.recv(1) == b'!'
            server.result(timeout=_HANDSHAKE_TIMEOUT_SECONDS)
        finally:
            client_socket.close()
            server_socket.close()
