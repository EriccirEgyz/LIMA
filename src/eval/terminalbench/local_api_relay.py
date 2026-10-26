"""Temporary TCP forwarding for a private Docker namespace's local model API."""

import asyncio
from concurrent.futures import Future
from contextlib import contextmanager, suppress
from ipaddress import ip_address
import socket
from threading import Thread
from typing import Iterator


MAX_CONNECTIONS = 256
BUFFER_SIZE = 64 * 1024
CONNECT_TIMEOUT = 10
CLOSE_TIMEOUT = 3


async def _close_writer(writer: asyncio.StreamWriter | None) -> None:
    if writer is None:
        return
    writer.close()
    try:
        await asyncio.wait_for(writer.wait_closed(), CLOSE_TIMEOUT)
    except TimeoutError:
        writer.transport.abort()
    except (ConnectionError, OSError):
        pass


async def _copy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while data := await reader.read(BUFFER_SIZE):
        writer.write(data)
        await writer.drain()
    # An HTTP client may half-close its request and still need the response.
    if writer.can_write_eof() and not writer.is_closing():
        writer.write_eof()
        await writer.drain()


class _Relay:
    def __init__(self, upstream_host: str, upstream_port: int, allowed_peers: set[str]):
        self.upstream = (upstream_host, upstream_port)
        self.allowed_peers = allowed_peers
        self.tasks: set[asyncio.Task] = set()
        self.writers: set[asyncio.StreamWriter] = set()
        self.closing = False

    def accept(self, client: socket.socket, peer: str) -> None:
        if (self.closing or str(ip_address(peer)) not in self.allowed_peers
                or len(self.tasks) >= MAX_CONNECTIONS):
            client.close()
            return
        task = asyncio.create_task(self.forward(client))
        self.tasks.add(task)

        def finished(done: asyncio.Task) -> None:
            self.tasks.discard(done)
            # Also closes sockets whose forwarding task never got to start.
            client.close()

        task.add_done_callback(finished)

    async def forward(self, client: socket.socket) -> None:
        writer = None
        upstream_writer = None
        pumps = []
        try:
            reader, writer = await asyncio.open_connection(sock=client, limit=BUFFER_SIZE)
            self.writers.add(writer)
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection(*self.upstream), CONNECT_TIMEOUT,
            )
            self.writers.add(upstream_writer)
            pumps = [
                asyncio.create_task(_copy(reader, upstream_writer)),
                asyncio.create_task(_copy(upstream_reader, writer)),
            ]
            await asyncio.gather(*pumps)
        except (ConnectionError, OSError, TimeoutError):
            # Reset/refused connections are isolated; no requests are inspected.
            pass
        finally:
            for task in pumps:
                task.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)
            await asyncio.gather(_close_writer(writer), _close_writer(upstream_writer))
            self.writers.discard(writer)
            self.writers.discard(upstream_writer)

    async def close(self) -> None:
        self.closing = True
        tasks = list(self.tasks)
        # Abort buffered writes before cancellation, including half-closed peers.
        # This keeps shutdown bounded even when the other endpoint stops reading.
        for writer in self.writers:
            writer.transport.abort()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _specific_ip(value: str) -> str:
    address = ip_address(value)
    if address.is_unspecified or address.is_multicast:
        raise ValueError("Relay addresses must be explicit, non-multicast IP addresses")
    return str(address)


@contextmanager
def local_api_relay(
    listen_host: str,
    listen_port: int,
    upstream_host: str,
    upstream_port: int,
    allowed_peers: tuple[str, ...],
) -> Iterator[int]:
    """Forward bytes for the context's lifetime and yield the bound TCP port.

    All addresses must be explicit IPs, and only the listed peers may connect.
    Port zero is supported for isolated local checks. Binding failures propagate
    to the caller; an existing listener is never reused or stopped. The relay
    forwards HTTP/SSE unchanged, without inspecting or recording any payload.
    """
    listen_host = _specific_ip(listen_host)
    upstream_host = _specific_ip(upstream_host)
    peers = {_specific_ip(peer) for peer in allowed_peers}
    if not peers:
        raise ValueError("Relay allowed_peers must not be empty")
    if not 0 <= listen_port <= 65535 or not 1 <= upstream_port <= 65535:
        raise ValueError("Invalid relay port")

    family = socket.AF_INET6 if ip_address(listen_host).version == 6 else socket.AF_INET
    listener = socket.socket(family, socket.SOCK_STREAM)
    loop = None
    thread = None
    started: Future[int] = Future()
    finished: Future[None] = Future()
    stopped = asyncio.Event()

    async def serve() -> None:
        relay = _Relay(upstream_host, upstream_port, peers)

        def accept_ready() -> None:
            # Register every accepted socket before yielding to the event loop.
            # start_server() instead has internal accept tasks that can outlive
            # shutdown before its StreamReader callback has seen the socket.
            for _ in range(128):
                try:
                    client, peer = listener.accept()
                except (BlockingIOError, InterruptedError):
                    return
                client.setblocking(False)
                relay.accept(client, peer[0])

        try:
            loop.add_reader(listener.fileno(), accept_ready)
            started.set_result(listener.getsockname()[1])
            await stopped.wait()
        finally:
            loop.remove_reader(listener.fileno())
            listener.close()
            await relay.close()

    def run() -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(serve())
        except BaseException as error:
            if not started.done():
                started.set_exception(error)
            finished.set_exception(error)
        else:
            finished.set_result(None)
        finally:
            loop.close()

    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((listen_host, listen_port))
        listener.listen(128)
        listener.setblocking(False)
        loop = asyncio.new_event_loop()
        thread = Thread(target=run, name="terminalbench-api-relay")
        thread.start()
        yield started.result()
    finally:
        if thread is not None and thread.ident is not None:
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(stopped.set)
            thread.join()
        elif loop is not None:
            loop.close()
        listener.close()
    finished.result()
