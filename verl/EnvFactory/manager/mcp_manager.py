import asyncio
import json
import logging
import os
import threading
from typing import Optional, Tuple, List, Dict, Any

import ray
from dotenv import load_dotenv
from fastmcp import Client

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "WARN"))


class MCPManager:
    """Manages multiple concurrent MCP clients using FastMCP."""
    _instance = None  # Private class variable to store the unique instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MCPManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):  # The singleton should only be inited once
            """Set a new event loop in a separate thread"""
            self._initialized = False

            # Client management
            self.clients: Dict[str, dict] = {} # stateful clients
            self.stateless_clients: Dict[str, Client] = {} # stateless clients
            self.server_to_path_mapping: Dict[str, str] = {}
            self.tools: Dict[str, dict] = {}

            # Loop management
            self.loop = asyncio.new_event_loop()
            self.loop_thread = threading.Thread(target=self.start_loop, daemon=True)
            self.loop_thread.start()

    def start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def register_mcp_server_async(self, server_name: str, server_path: str, is_stateless: bool = False):
        """Register an MCP server and extract tool schemas."""
        client = Client(server_path)
        await client._connect()
        tools = await client.list_tools()
        schemas = [
            (f"{server_name}-{tool.name}", {
                "type": "function",
                "function": {
                    "name": f"{server_name}-{tool.name}",
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            })
            for tool in tools
        ]

        self.server_to_path_mapping[server_name] = server_path
        for tool_name, schema in schemas:
            self.tools[tool_name] = schema

        if is_stateless:
            self.stateless_clients[server_name] = client
        else:
            await client.close()

    async def _init_config_async(self, config: dict):
        """Async initialization of all configured servers."""
        if self._initialized:
            return

        tasks = [
            self.register_mcp_server_async(
                server_name, server_config["tool_path"], server_config["stateless"]
            )
            for server_name, server_config in config["mcpServers"].items()
        ]
        await asyncio.gather(*tasks)
        self._initialized = True

    def init_config(self, config_path):
        """Initialize manager from MCP config file."""
        if self._initialized:
            return

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        logger.info(f"Initializing {len(config["mcpServers"])} MCP servers from: {config_path}...")
        future = asyncio.run_coroutine_threadsafe(
            self._init_config_async(config), self.loop,
        )
        return future.result(timeout=150)

    def filter_tools(self, servers: Optional[List[str]] = None) -> List[dict]:
        """Filter tools by server names."""
        if servers is None:
            return list(self.tools.values())

        result = []
        for name, schema in self.tools.items():
            server_name, _, tool_name = name.partition("-")
            if server_name in servers and tool_name not in ["load_scenario", "save_scenario"]:
                result.append(schema)
        return result

    def get_client(self, client_id: str) -> Tuple[Client, bool]:
        """Get or create a client based on client_id."""
        server_name = client_id.split("-")[0]

        if client_id in self.clients:
            client_info = self.clients[client_id]
            return client_info["client"], client_info["loaded"]

        if server_name in self.stateless_clients:
            return self.stateless_clients[server_name], True

        server_path = self.server_to_path_mapping[server_name]
        client = Client(server_path)
        self.clients[client_id] = {"client": client, "loaded": False}
        return client, False

    async def _call_tool_async(self, client_id: str, tool_name: str, tool_args: dict | str) -> str:
        """Execute tool with proper session handling."""
        name = tool_name.split("-")[-1]
        args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
        client, loaded = self.get_client(client_id)
        if "load_scenario" == name and loaded:
            return "Already loaded scenario, skipping load."

        if not client.is_connected():
            await client._connect()
        result = await client.call_tool(name, args)
        return ",".join(item.text for item in result.content if hasattr(item, "text"))

    def call_tool(self, client_id: str, tool_name: str, tool_args: dict | str):
        """Sync wrapper to call a tool."""
        future = asyncio.run_coroutine_threadsafe(
            self._call_tool_async(client_id, tool_name, tool_args), self.loop,
        )

        try:
            return future.result(timeout=30)
        except TimeoutError:
            return f"The {tool_name} timed out after 30 seconds."
        except Exception as exc:
            return f"The {tool_name} error: {exc}."

    async def _close_client_async(self, client_id: str):
        client_info = self.clients.pop(client_id, None)
        if client_info and client_info["client"].is_connected():
            try:
                await client_info["client"].close()
            except Exception as e:
                logger.info(f"⚠️ Error when closing client {client_id}: {e}")

    def close_client(self, client_id: str):
        future = asyncio.run_coroutine_threadsafe(
            self._close_client_async(client_id), self.loop,
        )
        return future.result(timeout=30)

    async def _save_and_close_clients_async(self, client_ids: List[str]) -> Dict[str, Any]:
        async def _save_one(client_id: str) -> Tuple[str, Any]:
            server_name = client_id.split("-")[0]
            try:
                result = await self._call_tool_async(client_id, "save_scenario", {})
            except Exception as e:
                logger.info(f"Error saving scenario for {server_name}: {e}")
                result = str(e)
            try:
                await self._close_client_async(client_id)
            except Exception:
                pass
            return server_name, result

        pairs = await asyncio.gather(*[_save_one(cid) for cid in client_ids])
        return dict(pairs)

    def save_and_close_clients(self, client_ids: List[str]) -> Dict[str, Any]:
        future = asyncio.run_coroutine_threadsafe(
            self._save_and_close_clients_async(client_ids), self.loop,
        )
        try:
            return future.result(timeout=60)
        except TimeoutError:
            return {cid.split("-")[0]: {} for cid in client_ids}

    async def _close_all_stateless_clients_async(self):
        clients = list(self.stateless_clients.values())
        self.stateless_clients.clear()

        await asyncio.gather(
            *(c.close() for c in clients),
            return_exceptions=True,
        )

    async def _close_all_clients_async(self):
        clients = [info["client"] for info in self.clients.values()]
        self.clients.clear()

        await asyncio.gather(
            *(c.close() for c in clients),
            return_exceptions=True,
        )

    def close_all_clients(self):
        if not self.loop.is_running():
            return

        future = asyncio.run_coroutine_threadsafe(
            self._close_all_clients_async(),
            self.loop,
        )
        return future.result(timeout=30)

    def shutdown(self, timeout: int = 60):
        if not self.loop.is_running():
            return

        asyncio.run_coroutine_threadsafe(
            self._close_all_stateless_clients_async(), self.loop
        ).result(timeout=timeout)

        self.close_all_clients()

        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=timeout)

    def __del__(self):
        self.shutdown()


@ray.remote(num_cpus=1)
class MCPManagerActor:
    """Ray actor for MCP manager."""

    def __init__(self):
        self._manager = MCPManager()

    # Lifecycle
    def init_config(self, config_path: str):
        return self._manager.init_config(config_path)

    def shutdown(self):
        return self._manager.shutdown()

    # Data access
    def get_tools(self) -> Dict[str, dict]:
        return self._manager.tools

    def get_tool_schemas(self) -> List[dict]:
        return list(self._manager.tools.values())

    def get_server_to_path_mapping(self) -> Dict[str, str]:
        return self._manager.server_to_path_mapping

    def filter_tools(self, servers: Optional[List[str]] = None) -> List[dict]:
        return self._manager.filter_tools(servers)

    # Tool execution
    def call_tool(self, client_id: str, tool_name: str, tool_args: dict) -> str:
        return self._manager.call_tool(client_id, tool_name, tool_args)

    def save_and_close_clients(self, client_ids: List[str]) -> Dict[str, Any]:
        return self._manager.save_and_close_clients(client_ids)
