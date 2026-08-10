import importlib.util
import json
import logging
import os
import sys
from typing import Callable, Optional, Tuple, List, Dict, Any

import ray
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "WARN"))


class ToolClient:
    """Loads a local Python tool module and calls its tools directly, bypassing FastMCP async."""

    def __init__(self, server_name: str, server_path: str):
        self.server_name = server_name
        self.server_path = server_path
        self.module = None
        self.tools: Dict[str, dict] = {}       # full_name -> schema
        self._fns: Dict[str, Callable] = {}    # short_name -> callable
        self.connected = False

    def connect(self) -> None:
        """Load the Python module and extract tool schemas + function references."""
        if self.connected:
            return

        spec = importlib.util.spec_from_file_location(
            f"envfactory.tools.{self.server_name}", self.server_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {self.server_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "mcp"):
            raise AttributeError(
                f"Module {self.server_path} has no 'mcp' attribute. "
                "Each tool module must define a FastMCP instance named 'mcp'."
            )

        self.module = module
        self._extract_tools(module.mcp)
        self.connected = True

    def _extract_tools(self, mcp) -> None:
        """Extract tool schemas and direct function references from FastMCP."""
        internal_tools = mcp._tool_manager._tools

        for name, tool in internal_tools.items():
            full_name = f"{self.server_name}-{name}"
            params = getattr(tool, "parameters", None) or getattr(tool, "inputSchema", {})
            self.tools[full_name] = {
                "type": "function",
                "function": {
                    "name": full_name,
                    "description": tool.description,
                    "parameters": params,
                },
            }
            self._fns[name] = tool.fn

    def call_tool(self, tool_name: str, tool_args: dict) -> str:
        """Call a tool directly by invoking its underlying function."""
        if not self.connected:
            raise RuntimeError(f"Client {self.server_name} is not connected.")

        fn = self._fns.get(tool_name)
        if fn is None:
            raise ValueError(f"Unknown tool '{tool_name}' in server '{self.server_name}'")

        result = fn(**tool_args)
        if isinstance(result, dict):
            import json as _json
            return _json.dumps(result)
        return str(result)

    def get_tool_schemas(self) -> List[dict]:
        """Return all tool schemas for this server."""
        return list(self.tools.values())

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool exists in this server."""
        return tool_name in self._fns

    def close(self) -> None:
        """Close the client (no-op for local modules, just clears references)."""
        self.module = None
        self.tools.clear()
        self._fns.clear()
        self.connected = False


class ToolManager:
    """Manages multiple concurrent local tool clients."""
    _instance = None # Private class variable to store the unique instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ToolManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = False

            # Client management
            self.clients: Dict[str, dict] = {}
            self._prototypes: Dict[str, ToolClient] = {}
            self.server_to_path_mapping: Dict[str, str] = {}
            self.tools: Dict[str, dict] = {}

    def register_server(self, server_name: str, server_path: str) -> None:
        """Register a tool server and extract its tool schemas."""
        client = ToolClient(server_name, server_path)
        client.connect()

        self.server_to_path_mapping[server_name] = server_path
        self._prototypes[server_name] = client
        for tool_name, schema in client.tools.items():
            self.tools[tool_name] = schema

    def init_config(self, config_path: str) -> None:
        """Initialize manager from MCP config file."""
        if self._initialized:
            return

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        logger.info(
            f"Initializing {len(config['mcpServers'])} tool servers from: {config_path}..."
        )
        for server_name, server_config in config["mcpServers"].items():
            self.register_server(server_name, server_config["tool_path"])

        self._initialized = True

    def get_tools(self) -> Dict[str, dict]:
        """Return all registered tool schemas (name -> schema dict)."""
        return dict(self.tools)

    def get_tool_schemas(self) -> List[dict]:
        """Return all registered tool schemas as a list."""
        return list(self.tools.values())

    def filter_tools(self, servers: Optional[List[str]] = None) -> List[dict]:
        """Filter tools by server names."""
        if servers is None:
            return list(self.tools.values())

        result = []
        for name, schema in self.tools.items():
            server_name, _, tool_name = name.partition("-")
            if server_name in servers and tool_name not in [
                "load_scenario",
                "save_scenario",
            ]:
                result.append(schema)
        return result

    def _get_or_create_client(self, client_id: str) -> Tuple[ToolClient, bool]:
        """Get or create a client for the given client_id.

        client_id format: ``{server_name}-{request_id}``
        Each unique client_id gets its own ToolClient instance.
        """
        if client_id in self.clients:
            client_info = self.clients[client_id]
            return client_info["client"], client_info["loaded"]

        server_name = client_id.split("-")[0]

        # Create a new client by loading the module again (independent instance)
        server_path = self.server_to_path_mapping.get(server_name)
        if server_path is None:
            raise ValueError(f"No server path registered for '{server_name}'")

        client = ToolClient(server_name, server_path)
        client.connect()
        self.clients[client_id] = {"client": client, "loaded": False}
        return client, False

    def call_tool(
        self, client_id: str, tool_name: str, tool_args: dict | str
    ) -> str:
        """Execute a tool with proper session handling.

        Args:
            client_id: ``{server_name}-{request_id}`` identifying the client session.
            tool_name: Full tool name ``{server_name}-{tool_func}`` or just ``{tool_func}``.
            tool_args: Arguments as dict or JSON string.
        """
        name = tool_name.split("-")[-1]
        args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
        client, loaded = self._get_or_create_client(client_id)

        if name == "load_scenario" and loaded:
            return "Already loaded scenario, skipping load."

        try:
            result = client.call_tool(name, args)
            if name == "load_scenario":
                self.clients[client_id]["loaded"] = True
            return result
        except TimeoutError:
            return f"The {tool_name} timed out."
        except Exception as exc:
            return f"The {tool_name} error: {exc}."

    def close_client(self, client_id: str) -> None:
        """Close a specific client by client_id."""
        client_info = self.clients.pop(client_id, None)
        if client_info:
            client_info["client"].close()

    def save_and_close_clients(self, client_ids: List[str]) -> Dict[str, Any]:
        """Save scenarios and close clients."""
        results = {}
        for client_id in client_ids:
            server_name = client_id.split("-")[0]
            try:
                result = self.call_tool(client_id, f"{server_name}-save_scenario", {})
            except Exception as e:
                logger.info(f"Error saving scenario for {server_name}: {e}")
                result = str(e)
            try:
                self.close_client(client_id)
            except Exception:
                pass
            results[server_name] = result
        return results

    def close_all_clients(self) -> None:
        """Close all clients."""
        clients = list(self.clients.values())
        self.clients.clear()
        for client_info in clients:
            try:
                client_info["client"].close()
            except Exception:
                pass

    def shutdown(self) -> None:
        """Shutdown the tool manager."""
        self.close_all_clients()

    def __del__(self):
        self.shutdown()


@ray.remote(num_cpus=1)
class ToolManagerActor:
    """Ray actor for ToolManager — same interface as MCPManagerActor."""

    def __init__(self):
        self._manager = ToolManager()

    # Lifecycle
    def init_config(self, config_path: str):
        return self._manager.init_config(config_path)

    def shutdown(self):
        return self._manager.shutdown()

    # Data access
    def get_tools(self) -> Dict[str, dict]:
        return self._manager.get_tools()

    def get_tool_schemas(self) -> List[dict]:
        return self._manager.get_tool_schemas()

    def get_server_to_path_mapping(self) -> Dict[str, str]:
        return self._manager.server_to_path_mapping

    def filter_tools(self, servers: Optional[List[str]] = None) -> List[dict]:
        return self._manager.filter_tools(servers)

    # Tool execution
    def call_tool(self, client_id: str, tool_name: str, tool_args: dict) -> str:
        return self._manager.call_tool(client_id, tool_name, tool_args)

    def save_and_close_clients(self, client_ids: List[str]) -> Dict[str, Any]:
        return self._manager.save_and_close_clients(client_ids)
