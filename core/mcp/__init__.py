"""
MCP (Model Context Protocol) Integration

Connects to MCP servers to fetch documents from various sources:
- Notion
- GitHub
- Google Drive
- etc.
"""

from core.mcp.notion_connector import NotionMCPConnector

__all__ = ["NotionMCPConnector"]
