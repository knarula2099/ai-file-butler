# file_butler/interfaces/cli.py
"""
AI File Butler CLI - Enhanced with LLM and ML capabilities

This module provides the main CLI interface for the AI File Butler tool.
It now includes support for multiple organization strategies including:
- Rules-based organization (default)
- ML clustering-based organization  
- LLM-powered semantic organization
"""

# Import the enhanced CLI which has all the LLM and ML capabilities
from .enhanced_cli import cli

# Re-export the CLI for backward compatibility
__all__ = ['cli']

if __name__ == "__main__":
    cli()