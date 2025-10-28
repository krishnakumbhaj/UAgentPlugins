"""Universal uAgentsPlugin for uAgents - works with any AI framework.

Simple plugin to wrap LlamaIndex, CrewAI, LangChain, OpenAI SDK, AutoGen, or custom logic in uAgents.

Usage:
    from uagents_plugin import UAgentsPlugin
    # Your AI agent/workflow/engine
    agent_ai = create_your_ai_agent()
    plugin = UAgentsPlugin(agent_ai)
    agent.include(plugin.protocol)
    plugin.run(agent)
"""

from .adapter import UAgentsPlugin

__all__ = ["UAgentsPlugin"]
