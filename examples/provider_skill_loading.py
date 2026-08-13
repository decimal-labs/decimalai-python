"""DecimalAI Provider-Specific Skill Loading Example

Demonstrates the ``install(enable_skill_loader=True)`` option across
different LLM provider integrations, and how the ``load_skill`` tool is
controlled: it is NOT an ``install()`` flag on the tool-loop adapters —
it registers automatically with the loader and is config-driven via
``decimalai.init(load_skill_tool=...)`` / ``DECIMALAI_LOAD_SKILL_TOOL``.

This example shows:
- Anthropic SDK with skill injection into system prompts (no tool loop)
- OpenAI Agents SDK with the live load_skill tool
- Pydantic AI with automatic skill injection + load_skill tool

Prerequisites:
    pip install "decimalai[openai-agents,pydantic-ai]" anthropic

Environment variables:
    DECIMAL_API_KEY=dai_sk_...  # Get at https://app.decimal.ai/settings
    ANTHROPIC_API_KEY=sk-ant-...
    OPENAI_API_KEY=sk-...
"""

import asyncio
import os


def example_anthropic_skill_loader():
    """Anthropic SDK example with skill injection into system prompts.

    The enable_skill_loader=True option monkey-patches client.messages.create()
    so skills are automatically injected into the system prompt before each request.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Anthropic SDK with Skill Injection")
    print("="*70)

    import decimalai
    decimalai.init(verify=False)  # Skip backend verify for demo

    try:
        import anthropic

        from decimalai.anthropic import install, skill_system

        # Install skill loader — skills will auto-inject into system prompts
        install(enable_skill_loader=True)

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        # Every client.messages.create() now has skills injected automatically.
        # Internally, this monkey-patches the create method.

        # You can also manually inject skills using skill_system():
        system_with_skills = skill_system(
            base="You are a helpful customer support assistant.",
            query="How do I reset my password?",  # Optional: for semantic routing
        )

        print("✅ Anthropic skill loader installed")
        print(f"Sample system prompt with skills:\n{system_with_skills[:200]}...\n")

    except ImportError as e:
        print(f"⚠️  Anthropic example skipped: {e}")


def example_openai_agents_skill_loading():
    """OpenAI Agents SDK example with live load_skill tool.

    The enable_skill_loader=True option:
    1. Wraps Agent.instructions to inject skills into prompts
    2. Registers a load_skill tool so the model can fetch skill bodies mid-turn

    Since OpenAI Agents owns the tool loop, skill results route back
    automatically for the model to act on.

    Note: install() here takes no enable_load_skill_tool parameter. The
    tool registers automatically whenever the loader is enabled; the
    kill switch is decimalai.init(load_skill_tool=False) or the env var
    DECIMALAI_LOAD_SKILL_TOOL=0.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: OpenAI Agents SDK with Skill Tool")
    print("="*70)

    import decimalai
    decimalai.init(verify=False)

    try:
        from agents import Agent, function_tool

        from decimalai.openai_agents import install

        # Install skill loader — the load_skill tool registers automatically
        # (config-driven: init(load_skill_tool=) / DECIMALAI_LOAD_SKILL_TOOL,
        # on by default; NOT an install() parameter on this adapter)
        install(enable_skill_loader=True)

        @function_tool
        def check_account_status(customer_id: str) -> str:
            """Check the status of a customer account."""
            return f"Account {customer_id}: Active, good standing"

        agent = Agent(
            name="support-agent",
            instructions="You are a helpful customer support agent. Use tools to help.",
            tools=[check_account_status],
            model="gpt-4o",
        )

        print("✅ OpenAI Agents skill loader installed")
        print(f"Agent '{agent.name}' is ready with:")
        print("  - Skills auto-injected into instructions")
        print("  - load_skill tool available for fetching skill bodies\n")

    except ImportError as e:
        print(f"⚠️  OpenAI Agents example skipped: {e}")


async def example_pydantic_ai_skill_loading():
    """Pydantic AI example with automatic skill injection and load_skill tool.

    The enable_skill_loader=True option:
    1. Registers a system prompt function that injects skills
    2. Registers a load_skill tool on each Agent (automatic — controlled by
       init(load_skill_tool=) / DECIMALAI_LOAD_SKILL_TOOL, not an install() flag)

    Skills are injected before every agent.run() call, and the model can
    call load_skill to fetch full skill bodies.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Pydantic AI with Automatic Skill Injection")
    print("="*70)

    import decimalai
    decimalai.init(verify=False)

    try:
        from pydantic_ai import Agent

        from decimalai.pydantic_ai import install

        # Install skill loader — this patches Agent.__init__
        install(enable_skill_loader=True)

        # Every new Agent will have:
        # 1. A system_prompt function that injects skills
        # 2. A load_skill tool for fetching skill bodies

        agent = Agent(
            model="anthropic:claude-opus-4-7",
            system_prompt="You are a helpful assistant.",
        )

        print("✅ Pydantic AI skill loader installed")
        print("Every new Agent will have:")
        print("  - Skills auto-injected into system_prompt")
        print("  - load_skill tool for fetching skill bodies\n")

    except ImportError as e:
        print(f"⚠️  Pydantic AI example skipped: {e}")


def example_manual_skill_system():
    """Advanced: Manually inject skills without auto-loader.

    If you prefer more control, you can manually call skill_system()
    to build skill-injected prompts.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Manual Skill Injection (Advanced)")
    print("="*70)

    import decimalai
    decimalai.init(verify=False)

    try:
        from decimalai.anthropic import skill_system

        base_prompt = "You are a helpful assistant. Answer user questions clearly."
        user_query = "How do I report a bug?"

        # Manually build system prompt with skills
        system_with_skills = skill_system(
            base=base_prompt,
            query=user_query,  # Skills routed based on user input
            agent_name="support-agent",
        )

        print("✅ Manual skill injection example")
        print(f"Base prompt: {base_prompt}")
        print(f"User query: {user_query}")
        print("\nResulting system prompt (first 300 chars):")
        print(f"{system_with_skills[:300]}...\n")

    except ImportError as e:
        print(f"⚠️  Manual skill injection example skipped: {e}")


def main():
    """Run all provider skill-loading examples."""

    print("\n" + "="*70)
    print("DecimalAI Provider-Specific Skill Loading Examples")
    print("="*70)
    print("\nThis script demonstrates the provider skill-loading options:")
    print("  - install(enable_skill_loader=True): Auto-inject skills into prompts")
    print("  - load_skill tool: auto-registered with the loader on adapters that")
    print("    own a tool loop (openai_agents, pydantic_ai); config-driven via")
    print("    init(load_skill_tool=) / DECIMALAI_LOAD_SKILL_TOOL")

    # Example 1: Anthropic
    try:
        example_anthropic_skill_loader()
    except Exception as e:
        print(f"Anthropic example error: {e}")

    # Example 2: OpenAI Agents
    try:
        example_openai_agents_skill_loading()
    except Exception as e:
        print(f"OpenAI Agents example error: {e}")

    # Example 3: Pydantic AI
    try:
        asyncio.run(example_pydantic_ai_skill_loading())
    except Exception as e:
        print(f"Pydantic AI example error: {e}")

    # Example 4: Manual injection
    try:
        example_manual_skill_system()
    except Exception as e:
        print(f"Manual injection example error: {e}")

    print("\n" + "="*70)
    print("Next Steps")
    print("="*70)
    print("""
✅ All skill loaders installed and configured!

Provider-specific options summary:

1. ANTHROPIC SDK (decimalai.anthropic)
   - install(enable_skill_loader=True) → skills injected into system prompts
   - load_skill tool: NOT available. install() accepts an
     enable_load_skill_tool parameter but it is dormant — there is no tool
     loop to route the result back, so passing True only logs a warning and
     stays on prompt injection.

2. OPENAI AGENTS (decimalai.openai_agents)
   - install(enable_skill_loader=True) → skills injected into instructions
     AND the load_skill tool registers automatically (the agent owns its
     tool loop). install() takes NO enable_load_skill_tool parameter here:
     the tool is on by default with the loader; disable it globally with
     decimalai.init(load_skill_tool=False) or DECIMALAI_LOAD_SKILL_TOOL=0.

3. PYDANTIC AI (decimalai.pydantic_ai)
   - install(enable_skill_loader=True) → skills injected into system_prompt
     AND the load_skill tool registers automatically (same config switch,
     also not an install() parameter).

For more information:
  📖 DecimalAI Skills: https://docs.decimal.ai/guides/skills
  🏠 Dashboard: https://app.decimal.ai/skills
""")


if __name__ == "__main__":
    main()
