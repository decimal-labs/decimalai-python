"""DecimalAI Provider-Specific Skill Loading Example

Demonstrates how each provider integration gives an agent its DecimalAI
skills: ``instrument(enable_skill_loader=True)`` on the Anthropic SDK, and
``agent_config()`` on the OpenAI Agents SDK and Pydantic AI. Their ``load_skill``
tool is not an ``agent_config()`` argument: it is config-driven via
``decimalai.init(load_skill_tool=...)`` / ``DECIMALAI_LOAD_SKILL_TOOL``.

This example shows:
- Anthropic SDK with skill injection into system prompts (no tool loop)
- OpenAI Agents SDK with the live load_skill tool
- Pydantic AI with agent_config()'s capability (skill list + load_skill tool)

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

        from decimalai.anthropic import instrument, skill_system

        # Turn on the skill loader — skills will auto-inject into system prompts
        instrument(enable_skill_loader=True)

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
    """OpenAI Agents SDK: the agent's prompt and skills from agent_config().

    Since 0.18 nothing is installed or patched for this framework. agent_config()
    returns the two pieces you put into your own Agent(...):
    1. ``instructions`` — the agent's saved prompt, then its skill list
    2. ``tools`` — the load_skill tool the model fetches a skill's body with

    The OpenAI Agents SDK owns the tool loop, so a load_skill result goes back to
    the model mid-turn.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: OpenAI Agents SDK with agent_config()")
    print("="*70)

    import decimalai
    decimalai.init(verify=False)

    try:
        from agents import Agent, function_tool

        from decimalai.openai_agents import agent_config

        @function_tool
        def check_account_status(customer_id: str) -> str:
            """Check the status of a customer account."""
            return f"Account {customer_id}: Active, good standing"

        support = agent_config(
            "support-agent",
            fallback="You are a helpful customer support agent. Use tools to help.",
        )
        agent = Agent(
            name=support.name,
            instructions=support.instructions,
            tools=[check_account_status, *support.tools],
            model="gpt-4o",
        )

        print("✅ OpenAI Agents agent configured from DecimalAI")
        print(f"Agent '{agent.name}' is ready with:")
        print(f"  - prompt version {support.prompt_version} and skills {support.skill_names}")
        print("  - load_skill tool available for fetching skill bodies\n")

    except ImportError as e:
        print(f"⚠️  OpenAI Agents example skipped: {e}")
    except Exception as e:  # e.g. no such agent in this workspace
        print(f"⚠️  OpenAI Agents example skipped: {e}")


async def example_pydantic_ai_skill_loading():
    """Pydantic AI example with agent_config()'s capability.

    ``decimalai.pydantic_ai.agent_config(name)`` returns a Pydantic AI capability
    you pass in your own ``Agent(capabilities=[...])``:
    1. its instructions — the agent's saved prompt, then the skill list picked
       for each run's prompt
    2. its toolset — the load_skill tool the model fetches a skill's body with

    Pydantic AI owns the tool loop, so a load_skill result goes back to the
    model mid-run. Nothing in Pydantic AI is patched.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Pydantic AI with agent_config()")
    print("="*70)

    import decimalai
    decimalai.init(verify=False)

    try:
        from pydantic_ai import Agent

        from decimalai.pydantic_ai import agent_config

        def check_account_status(customer_id: str) -> str:
            """Check the status of a customer account."""
            return f"Account {customer_id}: Active, good standing"

        support = agent_config(
            "support-agent",
            fallback="You are a helpful customer support agent. Use tools to help.",
        )
        agent = Agent(
            "anthropic:claude-opus-4-7",
            name=support.name,
            tools=[check_account_status],
            capabilities=[support],
        )

        print("✅ Pydantic AI agent configured from DecimalAI")
        print(f"Agent '{agent.name}' is ready with:")
        print(f"  - prompt version {support.prompt_version} and skills {support.skill_names}")
        print("  - load_skill tool available for fetching skill bodies\n")

    except ImportError as e:
        print(f"⚠️  Pydantic AI example skipped: {e}")
    except Exception as e:  # e.g. no such agent in this workspace
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
    print("  - instrument(enable_skill_loader=True): Auto-inject skills into prompts")
    print("  - OpenAI Agents SDK: decimalai.openai_agents.agent_config(name)")
    print("  - Pydantic AI: decimalai.pydantic_ai.agent_config(name), a capability")
    print("  - load_skill tool on both: config-driven via init(load_skill_tool=) /")
    print("    DECIMALAI_LOAD_SKILL_TOOL")

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
   - instrument(enable_skill_loader=True) → skills injected into system prompts
   - load_skill tool: NOT available. instrument() accepts an
     enable_load_skill_tool parameter but it is dormant — there is no tool
     loop to route the result back, so passing True only logs a warning and
     stays on prompt injection.

2. OPENAI AGENTS (decimalai.openai_agents)
   - agent_config(name) → pass .instructions and .tools into your own Agent(...)
     (the agent owns its tool loop, so load_skill results go back to the model).
     Nothing is installed or patched for this framework.

3. PYDANTIC AI (decimalai.pydantic_ai)
   - agent_config(name) → pass it in your own Agent(capabilities=[...]): its
     instructions carry the prompt and skill list, its toolset load_skill.
     Nothing is installed or patched for this framework.

For more information:
  📖 DecimalAI Skills: https://docs.decimal.ai/guides/skills
  🏠 Dashboard: https://app.decimal.ai/skills
""")


if __name__ == "__main__":
    main()
