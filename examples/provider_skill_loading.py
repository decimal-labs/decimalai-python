"""DecimalAI Provider-Specific Skill Loading Example

Demonstrates how each provider integration gives an agent its DecimalAI
skills: ``agent_config()`` on the Anthropic Messages API, the OpenAI Agents SDK
and Pydantic AI. Their ``load_skill`` tool is not a ``agent_config()`` argument:
it is config-driven via ``decimalai.init(load_skill_tool=...)`` /
``DECIMALAI_LOAD_SKILL_TOOL``.

This example shows:
- Anthropic Messages API: one run per user message, handed to the SDK's Tool Runner
- OpenAI Agents SDK with the live load_skill tool
- Pydantic AI with agent_config()'s capability (skill list + load_skill tool)
- Anthropic again, in a hand-written client.messages.create() loop

The two Anthropic examples call the model only when ANTHROPIC_API_KEY is set.

Prerequisites:
    pip install "decimalai[anthropic,openai-agents,pydantic-ai]"

Environment variables:
    DECIMAL_API_KEY=dai_sk_...  # Get at https://app.decimal.ai/settings
    ANTHROPIC_API_KEY=sk-ant-...
    OPENAI_API_KEY=sk-...
"""

import asyncio
import os


MODEL = "claude-haiku-5-5"


def _text(message) -> str:
    """The text blocks of an Anthropic message, joined."""
    return "".join(block.text for block in message.content if block.type == "text")


def example_anthropic_tool_runner():
    """Anthropic Messages API: the agent's prompt and skills, one run per message.

    Nothing in the Anthropic SDK is installed or patched. agent_config() reads the
    agent, and each user message is a run whose two pieces you put into your own
    request:
    1. ``run.system`` — the agent's saved prompt, then the skill list routed on
       this message
    2. ``run.tools`` — the load_skill tool the model fetches a skill's body with

    The SDK's Tool Runner owns the tool loop, so a load_skill result goes back to
    the model mid-turn. support.run() calls no model: the model calls go inside
    the ``with`` block, which is also the run's trace.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Anthropic Messages API with agent_config()")
    print("="*70)

    import decimalai
    decimalai.init(verify=False)  # Skip backend verify for demo

    try:
        import anthropic
        from anthropic import beta_tool

        from decimalai.anthropic import agent_config

        @beta_tool
        def check_account_status(customer_id: str) -> str:
            """Check the status of a customer account."""
            return f"Account {customer_id}: Active, good standing"

        support = agent_config(
            "support-agent",
            fallback="You are a helpful customer support agent. Use tools to help.",
        )
        print("✅ Anthropic agent configured from DecimalAI")
        print(f"  - prompt version {support.prompt_version} and skills {support.skill_names}")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("  - set ANTHROPIC_API_KEY to send one message through the Tool Runner\n")
            return

        client = anthropic.Anthropic()
        question = "How do I reset my password?"
        with support.run(question) as run:
            runner = client.beta.messages.tool_runner(
                model=MODEL,
                max_tokens=1024,
                system=run.system,                          # saved prompt, then skill list
                tools=[check_account_status, *run.tools],   # your tools, then load_skill
                messages=[{"role": "user", "content": question}],
            )
            answer = runner.until_done()                    # consume it inside the run
        print(f"Answer: {_text(answer)[:300]}\n")

    except ImportError as e:
        print(f"⚠️  Anthropic example skipped: {e}")
    except Exception as e:  # e.g. no such agent in this workspace
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


def example_anthropic_messages_create_loop():
    """Advanced: Anthropic in a client.messages.create() loop you write yourself.

    The run gives the same load_skill tool as a plain definition
    (``run.tool_params``) and answers the model's load_skill calls for you:
    ``run.tool_result(block)`` returns the tool_result for a load_skill block and
    None for any other tool, which your own code answers.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Anthropic in a hand-written messages.create() loop (Advanced)")
    print("="*70)

    import decimalai
    decimalai.init(verify=False)

    try:
        import anthropic

        from decimalai.anthropic import agent_config

        report_bug = {
            "name": "report_bug",
            "description": "File a bug report and return its ticket id.",
            "input_schema": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        }
        support = agent_config(
            "support-agent",
            fallback="You are a helpful assistant. Answer user questions clearly.",
        )
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("  - set ANTHROPIC_API_KEY to run the loop\n")
            return

        client = anthropic.Anthropic()
        question = "How do I report a bug?"
        with support.run(question) as run:
            messages = [{"role": "user", "content": question}]
            while True:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    system=run.system,
                    tools=[report_bug, *run.tool_params],
                    messages=messages,
                )
                messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason != "tool_use":
                    break
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    result = run.tool_result(block)  # load_skill: answered by DecimalAI
                    if result is None:               # one of your tools
                        result = {"type": "tool_result", "tool_use_id": block.id,
                                  "content": "Filed as BUG-1042."}
                    results.append(result)
                messages.append({"role": "user", "content": results})

        print("✅ Hand-written loop finished")
        print(f"Answer: {_text(response)[:300]}\n")

    except ImportError as e:
        print(f"⚠️  messages.create() loop example skipped: {e}")
    except Exception as e:  # e.g. no such agent in this workspace
        print(f"⚠️  messages.create() loop example skipped: {e}")


def main():
    """Run all provider skill-loading examples."""

    print("\n" + "="*70)
    print("DecimalAI Provider-Specific Skill Loading Examples")
    print("="*70)
    print("\nThis script demonstrates the provider skill-loading options:")
    print("  - Anthropic Messages API: decimalai.anthropic.agent_config(name), a run per message")
    print("  - OpenAI Agents SDK: decimalai.openai_agents.agent_config(name)")
    print("  - Pydantic AI: decimalai.pydantic_ai.agent_config(name), a capability")
    print("  - load_skill tool on all three: config-driven via init(load_skill_tool=) /")
    print("    DECIMALAI_LOAD_SKILL_TOOL")

    # Example 1: Anthropic
    try:
        example_anthropic_tool_runner()
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

    # Example 4: Anthropic, hand-written loop
    try:
        example_anthropic_messages_create_loop()
    except Exception as e:
        print(f"messages.create() loop example error: {e}")

    print("\n" + "="*70)
    print("Next Steps")
    print("="*70)
    print("""
✅ All skill loaders installed and configured!

Provider-specific options summary:

1. ANTHROPIC MESSAGES API (decimalai.anthropic)
   - agent_config(name), then `with support.run(question) as run:` for each user
     message → pass run.system and [*your_tools, *run.tools] to the Tool Runner,
     or run.tool_params and run.tool_result(block) to your own messages.create()
     loop. A chat keeps one support.conversation(), whose system and tools stay
     the same for every message. Nothing is installed or patched in the SDK.

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
