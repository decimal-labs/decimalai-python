# sdk-demo.tape — renders docs/assets/sdk-demo.gif with VHS (charmbracelet/vhs).
#
# To re-render, provide a prelude via the environment (evaluated off-screen —
# nothing typed on screen carries a key). See sdk-demo-prelude.example.sh next
# to this tape; it exports the API key and injects the seeded demo's candidate
# manifest id into the regression-check call:
#   DECIMALAI_TAPE_PRELUDE="source sdk-demo-prelude.example.sh" vhs sdk-demo.tape

Output sdk-demo.gif

Set Shell bash
Set FontSize 28
Set Width 1560
Set Height 1100
Set Padding 24
Set WindowBar Colorful
Set Theme "Dracula"
Set TypingSpeed 40ms

Hide
Type `eval "${DECIMALAI_TAPE_PRELUDE:-}"; PS1="> "; clear`
Enter
Show

Sleep 800ms
Type "# auth: export DECIMAL_API_KEY=dai_sk_...   (free key: app.decimal.ai/settings)"
Enter
Sleep 600ms
Type "decimalai demo regression"
Enter
Wait@30s
Sleep 2s
Type@25ms `decimalai regression-check --agent-name "[Demo] support-agent"`
Enter
Wait@30s
Sleep 5s
