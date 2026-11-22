# Records the README demo as a GIF with VHS (https://github.com/charmbracelet/vhs):
#
#   vhs docs/assets/demo.tape
#
# The README uses docs/assets/demo.svg (an animated SVG of the same three commands, from real output);
# this tape is for when a GIF/MP4 is handier, e.g. for social posts.

Output docs/assets/demo.gif

Set Shell "bash"
Set FontSize 14
Set Width 1300
Set Height 800
Set Padding 20
Set Theme "GitHub Dark"
Set TypingSpeed 35ms

Hide
Type "cd $(mktemp -d) && clear"
Enter
Show

Type "uvx omni-local-llm fit unsloth/Qwen3-8B-GGUF --context 8192"
Enter
Sleep 9s

Type "uvx omni-local-llm fit bartowski/SmolLM2-135M-Instruct-GGUF --quant Q4_K_M --pull --save-alias smol"
Enter
Sleep 9s

Type `uvx --from "omni-local-llm[llama-cpp]" omnillm run --model smol "What is the capital of France? Answer in one word."`
Enter
Sleep 8s
