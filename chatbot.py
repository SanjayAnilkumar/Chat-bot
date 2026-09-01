# ============================================================
# GPT-OSS 20B NORMAL CHATBOT
# GROQ API + GRADIO 6
# GOOGLE COLAB
# ============================================================

!pip -q install -U groq gradio


# ============================================================
# IMPORTS
# ============================================================

import gradio as gr
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """
You are a helpful, intelligent and friendly AI assistant.

Rules:
- Answer accurately.
- Explain difficult topics clearly.
- Use examples when useful.
- Do not invent facts.
- Maintain conversation context.
- Be concise unless the user asks for detail.
"""


# ============================================================
# CHAT FUNCTION
# ============================================================

def chat(
    message,
    history,
    api_key,
    temperature,
    reasoning
):

    # Check API key
    if not api_key or not api_key.strip():

        return {
            "role": "assistant",
            "content": "⚠️ Please enter your Groq API key."
        }

    try:

        # Create Groq client
        client = Groq(
            api_key=api_key.strip()
        )

        # ------------------------------------------------------
        # CREATE MESSAGES
        # ------------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        # ------------------------------------------------------
        # ADD GRADIO HISTORY
        # ------------------------------------------------------

        if history:

            for item in history:

                # Gradio 6 messages format
                if isinstance(item, dict):

                    role = item.get("role")
                    content = item.get("content")

                    # Only send user/assistant text
                    if role in ["user", "assistant"]:

                        # Handle normal string content
                        if isinstance(content, str):

                            messages.append({
                                "role": role,
                                "content": content
                            })

                        # Handle Gradio 6 structured content
                        elif isinstance(content, list):

                            text_parts = []

                            for block in content:

                                if isinstance(block, dict):

                                    if block.get("type") == "text":

                                        text_parts.append(
                                            block.get(
                                                "text",
                                                ""
                                            )
                                        )

                                elif isinstance(
                                    block,
                                    str
                                ):

                                    text_parts.append(
                                        block
                                    )

                            combined = "\n".join(
                                text_parts
                            )

                            if combined:

                                messages.append({
                                    "role": role,
                                    "content": combined
                                })

        # ------------------------------------------------------
        # CURRENT USER MESSAGE
        # ------------------------------------------------------

        messages.append({
            "role": "user",
            "content": message
        })

        # ------------------------------------------------------
        # GROQ API
        # ------------------------------------------------------

        response = client.chat.completions.create(

            model=MODEL,

            messages=messages,

            temperature=float(
                temperature
            ),

            reasoning_effort=reasoning,

            max_completion_tokens=4096
        )

        # ------------------------------------------------------
        # RESPONSE
        # ------------------------------------------------------

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        # Return ONLY the new assistant message.
        # Gradio 6 automatically adds it to history.
        return {
            "role": "assistant",
            "content": answer
        }

    except Exception as e:

        return {
            "role": "assistant",
            "content": (
                "❌ Groq API Error:\n\n"
                + str(e)
            )
        }


# ============================================================
# CLEAR CHAT
# ============================================================

def clear_chat():
    return []


# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title="GPT-OSS 20B Chatbot"
) as demo:

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    gr.Markdown(
        """
        # 🤖 GPT-OSS 20B Chatbot

        ### ⚡ Powered by Groq
        """
    )

    # --------------------------------------------------------
    # MAIN ROW
    # --------------------------------------------------------

    with gr.Row():

        # ====================================================
        # CHAT AREA
        # ====================================================

        with gr.Column(
            scale=4
        ):

            chatbot = gr.Chatbot(
                label="AI Assistant",
                height=550
            )

            # Input
            message = gr.Textbox(
                placeholder="Type your message...",
                label="Message",
                lines=2
            )

            with gr.Row():

                send = gr.Button(
                    "🚀 Send",
                    variant="primary"
                )

                clear = gr.Button(
                    "🗑️ Clear"
                )

        # ====================================================
        # SETTINGS
        # ====================================================

        with gr.Column(
            scale=1
        ):

            gr.Markdown(
                "## ⚙️ Settings"
            )

            api_key = gr.Textbox(
                label="Groq API Key",
                placeholder="gsk_...",
                type="password"
            )

            temperature = gr.Slider(
                minimum=0,
                maximum=1.5,
                value=0.7,
                step=0.1,
                label="Temperature"
            )

            reasoning = gr.Dropdown(
                choices=[
                    "low",
                    "medium",
                    "high"
                ],
                value="medium",
                label="Reasoning Effort"
            )

            gr.Markdown(
                """
                ### 🧠 Model

                `openai/gpt-oss-20b`

                ### ⚡ Provider

                Groq API

                ### 💬 Type

                Normal Chatbot

                ### 🧠 Memory

                Conversation History
                """
            )

    # ========================================================
    # SEND BUTTON
    # ========================================================

    send.click(
        chat,
        inputs=[
            message,
            chatbot,
            api_key,
            temperature,
            reasoning
        ],
        outputs=chatbot
    ).then(
        lambda: "",
        outputs=message
    )

    # ========================================================
    # ENTER KEY
    # ========================================================

    message.submit(
        chat,
        inputs=[
            message,
            chatbot,
            api_key,
            temperature,
            reasoning
        ],
        outputs=chatbot
    ).then(
        lambda: "",
        outputs=message
    )

    # ========================================================
    # CLEAR
    # ========================================================

    clear.click(
        clear_chat,
        outputs=chatbot
    )


# ============================================================
# LAUNCH
# ============================================================

print()
print("=" * 60)
print("🚀 GPT-OSS 20B CHATBOT")
print("=" * 60)
print()
print("Open the Gradio public URL below.")
print("Enter your Groq API key.")
print("Start chatting.")
print()

demo.launch(
    share=True
)
