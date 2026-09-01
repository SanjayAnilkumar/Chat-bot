# ============================================================
# GPT-OSS 20B NORMAL CHATBOT
# GROQ API + GRADIO 6
# GOOGLE COLAB
# ============================================================

!pip -q install -U groq gradio


import gradio as gr
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """
You are a helpful, intelligent and friendly AI assistant.

Rules:
- Answer questions accurately.
- Explain difficult topics clearly.
- Use examples when useful.
- Do not make up information.
- Remember the conversation context.
- Be concise unless the user asks for detailed information.
"""


# ============================================================
# CHAT FUNCTION
# ============================================================

def chat(message, history, api_key, temperature, reasoning):

    # Make sure history exists
    if history is None:
        history = []

    # Check API key
    if not api_key or not api_key.strip():

        history = history + [
            {
                "role": "user",
                "content": message
            },
            {
                "role": "assistant",
                "content": "⚠️ Please enter your Groq API key."
            }
        ]

        return history, ""

    try:

        # ------------------------------------------------------
        # GROQ CLIENT
        # ------------------------------------------------------

        client = Groq(
            api_key=api_key.strip()
        )

        # ------------------------------------------------------
        # BUILD MESSAGES FOR GROQ
        # ------------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        # ------------------------------------------------------
        # CONVERT GRADIO HISTORY
        # ------------------------------------------------------

        for item in history:

            # Gradio message dictionary
            if isinstance(item, dict):

                role = item.get("role")
                content = item.get("content")

                if role in ["user", "assistant"]:

                    if isinstance(content, str):

                        messages.append({
                            "role": role,
                            "content": content
                        })

            # Handle ChatMessage objects if present
            elif hasattr(item, "role") and hasattr(item, "content"):

                role = item.role
                content = item.content

                if role in ["user", "assistant"]:

                    if isinstance(content, str):

                        messages.append({
                            "role": role,
                            "content": content
                        })

        # ------------------------------------------------------
        # CURRENT USER MESSAGE
        # ------------------------------------------------------

        messages.append({
            "role": "user",
            "content": message
        })

        # ------------------------------------------------------
        # CALL GROQ
        # ------------------------------------------------------

        response = client.chat.completions.create(

            model=MODEL,

            messages=messages,

            temperature=float(temperature),

            reasoning_effort=reasoning,

            max_completion_tokens=4096
        )

        # ------------------------------------------------------
        # GET RESPONSE
        # ------------------------------------------------------

        answer = response.choices[0].message.content

        if not answer:

            answer = "I couldn't generate a response."

        # ------------------------------------------------------
        # ADD BOTH MESSAGES TO HISTORY
        # ------------------------------------------------------

        new_history = history + [

            {
                "role": "user",
                "content": message
            },

            {
                "role": "assistant",
                "content": answer
            }

        ]

        # IMPORTANT:
        # Return the COMPLETE history.
        # This is required for Gradio 6.
        return new_history, ""

    except Exception as e:

        error = (
            "❌ Groq API Error\n\n"
            + str(e)
        )

        new_history = history + [

            {
                "role": "user",
                "content": message
            },

            {
                "role": "assistant",
                "content": error
            }

        ]

        return new_history, ""


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
    # HEADER
    # --------------------------------------------------------

    gr.Markdown(
        """
        # 🤖 GPT-OSS 20B Chatbot

        ### ⚡ Powered by Groq
        """
    )

    # --------------------------------------------------------
    # MAIN LAYOUT
    # --------------------------------------------------------

    with gr.Row():

        # ====================================================
        # CHAT
        # ====================================================

        with gr.Column(
            scale=4
        ):

            chatbot = gr.Chatbot(
                label="AI Assistant",
                height=550
            )

            message = gr.Textbox(
                label="Message",
                placeholder="Type your message...",
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

                Normal AI Chatbot

                ### 🧠 Memory

                Conversation History
                """
            )

    # ========================================================
    # SEND
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
        outputs=[
            chatbot,
            message
        ]
    )

    # ========================================================
    # ENTER
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
        outputs=[
            chatbot,
            message
        ]
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
print("=" * 65)
print("🚀 GPT-OSS 20B CHATBOT")
print("=" * 65)
print()
print("Your chatbot is starting...")
print()
print("Enter your Groq API key in the UI.")
print()

demo.launch(
    share=True
)
