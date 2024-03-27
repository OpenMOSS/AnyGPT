# modified from fastchat https://github.com/lm-sys/FastChat
"""
Conversation prompt templates.

We kindly request that you import fastchat instead of copying this file if you wish to use it.
If you have any changes in mind, please contribute back so the community can benefit collectively and continue to maintain these valuable templates.
"""

import dataclasses
from enum import auto, IntEnum
from typing import List, Any, Dict, Union, Tuple
from .anything2token import start_of_music, start_of_image, start_of_speech

class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: Tuple[str] = ("USER", "ASSISTANT")
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self, force_image_generation=False, force_speech_generation=False, force_music_generation=False, force_res_prefix=None) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        ret = system_prompt + "\n"
        for i in range(len(self.messages)):
            role, message = self.messages[i]
            if message and i % 2 == 0:
                ret += role + ": " + message + self.sep + "\n"
            elif message and i % 2 == 1:
                ret += role + ": " + message + self.sep2
                if i != len(self.messages) - 1:
                    ret += "\n"
            else:
                ret += role + ":"
        if self.messages[-1][0] == self.roles[0]:
            ret += self.roles[1] + ":"
            if force_res_prefix != None:
                ret += " "+force_res_prefix
            if force_image_generation:
                ret+=" "+start_of_image
            elif force_speech_generation:
                ret+=" "+start_of_speech
            elif force_music_generation:
                ret+=" "+start_of_music
        return ret
    
    def get_history(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        ret = system_prompt + "\n"
        for i in range(len(self.messages)):
            role, message = self.messages[i]
            if message and i % 2 == 0:
                ret += role + ": " + message + self.sep
                if i != len(self.messages) - 1:
                    ret += "\n"
            elif message and i % 2 == 1:
                ret += role + ": " + message + self.sep2
                if i != len(self.messages) - 1:
                    ret += "\n"
            else:
                ret += role + ":"
        return ret
    
    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# An empty template for raw conversation.
register_conv_template(
    Conversation(
        name="raw",
        system_message="",
        roles=("", ""),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
    )
)


user_name = "[Human]"
chatbot_name = "[AnyGPT]"
user_end = "<eoh>"
chatbot_end = "<eom>"
eos_token = "<eos>"
anygpt_system_prompt = "You are an AI assistant named AnyGPT who can understand and generate multimodal content, including text, speech, images and audio."

# Our template
register_conv_template(
    Conversation(
        name="AnyGPT",
        system_message=anygpt_system_prompt,
        roles=(user_name, chatbot_name),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=user_end,
        sep2=chatbot_end
    )
)


# It's just because the name MMGPT was used for model training in the early stages of research.
mmgpt_system_prompt = "You are an AI assistant named MMGPT who can understand and generate multimodal content, including text, speech, images and audio."
register_conv_template(
    Conversation(
        name="MMGPT",
        system_message="You are an AI assistant named MMGPT who can understand and generate multimodal content, including text, speech, images and audio.",
        roles=(user_name, "[MMGPT]"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=user_end,
        sep2=eos_token
    )
)


if __name__ == "__main__":    
    print("-- Our template --")
    conv = get_conv_template("AnyGPT")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    # conv.append_message(conv.roles[1], "I'm fine, thank you.")
    print(conv.get_prompt())
    print(conv.get_prompt(force_image_generation=True))
    # print(conv.get_history())
    
    conv = get_conv_template("MMGPT")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    # conv.append_message(conv.roles[1], "I'm fine, thank you.")
    print(conv.get_prompt())
    # print(conv.get_history())