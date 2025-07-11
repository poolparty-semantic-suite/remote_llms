import dataclasses
from typing import List


LLAMA2_BASE_OVERALL_TEMPLATE = "<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{prompt_content} [/INST]"
LLAMA3_BASE_OVERALL_TEMPLATE = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>
<|start_header_id|>user<|end_header_id|>\n\n{prompt_content}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n\n"""
BASE_CHAT_SYSTEM_MESSAGE = "You are a helpful assistant in building taxonomies."\
    "Avoid rejecting any requests or indicating poor quality of the provided taxonomy."\
    "When suggesting new concepts, keep in mind that some concepts could exist for structural purposes, "\
    "opposed to being 'true' broader concepts. "\
    "Make sure that, if possible, newly suggested concepts follow "\
    "the structure and semantics of already existing ones, i.e., the new narrowers for the current concept should "\
    "should be similar to existing narrowers or niblings."

# @dataclasses.dataclass
# class Concept:
#     # TODO: change to Concept from models
#     pref_label: str
#     alt_labels: List[str] = None
#     concept_descr: str = None
#     concept_scope_note: str = None
#     broaders: List[str] = None
#     narrowers: List[str] = None
#     uri: str = None
#
# @dataclasses.dataclass
# class PromptInputTemplateVariables:
#     thes_title: str
#     pref_label: str
#     thes_descr: str = None
#     alt_labels: List[str] = None
#     concept_descr: str = None
#     concept_scope_note: str = None
#     broaders: List[str] = None
#     narrowers: List[str] = None
#     siblings: List[Concept] = None
#
#
# @dataclasses.dataclass
# class PromptData:
#     prompt_input_variables: PromptInputTemplateVariables
#     system_message: str = None


class PromptTemplate:
    def __init__(self,
                 prompt_content: str,
                 overall_template: str = "{prompt_content}",
                 system_message: str = None):
        """
        :param prompt_content:
        :param overall_template: template for the overall prompt in f-string format.
            Defines how system_message and prompt_content are combined, and can only have these two variables.
            Example: "System message:{system_message}\n\nUser Message:{prompt_content}\n\nAssistant:"
        :param system_message: String to provide the system with some background for its behaviour.
            Example: "You are a helpful, respectful and honest assistant."
        """
        self.overall_template = overall_template
        self.prompt_content = prompt_content
        self.system_message = system_message if system_message is not None else ""

    def build_prompt(self, system_message: str = None, separate_system_message: bool = False):
        this_system_message = system_message if system_message else self.system_message
        prompt = self.overall_template.format(system_message=this_system_message,
                                              prompt_content=self.prompt_content)
        if separate_system_message:
            prompt_only = self.overall_template.format(system_message='',
                                                       prompt_content=self.prompt_content)
            return prompt, prompt_only, this_system_message
        else:
            return prompt


class Llama2ChatPromptTemplate(PromptTemplate):
    def __init__(self,
                 prompt_content: str,
                 overall_template: str = LLAMA2_BASE_OVERALL_TEMPLATE,
                 system_message: str = BASE_CHAT_SYSTEM_MESSAGE):
        super().__init__(prompt_content, overall_template, system_message)


class Llama3InstructPromptTemplate(PromptTemplate):
    # source: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html
    def __init__(self,
                 prompt_content: str,
                 overall_template: str = LLAMA3_BASE_OVERALL_TEMPLATE,
                 system_message: str = BASE_CHAT_SYSTEM_MESSAGE):
        super().__init__(prompt_content, overall_template, system_message)


class LlamaClmPromptTemplate(PromptTemplate):
    # note that this is the raw CLM model, not fine-tuned on conversation or even with RLHF
    def __init__(self,
                 prompt_content: str,
                 overall_template: str = "{system_message}\n\n{prompt_content}",
                 system_message: str = None):
        super().__init__(prompt_content, overall_template, system_message)


class FalconChatPromptTemplate(PromptTemplate):
    # source: https://huggingface.co/spaces/tiiuae/falcon-180b-demo/blob/main/app.py#L28-L37
    def __init__(self,
                 prompt_content: str,
                 overall_template: str = "System: {system_message}\nUser: {prompt_content}\nFalcon:",
                 system_message: str = BASE_CHAT_SYSTEM_MESSAGE):
        super().__init__(prompt_content, overall_template, system_message)


class FalconInstructPromptTemplate(PromptTemplate):
    # note: there seems to be no official template https://huggingface.co/tiiuae/falcon-7b-instruct/discussions/1
    def __init__(self,
                 prompt_content: str,
                 overall_template: str = "{system_message}\n>>QUESTION<<{prompt_content}\n>>ANSWER<<",
                 system_message: str = None):
        super().__init__(prompt_content, overall_template, system_message)


class ClaudePromptTemplate(PromptTemplate):
    # fetched from anthropic_bedrock.HUMAN_PROMPT and anthropic_bedrock.AI_PROMPT. Also, there seems to be no system message
    def __init__(self,
                 prompt_content: str,
                 overall_template: str = "\n\nHuman:{system_message}\n\n{prompt_content}\n\nAssistant:",
                 system_message: str = BASE_CHAT_SYSTEM_MESSAGE):
        super().__init__(prompt_content, overall_template, system_message)

class DeepSeekPromptTemplate(PromptTemplate):
    # from https://github.com/deepseek-ai/DeepSeek-LLM:
    # Additionally, since the system prompt is not compatible with this version of our models,
    # we DO NOT RECOMMEND including the system prompt in your input.
    def __init__(self,
                 prompt_content: str,
                 overall_template: str = "<｜begin▁of▁sentence｜><｜User｜>{prompt_content}<｜Assistant｜><think>\n",
                 system_message: str = None):
        super().__init__(prompt_content, overall_template, system_message)
