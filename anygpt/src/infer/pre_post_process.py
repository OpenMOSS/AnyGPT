import torch
import json
import sys
sys.path.append("/mnt/petrelfs/zhanjun.p/mllm")
sys.path.append("/mnt/petrelfs/zhanjun.p/src")
from transformers import GenerationConfig
from anygpt.src.m_utils.prompter import Prompter
from tqdm import tqdm

from m_utils.conversation import get_conv_template
conversation = get_conv_template('MMGPT')
prompter = Prompter()


def extract_text_between_tags(text, tag1='[MMGPT] :', tag2='<eoa>'):
    try:
        # print(text)
        start = text.index(tag1) + len(tag1)
        end = text.index(tag2, start)
        extracted_text = text[start:end].strip()  # 抽取内容并去除前后的空格
        if not extracted_text:  # 如果抽取的内容为空
            try:
                extracted_text = text[start:]
            except:
                extracted_text = text
        return extracted_text
    except ValueError:  # 找不到tag2时，返回tag1之后的所有内容
        # print("extract tags error:")
        try:
            extracted_text = text[start:]
        except:
            extracted_text = text
        return extracted_text
    
def extract_content_between_final_tags(text, tag1='[MMGPT]', tag2='<eom>'):
    """
    Extracts the content between the last occurrence of tag1 and the last occurrence of tag2 in a given string.

    :param text: String containing the text with tags.
    :param tag1: The first tag to look for.
    :param tag2: The second tag to look for.
    :return: The content between the last occurrence of tag1 and tag2. Returns an empty string if tags are not found in order.
    """
    last_tag1 = text.rfind(tag1)
    last_tag2 = text.rfind(tag2)

    if last_tag1 == -1 or last_tag2 == -1 or last_tag1 > last_tag2:
        return None

    # Extracting the content between the two tags
    start = last_tag1 + len(tag1)
    end = last_tag2
    return text[start:end]



def preprocess(
        input_data,
        modality,
        to_modality,
        sft_template=False
    ):
        # processed_parts = []
        
        if modality == "text":
            processed_inputs = input_data
        elif to_modality == "text":
            # processed_inputs = []
            # for image_path in input_data:
            #     processed_inputs.append()
            processed_inputs = input_data
        else:
            raise TypeError("wrong modality")
        if sft_template:
            prompt_seq_list = []
            for content in processed_inputs:
                prompt = prompter.generate_insturction_prompt(task="Text-to-Image Generation", instruction=content) 
                conversation.append_message(conversation.roles[0], prompt)
                prompt_seq_list.append(conversation.get_prompt())
                conversation.messages=[]
        else:
            prompt_seq_list = [prompter.generate_prompt_input(modality_str=content, modality=modality,
                                                            to_modality=to_modality) for content in processed_inputs]
        return prompt_seq_list
    

def response(tokenizer, model, modality, to_modality, input_data, device, voice_prompt=None, config=None, sft_template=False):
    print("preroceessing...")
    preprocessed_prompts = preprocess(input_data=input_data, modality=modality, to_modality=to_modality, sft_template=sft_template)
    print(preprocessed_prompts)
    input_ids = tokenizer(preprocessed_prompts, return_tensors="pt", padding=True).input_ids
    input_ids = input_ids.to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            generation_config=config,
            return_dict_in_generate=True,
            output_scores=True,
        )
    generated_ids = generated_ids.sequences
    responses = tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)
    print(responses)
    response_list = [extract_text_between_tags(response, tag1="[MMGPT]", tag2="<eos>").strip().strip(":") .strip()
                            for response in responses]
    return response_list


def inference(tokenizer, model, modality, to_modality, input_data, device=None, batch_size=32, config=None,sft_template=False):
    if device == None:
        device = next(model.parameters()).device
    response_list = []
    for i in tqdm(range(0, len(input_data), batch_size)):
        batch_data = input_data[i:i+batch_size] 
        print(batch_data)   
        batch_response = response(tokenizer=tokenizer, model=model, modality=modality, to_modality=to_modality, 
                                 input_data=batch_data, device=device, config=config, sft_template=sft_template)
        response_list.extend(batch_response)
        # print("batch {} done".format(i))
    return response_list

def get_prompt(task, input_images, instructions, question_type_id, text_answer=None):
    prompt_seq_list = []
    # assert len(input_images) == len(instructions)
    for i in range(len(instructions)):
        prompt = prompter.generate_insturction_prompt(task=task, question_type_id=question_type_id, instruction=instructions[i], image_list=input_images[i]) 
        conversation.append_message(conversation.roles[0], prompt)
        if question_type_id == 26:
            prompt = conversation.get_prompt(force_image_generation=True)
        elif question_type_id == 27:
            prompt = conversation.get_prompt(force_image_generation=True, force_res_prefix=text_answer[i])
        prompt_seq_list.append(prompt)
        conversation.messages=[]
    return prompt_seq_list

def inference_seedbench(tokenizer, model, task, question_type_id, input_data, device=None, batch_size=32, config=None):
    if device == None:
        device = next(model.parameters()).device
    response_list = []
    for i in tqdm(range(0, len(input_data), batch_size)):
        batch_data = input_data[i:i+batch_size] 
        # print(batch_data)  
        input_images = [item['images'] for item in batch_data]
        instructions = [item['instructions'] for item in batch_data]
        text_answer = [item['text_answer'] for item in batch_data]
        preprocessed_prompts = get_prompt(task, input_images, instructions, question_type_id, text_answer)
        print(preprocessed_prompts)
        input_ids = tokenizer(preprocessed_prompts, return_tensors="pt", padding=True).input_ids
        input_ids = input_ids.to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                generation_config=config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        generated_ids = generated_ids.sequences
        responses = tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)
        
        batch_response = [extract_text_between_tags(response, tag1="[MMGPT]", tag2="<eos>").strip() 
                            for response in responses]
        response_list.extend(batch_response)
        print("batch {} done".format(i))
    return response_list
    
    
# save responses
def save_responses(response_list, output_path):
    with open(output_path, 'a') as f:
        for response in response_list:
            f.write(response + '\n')
            
from cleantext import clean
import re
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
inverse_normalizer = InverseNormalizer(lang='en')

def text_normalization(original_text):
    text= clean(original_text,
        fix_unicode=True,               # fix various unicode errors
        to_ascii=True,                  # transliterate to closest ASCII representation
        lower=True,                     # lowercase text
        no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
        no_urls=False,                  # replace all URLs with a special token
        no_emails=False,                # replace all email addresses with a special token
        no_phone_numbers=False,         # replace all phone numbers with a special token
        no_numbers=False,               # replace all numbers with a special token
        no_digits=False,                # replace all digits with a special token
        no_currency_symbols=False,      # replace all currency symbols with a special token
        no_punct=False,                 # remove punctuations
        replace_with_punct="",          # instead of removing punctuations you may replace them
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_number="<NUMBER>",
        replace_with_digit="0",
        replace_with_currency_symbol="<CUR>",
        lang="en"                       # set to 'de' for German special handling
    )
    text=inverse_normalizer.inverse_normalize(text, verbose=False)
    text=text.lower()
    # A dictionary of contractions and their expanded forms, including "didn't"
    contractions = {
        "i'm": "i am", "don't": "do not", "can't": "cannot", "it's": "it is",
        "isn't": "is not", "he's": "he is", "she's": "she is", "that's": "that is",
        "what's": "what is", "where's": "where is", "there's": "there is",
        "who's": "who is", "how's": "how is", "i've": "i have", "you've": "you have",
        "we've": "we have", "they've": "they have", "i'd": "i would", "you'd": "you would",
        "he'd": "he would", "she'd": "she would", "we'd": "we would", "they'd": "they would",
        "i'll": "i will", "you'll": "you will", "he'll": "he will", "she'll": "she will",
        "we'll": "we will", "they'll": "they will", "didn't": "did not"
    }

    # Manually handle contractions
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    # Remaining rules are the same as previous implementation
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    fillers = ["hmm", "mm", "mhm", "mmm", "uh", "um"]
    filler_pattern = r'\b(?:' + '|'.join(fillers) + r')\b'
    text = re.sub(filler_pattern, "", text)
    text = re.sub(r"\s’", "’", text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"\.(?!\d)", "", text)
    text = re.sub(r"[^\w\s.,%$]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


if __name__ == "__main__":
    # Example usage
    example_text = "Hello [MMGPT] this is some text [eom] and more text [MMGPT] here is the content [eom] end"
    example_text = "Hello [MMGPT] this is some text"
    # example_text = "Hello this is some text"
    tag1 = '[MMGPT]'
    tag2 = '[eom]'
    print(extract_content_between_final_tags(example_text, tag1, tag2))
    print(extract_text_between_tags(example_text, tag1, tag2))