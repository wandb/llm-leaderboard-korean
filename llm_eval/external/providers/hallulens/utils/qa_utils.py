# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re

def split_doc(one_input, language, encoding, keep_end, keep_colon, MIN_LEN=None, MAX_LEN=None):
    if not (MIN_LEN and MAX_LEN):
        if language == "zh":
            MIN_LEN, MAX_LEN = 108, 270
        elif language == "en":
            MIN_LEN, MAX_LEN = 80, 200

    if len(encoding.encode(one_input)) <= MAX_LEN:
        return [one_input]
    
    if language == 'zh': # split by title firstly
        sections2 = re.split(r"(##+)|(\n(?:\d+|[一二三四五六七八九十⓪①②③④⑤⑥⑦⑧⑨⑩零壹贰叁肆伍陆柒捌玖拾])[、.：:].{0,6}\n+)", one_input) 
    elif language == 'en':
        sections2 = re.split(r"(##+)|(\n(?:\d+|[⓪①②③④⑤⑥⑦⑧⑨⑩])[、.：:].{0,15}\n+)", one_input) 
        
    if sections2[0].strip():
        sections = [sections2[0]]
    else:
        sections = []
    assert (len(sections2)-1) % 3 == 0
    for i in range(1, len(sections2), 3):
        if i+2<len(sections2):
            sent = ""
            if sections2[i]:
                sent += sections2[i]
            if sections2[i+1]:
                sent += sections2[i+1]
            if sections2[i+2]:
                sent += sections2[i+2]
        else:
            sent = sections2[i]
        if sent:
            sections.append(sent)
            
    one_output = []
    for section in sections:
        one_output += split_context(section, MAX_LEN, language, encoding, keep_end, keep_colon)
        
    one_output2 = []
    # last_p = re.sub("\s+", " ", one_output[0])
    last_p = one_output[0]
    for p in one_output[1:]:
        # p = re.sub("\s+", " ", p)
        if len(encoding.encode(p)) < MIN_LEN:
            if language == "zh":
                last_p = last_p+p
            else:
                last_p = last_p+" "+p
            # print("p", p)
        else:
            if last_p.strip():
                one_output2.append(last_p)
            last_p = p
    if last_p.strip():
        one_output2.append(last_p)
    # assert one_output2 == [re.sub("\s+", " ", p) for p in one_output]
    return one_output2
    

def split_context(one_input, MAX_LEN, language, encoding, keep_end, keep_colon):
    if language == "zh":
        MIN_LEN = 14
    elif language == "en":
        MIN_LEN = 12
    if len(encoding.encode(one_input)) <= MAX_LEN:
        return [one_input]
        
    one_output = []
    sents = sentence_tokenize(one_input, language, keep_end, keep_colon)
    this_segment = sents[0]
    
    subsent = ""
    for subsent in sents[1:-1]:
        if len(encoding.encode(this_segment)) + len(encoding.encode(subsent)) > MAX_LEN:
            one_output.append(this_segment)
            this_segment = subsent
        else:
            if keep_end:
                this_segment = this_segment + subsent
            elif language == "en" or (language == "zh" and (not re.search("\W$", this_segment))):
                this_segment = this_segment+" "+subsent
            else:
                this_segment = this_segment + subsent
            
    len_sent_1 = len(encoding.encode(sents[-1]))
    # last sentence 
    if len_sent_1 <= MIN_LEN or (len(encoding.encode(subsent)) + len_sent_1 <= MAX_LEN):
        if keep_end:
            this_segment = this_segment + sents[-1]
        elif language == "en" or (language == "zh" and (not re.search("\W$", this_segment))):
            this_segment = this_segment+" "+sents[-1]
        else:
            this_segment = this_segment + sents[-1]
    else:
        one_output.append(this_segment)
        this_segment = sents[-1]
        
    one_output.append(this_segment)
    
    return one_output

def sentence_tokenize_process_dot(text, recover=False):
    if not recover:
        text = re.sub(r"O\.S\.B\.M. ", r"O.S.B.M.", text)
        text = re.sub(r"(\W|^)([A-Z]\.) ?([A-Z]\.) ?([A-Za-z])", r"\1\2\3\4", text)
        text = re.sub(r"(\W|^)([A-Z]\.) ?([A-Za-z])", r"\1\2\3", text)  # J. K. Campbell
        text = re.sub(r"((\n\s*)|(\. ))(\d+)\.\s+", r"\1\4.", text) #1. XXX
        text = re.sub(r"^(\d+)\.\s+", r"\1.", text) #1. XXX
        text = re.sub(r"(\W|^)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec|No|Op|D|Dr|St)\.\s+", r"\1\2.", text)
        text = re.sub(r"(\W|^)(et al)\.\s+([a-z])", r"\1\2.\3", text)
        text = re.sub(r"Alexander v\. Holmes", r"Alexander v.Holmes", text)
        text = re.sub(r"Brown v\. Board", r"Brown v.Board", text)
    else:
        text = re.sub(r"^(\d+)\.", r"\1. ", text) #1. XXX
        text = re.sub(r"(\W|^)([A-Z]\.) ?([A-Z]\.) ?([A-Za-z])", r"\1\2 \3 \4", text) # J. K. Campbell
        text = re.sub(r"(\W|^)([A-Z]\.) ?([A-Z][a-z])", r"\1\2 \3", text)  # J. Campbell
        text = re.sub(r"(\W|^)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec|No|Op|D|Dr|St)\.", r"\1\2. ", text)
        text = re.sub(r"(\W|^)(et al)\.([a-z])", r"\1\2. \3", text)
        
        text = re.sub("O\.S\.B\.M\.", "O.S.B.M. ", text)
        text = re.sub("U\. +S\.", "U.S.", text)
        text = re.sub("U\.S\. *S\. *R\.", "U.S.S.R.", text)
        text = re.sub("D\. +C\.", "D.C.", text)
        text = re.sub("D\. +Roosevelt", "D. Roosevelt", text)
        text = re.sub("A\. *D\. *(\W)", r"A.D.\1", text)
        text = re.sub("A\. +D\.", r"A.D.", text)
        text = re.sub("F\. +C\.", r"F.C.", text)
        text = re.sub("J\. +League", r"J.League", text)
        text = re.sub(r"Alexander v\. *Holmes", r"Alexander v. Holmes", text)
        text = re.sub(r"Brown v\. *Board", r"Brown v. Board", text)
    return text
    
def sentence_tokenize(text, language, keep_end, keep_colon=False):
    if language == 'zh':
        if not keep_colon:
            text = re.sub(r"([:：])(\s+)", r"\1", text)
        sents2 = []
        sents = re.split("(。|！|？|；|\n+)", text) 
        # print(sents)
        for i in range(0, len(sents), 2):
            if i+1<len(sents):
                sent = sents[i] + sents[i+1]
                if not keep_end:
                    sent = sent.strip()
            else:
                sent = sents[i]
                if not keep_end:
                    sent = sent.strip()
            if sent:
                sents2.append(sent)
        # print(sents2)
        return sents2  
    elif language == 'en':
        text = sentence_tokenize_process_dot(text)
        if not keep_colon:
            text = re.sub(r"([:：])(\s+)", r"\1 ", text) # more space than Chinese
        
        sents2 = []
        sents = re.split("((?:[.!?;]\s+)|(?:\n+))", text)
        # print(sents)
        for i in range(0, len(sents), 2):
            if i+1<len(sents):
                sent = sents[i] + sents[i+1]
                if not keep_end:
                    sent = sent.strip()
            else:
                sent = sents[i]
                if not keep_end:
                    sent = sent.strip()
            if sent:
                sent = sentence_tokenize_process_dot(sent, recover=True)
                sents2.append(sent)
        return sents2 
