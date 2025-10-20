"""
"chat_template":
{%- if tools is iterable and tools | length > 0 %}
    {{- '<|im_start|><|system|>'}}
    {{- '당신은 도구 호출 기능을 갖춘 유용한 도우미입니다. 사용자의 요청을 처리하기 위해서 필요한 도구가 주어진 목록에 있는 경우 도구 호출로 응답하세요.
필요한 도구가 목록에 없는 경우에는 도구 호출 없이 사용자가 요구한 정보를 제공하세요.
필요한 도구가 목록에 있지만 해당 도구를 호출하는데 필요한 argument 정보가 부족한 경우 해당 정보를 사용자에게 요청하세요.
사용자의 요청을 처리하기 위해 여러번 도구를 호출할 수 있어야 합니다.
도구 호출 이후 도구 실행 결과를 입력으로 받으면 해당 결과를 활용하여 답변을 생성하세요.

다음은 접근할 수 있는 도구들의 목록 입니다:
<tools>
'}}
    {%- for t in tools %}
        {{- t | tojson }}
        {{- '
' }}
    {%- endfor %}
    {{- '</tools>' }}
    {{- '

도구를 호출하려면 아래의 JSON으로 응답하세요.
도구 호출 형식: <tool_call>{"name": 도구 이름, "arguments": dictionary 형태의 도구 인자값}</tool_call>' }}
    {{- '<|im_end|>' }}
{%- endif %}

{%- for message in messages %}
    {%- if message.role == 'system' %}
        {{- '<|im_start|><|system|>' + message.content + '<|im_end|>'}}
    {%- elif message.role == 'user' %}
        {{- '<|im_start|><|user|>' + message.content + '<|im_end|>'}}
    {%- elif message.role == 'assistant' %}
        {{- '<|im_start|><|assistant|>'}}
        {%- set content = '' %}
        {%- if message.content is defined %}
            {%- set content = message.content %}
        {%- endif %}

{%- if add_generation_prompt and not (message.reasoning_content is defined and message.reasoning_content is not none) %}
    {%- if '</think>' in message.content %}
        {%- set content = message.content.split('</think>'.strip())[-1].lstrip('\n') %}
    {%- endif %}
{%- endif %}

        {{- content}}
        {%- if message.tool_calls is defined %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>' }}
                {{- '{' }}
                {{- '"name": "' }}
                {{- tool_call.name }}
                {{- '"' }}
                {%- if tool_call.arguments is defined %}
                    {{- ', ' }}
                    {{- '"arguments": ' }}
                    {{- tool_call.arguments|tojson }}
                {%- endif %}
                {{- '}' }}
                {{- '</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>'}}

    {%- elif message.role == 'tool' %}
        {{- '<|im_start|><|extra_id_13|><tool_output>' + message.content + '</tool_output><|im_end|>'}}
    {%- endif %}
{%- endfor %}

    {%- if add_generation_prompt %}
        {{- '<|im_start|><|assistant|>' }}
    {%- endif %}
"""
