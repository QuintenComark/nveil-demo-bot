import os
import base64
import re
import json

import streamlit as st
import openai
from openai import AssistantEventHandler
from tools import TOOL_MAP
from typing_extensions import override
from dotenv import load_dotenv
import streamlit_authenticator as stauth


#import yaml
#from yaml.loader import SafeLoader
#with open('auth.yaml') as file:
#    config = yaml.load(file, Loader=SafeLoader)


load_dotenv()



st.markdown(
    """
<style>
    .st-emotion-cache-1c7y2kd {
        flex-direction: row-reverse;
        text-align: right;
    }
    .stChatInput{
        background-color: #FFFFFF !important;
    }

    button{
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }

    .st-emotion-cache-1up18o9{
        background-color: transparent !important;
        color: #000 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


def str_to_bool(str_input):
    if not isinstance(str_input, str):
        return False
    return str_input.lower() == "true"

def sum_function(*args, **kwargs):
    # Combine args and kwargs into a single list of values
    values = list(args) + list(kwargs.values())
    return str(sum(values))

def celsius_to_kelvin(celsius):
    # The formula to convert Celsius to Kelvin is K = C + 273.15
    kelvin = celsius + 273.15
    return str(kelvin)

TOOL_MAP = {
    'sum': sum_function,
    'celsius_to_kelvin': celsius_to_kelvin,
    # ... other function mappings ...
    # ... other function mappings ...
}


st.sidebar.image('https://nveil.ai/wp-content/uploads/2024/06/nveil-white-1.png', width=200)

azure_openai_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
azure_openai_key = st.secrets["AZURE_OPENAI_KEY"]
# Load environment variables
# openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_api_key = st.secrets["OPENAI_API_KEY"]
authentication_required = str_to_bool(st.secrets["AUTHENTICATION_REQUIRED"])
# authentication_required = str_to_bool(os.environ.get("AUTHENTICATION_REQUIRED", False))
# instructions = os.environ.get("RUN_INSTRUCTIONS", "")
# assistant_title = os.environ.get("ASSISTANT_TITLE", "Nveil.ai Demo Bot")
assistant_title = st.secrets["ASSISTANT_TITLE"]
enabled_file_upload_message = st.secrets["ENABLED_FILE_UPLOAD_MESSAGE"]
# enabled_file_upload_message = os.environ.get("ENABLED_FILE_UPLOAD_MESSAGE", "Upload a file")


# Load authentication configuration
#if authentication_required:
#    if "credentials" in st.secrets:
#        authenticator = stauth.Authenticate(
#            st.secrets["credentials"].to_dict(),
#            st.secrets["cookie"]["name"],
#            st.secrets["cookie"]["key"],
#            st.secrets["cookie"]["expiry_days"],
#        )
#    else:
#        authenticator = None  # No authentication should be performed

if authentication_required:
    authenticator = stauth.Authenticate(
        st.secrets['credentials'],
        st.secrets['cookie']['name'],
        st.secrets['cookie']['key'],
        st.secrets['cookie']['expiry_days'],
        st.secrets['preauthorized']
    )
else:
    authenticator = None  # No authentication should be performed




client = None
if azure_openai_endpoint and azure_openai_key:
    client = openai.AzureOpenAI(
        api_key=azure_openai_key,
        api_version="2024-02-15-preview",
        azure_endpoint=azure_openai_endpoint,
    )
else:
    client = openai.OpenAI(api_key=openai_api_key)

my_assistants = client.beta.assistants.list(
    order="desc",
    limit="20",
)

 #Extract the assistant objects
assistant_dict = {assistant.name: assistant for assistant in my_assistants.data}
# Create a dropdown menu with the assistant names
selected_assistant_name = st.sidebar.selectbox('Select an assistant', list(assistant_dict.keys()))
# Get the selected assistant
selected_assistant = assistant_dict[selected_assistant_name]
# Get the ID and description of the selected assistant
assistant_id = selected_assistant.id


class EventHandler(AssistantEventHandler):
    @override
    def on_event(self, event):
        pass

    @override
    def on_text_created(self, text):
        st.session_state.current_message = ""
        with st.chat_message("Assistant"):
            st.session_state.current_markdown = st.empty()

    @override
    def on_text_delta(self, delta, snapshot):
        if snapshot.value:
            text_value = re.sub(
                r"\[(.*?)\]\s*\(\s*(.*?)\s*\)", "Download Link", snapshot.value
            )
            st.session_state.current_message = text_value
            st.session_state.current_markdown.markdown(
                st.session_state.current_message, True
            )

    @override
    def on_text_done(self, text):
        format_text = format_annotation(text)
        st.session_state.current_markdown.markdown(format_text, True)
        st.session_state.chat_log.append({"name": "assistant", "msg": format_text})

    @override
    def on_tool_call_created(self, tool_call):
        if tool_call.type == "code_interpreter":
            st.session_state.current_tool_input = ""
            with st.chat_message("Assistant"):
                st.session_state.current_tool_input_markdown = st.empty()

    @override
    def on_tool_call_delta(self, delta, snapshot):
        if 'current_tool_input_markdown' not in st.session_state:
            with st.chat_message("Assistant"):
                st.session_state.current_tool_input_markdown = st.empty()

        if delta.type == "code_interpreter":
            if delta.code_interpreter.input:
                st.session_state.current_tool_input += delta.code_interpreter.input
                input_code = f"### code interpreter\ninput:\n```python\n{st.session_state.current_tool_input}\n```"
                st.session_state.current_tool_input_markdown.markdown(input_code, True)

            if delta.code_interpreter.outputs:
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        pass

    @override
    def on_tool_call_done(self, tool_call):
        st.session_state.tool_calls.append(tool_call)
        if tool_call.type == "code_interpreter":
            if tool_call.id in [x.id for x in st.session_state.tool_calls]:
                return
            input_code = f"### code interpreter\ninput:\n```python\n{tool_call.code_interpreter.input}\n```"
            st.session_state.current_tool_input_markdown.markdown(input_code, True)
            st.session_state.chat_log.append({"name": "assistant", "msg": input_code})
            st.session_state.current_tool_input_markdown = None
            for output in tool_call.code_interpreter.outputs:
                if output.type == "logs":
                    output = f"### code interpreter\noutput:\n```\n{output.logs}\n```"
                    with st.chat_message("Assistant"):
                        st.markdown(output, True)
                        st.session_state.chat_log.append(
                            {"name": "assistant", "msg": output}
                        )
        elif (
            tool_call.type == "function"
            and self.current_run.status == "requires_action"
        ):
            with st.chat_message("Assistant"):
                msg = f"### Function Calling: {tool_call.function.name}"
                st.markdown(msg, True)
                st.session_state.chat_log.append({"name": "assistant", "msg": msg})
            tool_calls = self.current_run.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            for submit_tool_call in tool_calls:
                tool_function_name = submit_tool_call.function.name
                tool_function_arguments = json.loads(
                    submit_tool_call.function.arguments
                )
                tool_function_output = TOOL_MAP[tool_function_name](
                    **tool_function_arguments
                )
                tool_outputs.append(
                    {
                        "tool_call_id": submit_tool_call.id,
                        "output": tool_function_output,
                    }
                )

            with client.beta.threads.runs.submit_tool_outputs_stream(
                thread_id=st.session_state.thread.id,
                run_id=self.current_run.id,
                tool_outputs=tool_outputs,
                event_handler=EventHandler(),
            ) as stream:
                stream.until_done()



def create_thread(content, file):
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    if file is not None:
        messages[0].update({"file_ids": [file.id]})
    thread = client.beta.threads.create()
    return thread


def create_message(thread, content, file):
    attachments = []
    if file is not None:
        attachments.append(
            {"file_id": file.id, "tools": [{"type": "code_interpreter"}]}
        )
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=content, attachments=attachments
    )


def create_file_link(file_name, file_id):
    content = client.files.content(file_id)
    content_type = content.response.headers["content-type"]
    b64 = base64.b64encode(content.text.encode(content.encoding)).decode()
    link_tag = f'<a href="data:{content_type};base64,{b64}" download="{file_name}">Download Link</a>'
    return link_tag


def format_annotation(text):
    citations = []
    text_value = text.value
    for index, annotation in enumerate(text.annotations):
        text_value = text.value.replace(annotation.text, f" [{index}]")

        if file_citation := getattr(annotation, "file_citation", None):
            cited_file = client.files.retrieve(file_citation.file_id)
            citations.append(
                f"[{index}] {cited_file.filename}"
            )
        elif file_path := getattr(annotation, "file_path", None):
            link_tag = create_file_link(
                annotation.text.split("/")[-1],
                file_path.file_id,
            )
            text_value = re.sub(r"\[(.*?)\]\s*\(\s*(.*?)\s*\)", link_tag, text_value)
    text_value += "\n\n" + "\n".join(citations)
    return text_value


def run_stream(user_input, file):
    if "thread" not in st.session_state:
        st.session_state.thread = create_thread(user_input, file)
    create_message(st.session_state.thread, user_input, file)
    with client.beta.threads.runs.stream(
        thread_id=st.session_state.thread.id,
        assistant_id=assistant_id,
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()


def handle_uploaded_file(uploaded_file):
    file = client.files.create(file=uploaded_file, purpose="assistants")
    return file


def render_chat():
    for chat in st.session_state.chat_log:
        with st.chat_message(chat["name"]):
            st.markdown(chat["msg"], True)


if "tool_call" not in st.session_state:
    st.session_state.tool_calls = []

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

if "in_progress" not in st.session_state:
    st.session_state.in_progress = False


def disable_form():
    st.session_state.in_progress = True


def auth():
    
    col1, col2 = st.columns(2)
    with col1:
        try:
            authenticator.login()
        except Exception as e:
            st.error(e)

    
    with col2:
        try:
            (email_of_registered_user,
            username_of_registered_user,
            name_of_registered_user) = authenticator.register_user(pre_authorization=False)
            if email_of_registered_user:
                st.success('User registered successfully')
                st.session_state["user_registered"] = True
        except Exception as e:
            st.error(e)
    
    if st.session_state["authentication_status"] == False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] == None:
        st.warning('Please enter your username and password')

    # Saving config file
    #with open('auth.yaml', 'w', encoding='utf-8') as file:
    #    yaml.dump(config, file, default_flow_style=False)

    

def main():
    if (authenticator is not None):
        
        if not st.session_state["authentication_status"]:
            
            auth()
            return
        else:
            authenticator.logout(location="sidebar")

    


    
   
   
    

    st.title(assistant_title)
    user_msg = st.chat_input(
        "Message", on_submit=disable_form, disabled=st.session_state.in_progress
    )

    if st.sidebar.button('Clear chat'):
    # Clear the chat history
        st.session_state.chat_log = []
        # Create a new thread by creating a new assistant session
        # Replace 'Enter Assistant ID' with the actual method to create a new thread
        st.session_state.thread_id = create_thread("Enter Assistant ID", None).id


    
    if enabled_file_upload_message:
        uploaded_file = st.sidebar.file_uploader(
            enabled_file_upload_message,
            type=[
                "txt",
                "pdf",
                "png",
                "jpg",
                "jpeg",
                "csv",
                "json",
                "geojson",
                "xlsx",
                "xls",
            ],
            disabled=st.session_state.in_progress,
        )
    else:
        uploaded_file = None

    if user_msg:
        render_chat()
        with st.chat_message("user"):
            st.markdown(user_msg, True)
        st.session_state.chat_log.append({"name": "user", "msg": user_msg})

        file = None
        if uploaded_file is not None:
            file = handle_uploaded_file(uploaded_file)
        run_stream(user_msg, file)
        st.session_state.in_progress = False
        st.session_state.tool_call = None
        st.rerun()

    render_chat()


if __name__ == "__main__":
    main()
