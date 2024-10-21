from tempfile import NamedTemporaryFile

import base64
import streamlit as st
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from tools import ImageCaptionTool, ObjectDetectionTool
from openai import OpenAI
import logging

OPENAI_API_KEY=''

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

##############################
### initialize agent #########
##############################
tools = [ImageCaptionTool(), ObjectDetectionTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

llm = ChatOpenAI(
    openai_api_key='',
    temperature=0,
    model_name="gpt-4o"
)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stopping_method='generate'
)

# set title
st.title('Architecture assistant')

# set header
st.header("Please upload an image")

# upload file
file = st.file_uploader("", type=["jpeg", "jpg", "png"])

if file:
    # display image
    st.image(file, use_column_width=True)

    # text input
    user_question = st.text_input('Ask a question about your image:')

    ##############################
    ### compute agent response ###
    ##############################
    with NamedTemporaryFile(dir='.') as f:
        f.write(file.getbuffer())
        image_path = f.name
        logging.warning(image_path)

        base64_image = encode_image(image_path)

        client = OpenAI(api_key=OPENAI_API_KEY)
        if user_question and user_question != "":
            with st.spinner(text="In progress..."):
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": { "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                               "text": user_question
                            }
                            ]
                        }
                    ],
                    temperature=1,
                    max_tokens=2048,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    response_format={
                        "type": "text"
                    }
                )
                st.write(response.choices[0].message.content)

        # write agent response
        #if user_question and user_question != "":
         #   with st.spinner(text="In progress..."):
          #      response = agent.run('{}, this is the image path: {}'.format(user_question, image_path))
           #     st.write(response)

