import gradio as gr
import plotly.express as px
import pandas as pd
from langchain.document_loaders import JSONLoader
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import json
import time
from dotenv import load_dotenv

load_dotenv()

STREAMING_SLEEP_TIMER = 0.0001
BARCHART_SCORE_ADJUSTMENT = 0.05
BARCHART_SCORE_SCALING_MIN = 0.3
BARCHART_SCORE_SCALING_MAX = 0.7

# def metadata_func(record: dict, metadata: dict) -> dict:
#     metadata["consultant_name"] = record.get("name")

#     return metadata

# loader = JSONLoader(
#     file_path='./consultant_profile_example_data_summarized.json',
#     content_key="text",
#     jq_schema=".consultant_profiles[]",
#     metadata_func=metadata_func
# )

# consultant_data = loader.load()

with open("./consultant_profile_example_data_summarized.json", "r") as fp:
    consultant_data = json.load(fp)['consultant_profiles']
consultant_data = [Document(page_content=doc['text'], metadata={"name": doc['name']}) for doc in consultant_data]


embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', deployment='ptc-embedding-ada-002', chunk_size=1)
docsearch = FAISS.from_documents(consultant_data, embeddings) #, normalize_L2=True)
# embedding_retriever = docsearch.as_retriever(search_kwargs={"k": 5})


def summarize_template_offers():
    # Customized Prompt
    OFFER_SUMMARIZE = (
    """
    The following is a project offer for a consultant in German. Please summarize the offer in German and keep all important hard facts, such as dates:
    OFFER: {offer}
    """
    )

    return PromptTemplate(
        input_variables=["offer"],
        template=OFFER_SUMMARIZE,
    )

def q_and_a_template():
    ANSWER = """
    The following text contains a consultant profile in English described with the keyword PROFILE. It contains a short description of the person, career overview, skill and technology experiences, and short overview about projects. Then use the query defined in QUERY to give an answer in up to 2 sentences. Do not repeat yourself in those two sentences. Answer in English.
    PROFILE: {profile}
    QUERY: {query}
    """
    return PromptTemplate(
        input_variables=["profile", "query"],
        template=ANSWER,
    )

def summarize_workflows(llm, profile, template):
    """
    Helper function for separating the api-link creation and the post-processing on the response
    """
    # sum_prompt = summarize_template()
    summarize_chain = LLMChain(llm=llm, prompt=template())
    return summarize_chain.run(profile)

def build_sum_offer(llm, offer_content):
    offer_content_summarized = summarize_workflows(llm, offer_content, summarize_template_offers)
    return offer_content_summarized


# llm = AzureOpenAI(
#     model_name='text-davinci-003',
#     deployment_name='ptc-davinci-003',
#     temperature=0,
#     max_tokens=400
# )

# chat_llm = AzureChatOpenAI(
#     model_name='gpt-35-turbo',
#     deployment_name='ptc-chat-gpt-35',
#     temperature=0,
#     max_tokens=400,
# )
from langchain_community.llms import VLLMOpenAI

llm = VLLMOpenAI(
    openai_api_key="EMPTY",
    openai_api_base="https://2f06-34-91-97-80.ngrok-free.app/v1",
    model_name="TheBloke/Llama-2-7B-Chat-GPTQ",
    # model_kwargs={"stop": ["."]},
)

chat_llm = VLLMOpenAI(
    openai_api_key="EMPTY",
    openai_api_base="https://2f06-34-91-97-80.ngrok-free.app/v1",
    model_name="TheBloke/Llama-2-7B-Chat-GPTQ",
    # model_kwargs={"stop": ["."]},
)
def build_explain_prompt():
    template="You are a Business Manager working in a consulting firm and looking for new projects for the consultants. You have an OFFER and a PROFILE and you want to know the answer to a given QUESTION that relates to OFFER and PROFILE. Answer in 2 to 3 sentences"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="""OFFER: {offer} PROFILE: "{profile}" QUESTION: {question}"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    return chat_prompt

def explain_workflow(profile_name: str, offer: str, question: str):
    chat_prompt = build_explain_prompt()
    profile = [prof.page_content for prof in consultant_data if prof.metadata['name'] == profile_name][0]
    explain_chain = LLMChain(llm=chat_llm, prompt=chat_prompt)
    return explain_chain.run({"profile": profile, "offer": offer, "question": question})

# summarized_offer = build_sum_offer(llm, offer_content)

def find_best_matches_via_llm(offer):
    if offer == "" or offer is None:
        return
    print("Starting match finder")
    # summarized_offer = build_sum_offer(llm, offer)
    print("Summarization finished")
    results = docsearch.similarity_search_with_score(offer)
    results_processed = [{"score": 1 - doc[1], "name": doc[0].metadata['name']} for doc in results]
    scores = [doc['score'] for doc in results_processed]
    names = [doc['name'] for doc in results_processed]
    return create_barplot(scores, names)

def answer_question_via_llm(offer, profile_name, question):
    print("Starting QnA process")
    if question == "" or question is None:
        return ""
    q_and_a_chain = LLMChain(llm=llm, prompt=q_and_a_template())
    profile_text = [prof.page_content for prof in consultant_data if prof.metadata['name'] == profile_name][0]
    print(question)
    result = q_and_a_chain.run({"profile": profile_text, "query": question})
    output = "Question: "
    for char in question:
        time.sleep(STREAMING_SLEEP_TIMER)
        output += char
        yield output
    output += '\nAnswer: '
    for char in result:
        time.sleep(STREAMING_SLEEP_TIMER)
        output += char
        yield output

def get_summaries_of_consultants():
    summaries = {}

    for consultant in consultant_data:
        summaries[consultant.metadata['name']] = consultant.page_content
    
    return summaries

def update_summary_field_with_best_match():
    summaries = get_summaries_of_consultants()
    return summaries[best_match]

def get_summary_of_selected_consultant(name: gr.SelectData):
    summaries = get_summaries_of_consultants()

    summary_for_output = summaries[name.value]
    return summary_for_output

def adjust_scaling_of_scores(scores):
    output = []
    for x in scores:
        output.append((max(min(x, BARCHART_SCORE_SCALING_MAX), BARCHART_SCORE_SCALING_MIN) - min(max(x, BARCHART_SCORE_SCALING_MAX), BARCHART_SCORE_SCALING_MIN))/ (BARCHART_SCORE_SCALING_MAX-BARCHART_SCORE_SCALING_MIN))
    return output

def create_barplot(scores: list, names_list):
    updated_scores = adjust_scaling_of_scores(scores)
    data = {'Score': updated_scores, 'Consultant': names_list}
    df = pd.DataFrame(data=data)
    df = df.sort_values('Score', ascending=False)
    # set best match in global variable
    global best_match
    best_match = df.Consultant.iloc[0]
    # dynamic adjustment of score range
    #min_range = round(df.tail(1)['Score'].item(), 2) - BARCHART_SCORE_ADJUSTMENT
    #max_range = round(df.head(1)['Score'].item(), 2) + BARCHART_SCORE_ADJUSTMENT
    color_scale = ("rgb(253, 191, 55)", "rgb(245, 237, 99)", "rgb(102, 157, 146)")
    fig = px.bar(df, x='Score', y='Consultant', color="Score", range_color=[0,1], color_continuous_scale=[(0.00, color_scale[0]), (0.5, color_scale[0]), (0.51, color_scale[1]), (0.8, color_scale[1]), (0.8, color_scale[2]), (1.00, color_scale[2])], title="Score overview of the best matches")
    fig.update_xaxes(
        range=[0, 1],
        scaleanchor="x",
        scaleratio=1)
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig

def update_selected_summary_dropdown():
    return gr.Dropdown.update(value=best_match, interactive=True)

def get_dropdown_choices():
    output = [consultant.metadata['name'] for consultant in consultant_data]
    return output

def update_text_boxes_with_custom_text(text=""):
    yield text

def answer_profile_offer_question(offer, profile, question):
    if question == "" or question is None:
        question = "Is this profile a good fit for the offer? Answer in 2 to 3 sentences"#"Which skills of the profile match with the offer?"
    # insert call to langchain/chatgpt
    result = explain_workflow(profile, offer, question)
    output = "Question: "
    for char in question:
        time.sleep(STREAMING_SLEEP_TIMER)
        output += char
        yield output
    output += '\nAnswer: '
    for char in result:
        time.sleep(STREAMING_SLEEP_TIMER)
        output += char
        yield output


# Frontend part
title = """<h1 align="center">CBTW's Project-Profile Matcher - powered by LLMs!</h1>"""
description = """<br><h3 align="center">This is a PoC to test the feasibility of using LLMs to compare consultant profiles and project offers and find matches.</h3>"""
theme = gr.themes.Soft().set(
    body_background_fill="#FDF6E6", 
    body_text_color="#1C1C1C", 
    block_label_background_fill="#FDF6E6", 
    block_label_text_color="#1C1C1C"
)

best_match = ""

with gr.Blocks(theme=theme) as demo:
    gr.HTML(title)
    gr.HTML(description)
    
    gr.HTML("""<br><h3 align="center">Find the best matches for an offer</h3>""")
    with gr.Row():
        with gr.Column():
            offer_input_textbox = gr.Textbox(
                label="Offer", 
                placeholder="Offer text",
                show_label=False,
                max_lines=3
                ).style(container=False)
        with gr.Column():
            with gr.Row():
                offer_submit_button = gr.Button("Find similar profiles")

    with gr.Row():
        barchart = gr.Plot(
            label="Most similar consultant profiles",
            show_label=False
        )
        offer_comparison_textbox = gr.Textbox(
            label="Offer used for match",
            show_label=True,
            interactive=False
        )
        
    gr.HTML( """<h3 align="center">Get a summarized version of a consultant's profile</h3>""")
    with gr.Row():
        with gr.Column():
            summary_dropdown = gr.Dropdown(
                choices = get_dropdown_choices(),
                type="value",
                label="Consultants",
                value=best_match,
                interactive=True
            )
        with gr.Column():
            summary_text = gr.TextArea(
                value="",
                label="Summary of selected consultant profile"
            )

    gr.HTML( """<h3 align="center">Ask questions about the profile on the left and/or about the offer from above</h3>""")
    with gr.Row():
        with gr.Column():
            profile_explain_dropdown = gr.Dropdown(
                choices = get_dropdown_choices(),
                type = "value",
                label = "Select a Consultant"
            )

        with gr.Column():
            profile_explain_question_text_area = gr.Textbox(
                value="",
                label="Enter your question",
                info="e.g. 'What skills is the consultant missing for this offer?', 'Does the consultant have experience in XY?' ",
                placeholder="Why is this profile a good fit for the offer" #"Which skills of the profile match with the offer?"
            )

        with gr.Column():
            profile_explain_button = gr.Button(value="Explain").style(size="sm", container=False)
        
        with gr.Column():
            profile_explain_answer_textbox = gr.TextArea(
                value = "",
                label = "Explanation"
            )

    #gr.HTML( """<h3 align="center">Ask a question about the profile</h3>""")
    #with gr.Row():
    #    with gr.Column():
    #        profile_qna_dropdown = gr.Dropdown(
    #            choices = get_dropdown_choices(),
    #            type = "value",
    #            label="Select a Consultant"
    #        )
    #    
    #    with gr.Column():
    #        question_qna_text_area = gr.Textbox(
    #            value="",
    #            label="Ask a question about the profile"
    #        )
    #        qna_button = gr.Button("Get answer")
    #    
    #    with gr.Column():
    #        answer_qna_text_area = gr.TextArea(
    #            value="",
    #            label="Answer",
    #        )
            

    offer_submit_event = offer_input_textbox.submit(fn=find_best_matches_via_llm, inputs=[offer_input_textbox], outputs=[barchart], queue=False)
    offer_submit_event.then(fn=update_text_boxes_with_custom_text, inputs=[offer_input_textbox], outputs=[offer_comparison_textbox])
    offer_submit_event.then(fn=update_text_boxes_with_custom_text, inputs=[], outputs=[offer_input_textbox])
    offer_submit_event.then(fn=update_selected_summary_dropdown, outputs=[summary_dropdown])
    offer_submit_event.then(fn=update_summary_field_with_best_match, outputs={summary_text})

    offer_submit_click_event = offer_submit_button.click(fn=find_best_matches_via_llm, inputs=[offer_input_textbox], outputs=[barchart], queue=False)
    offer_submit_click_event.then(fn=update_text_boxes_with_custom_text, inputs=[offer_input_textbox], outputs=[offer_comparison_textbox])
    offer_submit_click_event.then(fn=update_text_boxes_with_custom_text, inputs=[], outputs=[offer_input_textbox])
    offer_submit_click_event.then(fn=update_selected_summary_dropdown, outputs=[summary_dropdown])
    offer_submit_click_event.then(fn=update_summary_field_with_best_match, outputs={summary_text})
    
    summary_dropdown_event = summary_dropdown.select(fn=get_summary_of_selected_consultant, outputs=[summary_text])
    summary_dropdown_event.then(lambda: gr.update(interactive=True), None, [summary_text], queue=False)

    explain_submit_event = profile_explain_question_text_area.submit(fn=answer_profile_offer_question, inputs=[offer_comparison_textbox, profile_explain_dropdown, profile_explain_question_text_area], 
                                                                 outputs=[profile_explain_answer_textbox], queue=True)
    explain_submit_event.then(fn=update_text_boxes_with_custom_text, inputs=[], outputs=[profile_explain_question_text_area])
    explain_submit_click_event = profile_explain_button.click(fn=answer_profile_offer_question, inputs=[offer_comparison_textbox, profile_explain_dropdown, profile_explain_question_text_area], 
                                                                 outputs=[profile_explain_answer_textbox], queue=True)
    explain_submit_click_event.then(fn=update_text_boxes_with_custom_text, inputs=[], outputs=[profile_explain_question_text_area])

    
    #question_qna_click_event = qna_button.click(fn=answer_question_via_llm, inputs=[offer_input_textbox, profile_qna_dropdown, question_qna_text_area],
    #                                           outputs=[answer_qna_text_area], queue=True)
    #question_qna_click_event.then(fn=update_text_boxes_with_custom_text, inputs=[], outputs=[question_qna_text_area])
    #question_qna_submit_event = question_qna_text_area.submit(fn=answer_question_via_llm, inputs=[offer_input_textbox, profile_qna_dropdown, question_qna_text_area],
    #                                           outputs=[answer_qna_text_area], queue=True)
    #question_qna_submit_event.then(fn=update_text_boxes_with_custom_text, inputs=[], outputs=[question_qna_text_area])



demo.queue(max_size=20, concurrency_count=5)
demo.launch() #server_port=8080)
