import asyncio
import json
import os
import sqlite3
import uuid
from typing import Annotated, Any, Dict, List, Literal, Sequence, Tuple

import dotenv
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

# TypedDict
from typing_extensions import TypedDict

from ouro import Ouro

dotenv.load_dotenv()


ouro = Ouro(api_key=os.getenv("OURO_API_KEY"))


# Define the state
class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
    summary: str
    current_question: str


# Function to load and process JSON data
def load_and_process_json(file_path: str):
    def metadata_func(record: dict, metadata: dict) -> dict:
        metadata["slug"] = record.get("slug")
        metadata["title"] = record.get("title")
        metadata["description"] = record.get("description")
        return metadata

    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".docs[], .guides[]",
        content_key="content",
        metadata_func=metadata_func,
    )

    data = loader.load()
    return data


# Function to split markdown content
def split_markdown(documents: List[Dict[str, Any]]):
    headers_to_split_on = [
        # ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    split_docs = []
    for doc in documents:
        splits = markdown_splitter.split_text(doc.page_content)
        for split in splits:
            split.metadata.update(doc.metadata)
        split_docs.extend(splits)

    return split_docs


# Main function to load, process, and add documents to the vector store
def load_process_and_add_documents(file_path: str):
    # Load JSON data
    documents = load_and_process_json(file_path)

    # Split markdown content
    split_documents = split_markdown(documents)

    return split_documents

    # Add to vector store
    # add_documents_to_vectorstore(split_documents)


# data = load_process_and_add_documents("./data/data.json")

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="agent",
    embedding_function=embeddings,
    persist_directory="./data/chroma",
)

# Add documents to vector store
# vectorstore.add_documents(documents=data)


docs_retriever = vectorstore.as_retriever(
    # search_type="mmr",
    search_kwargs={"k": 4}
)
search_docs = create_retriever_tool(
    docs_retriever,
    "search_docs",
    "Search and return information from the Luma documentation and guides.",
)


def summarize_conversation(state: AgentState):
    print("---SUMMARIZE CONVERSATION---")
    # First, we summarize the conversation
    summary = state.get("summary", "")

    if summary:
        # If a summary already exists, we use a different system prompt
        # to summarize it than if one didn't
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)

    # We now need to delete messages that we no longer want to show up
    # Keep the last 3 message groups (AI message + associated tool calls)
    messages_to_keep = 1
    delete_messages = [
        RemoveMessage(id=m.id) for m in state["messages"][:-messages_to_keep]
    ]

    return {
        "summary": response.content,
        "messages": delete_messages,
        "current_question": state["current_question"],
    }


def route_tools(
    state: AgentState,
) -> Literal["search_docs", "summarize_conversation", "__end__"]:
    next_node = tools_condition(state)
    # If no tools are invoked, return to the user
    if next_node == END:
        # If the conversation is long, summarize it before returning
        if len(state["messages"]) > 7:
            return "summarize_conversation"
        else:
            return END
    ai_message = state["messages"][-1]
    # This assumes single tool calls. To handle parallel tool calling, you'd want to
    # use an ANY condition
    first_tool_call = ai_message.tool_calls[0]
    # if first_tool_call["name"] in sensitive_tool_names:
    #     return "sensitive_tools"
    # return "safe_tools"

    # Store the current question when searching docs
    if first_tool_call["name"] == "search_docs":
        print("---SEARCH DOCS---")
        print("calls", first_tool_call)
        state["current_question"] = first_tool_call["args"]["query"]

    return first_tool_call["name"]


### Edges
def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)
    structured_llm = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | structured_llm

    messages = state["messages"]
    last_message = messages[-1]

    # TODO: this wont always be the case for longer conversations
    # question = messages[0].content
    print("state", state)
    question = state["current_question"]
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


tools = [search_docs]


# Define the nodes
def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")

    hermes_system_message = """You are Luma, an assistant for Luma text-to-video models. 
    You are tasked with helping the user with their query. 
    You will be provided with the user's query, and a summary of the conversation history if it exists. 
    You will then need to decide on the next action to take. 
    You can either search the Luma documentation and guides, or end the conversation. 
    You will then need to generate a response to the user's query.
    """

    current_question = state.get("current_question", "")
    # If a summary exists, we add this in as a system message
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]

    messages = [SystemMessage(content=hermes_system_message)] + messages

    # Initialize LLM
    model = ChatOpenAI(model="gpt-4o-mini")
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(messages)
    # We return a list, because this will get added to the existing list
    # Not always an AI message
    # return {"messages": [AIMessage(response.content)]}

    return {
        "messages": [response],
        "current_question": current_question,
        "summary": summary,
    }


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    # question = messages[0].content
    question = state["current_question"]

    msg = [
        SystemMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)
    response = model.invoke(msg)
    return {
        "messages": [response],
        "current_question": state["current_question"],
        "summary": state["summary"],
    }


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    # question = messages[0].content
    question = state["current_question"]
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.\n
        Use the following pieces of retrieved context to answer the question.\n
        If you don't know the answer, just say that you don't know.\n
        Use three sentences maximum and keep the answer concise.\n
        Question: {question} 
        Context: {context} 
        Answer:
        """,
        input_variables=["context", "question"],
    )
    # LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {
        "messages": [response],
        "current_question": state["current_question"],
        "summary": state["summary"],
    }


# Create the graph
workflow = StateGraph(AgentState)
# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
workflow.add_node("summarize_conversation", summarize_conversation)
workflow.add_node("search_docs", ToolNode([search_docs]))  # Search for documents
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant


# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# # We now add a conditional edge
# workflow.add_conditional_edges(
#     # First, we define the start node. We use `conversation`.
#     # This means these are the edges taken after the `conversation` node is called.
#     "agent",
#     # Next, we pass in the function that will determine which node is called next.
#     should_continue,
# )

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    route_tools,
)

# # Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "search_docs",
    # Assess agent decision
    grade_documents,
)
# workflow.add_edge("agent", END)

workflow.add_edge("rewrite", "agent")

workflow.add_edge("summarize_conversation", END)
workflow.add_edge("generate", END)


# Main execution function
def run_agent(graph, config, debug=False):
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        # This will add the message to the conversation
        inputs = {"messages": [HumanMessage(content=user_input)]}

        for event in graph.stream(inputs, config, stream_mode="values"):
            event["messages"][-1].pretty_print()

            if debug:
                for key, value in event.items():
                    for m in value["messages"]:
                        m.pretty_print()
                    if "summary" in value:
                        print(value["summary"])


async def respond_with_agent(user_message, websocket, config, debug=False):
    with SqliteSaver.from_conn_string("./data/agent.db") as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)
        # This will add the message to the conversation
        inputs = {"messages": [HumanMessage(content=user_message)]}
        response = ""
        message_id = config["message_id"]
        agent_user_id = config["agent_user_id"]
        recipient_id = config["recipient_id"]

        for event in graph.stream(inputs, config, stream_mode="messages"):  # values
            message = event[0]
            details = event[1]

            if (
                details["langgraph_node"] == "agent"
                or details["langgraph_node"] == "generate"
            ):
                response += message.content
                if websocket:
                    await websocket.send(
                        json.dumps(
                            {
                                "event": "llm-response",
                                "recipient_id": recipient_id,
                                "data": {
                                    "content": message.content,
                                    "id": message_id,
                                    "user_id": agent_user_id,
                                },
                            }
                        )
                    )

                # Artificially wait a few ms to simulate typing
                await asyncio.sleep(0.001)

        # Send a message to indicate the stream has ended
        if websocket:
            await websocket.send(
                json.dumps(
                    {
                        "event": "llm-response-end",
                        "recipient_id": recipient_id,
                        "data": {"id": message_id, "user_id": agent_user_id},
                    }
                )
            )

        print("response", response)

        return response


if __name__ == "__main__":

    with SqliteSaver.from_conn_string("./data/agent.db") as checkpointer:
        # Compile the graph
        graph = workflow.compile(checkpointer=checkpointer)
        config = {
            "configurable": {
                "thread_id": 6,
            }
        }
        # values = graph.get_state(config).values
        # pprint.pp(values)
        run_agent(graph, config, debug=False)
