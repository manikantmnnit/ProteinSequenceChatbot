import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
import langchain_community
from langchain_community.tools import TavilySearchResults,DuckDuckGoSearchResults

from source_file.llm_function import initialize_llm, get_llm_response
from source_file.sequence_analysis_function import parse_fasta, preprocessing_protein_sequence, basic_sequence_analysis
from langchain.agents import AgentExecutor, create_tool_calling_agent
from dotenv import load_dotenv
import os
load_dotenv()
from source_file.tools import get_all_tools

# App configuration
st.set_page_config(page_title="🔬 Protein Chatbot")
st.title("🔬 Protein Science Chatbot")
st.write("Ask me questions related to protein sequences")

# Sidebar configuration
with st.sidebar:
    groq_api_key = st.text_input("🔑 GROQ API Key", type="password")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 100, 2000, 1000, 100)

if not groq_api_key:
    st.warning("Please enter your GROQ API key in the sidebar")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["Protein Analysis", "Sequence Manipulations", "Owl"])

with tab1:
    st.header("Protein Analysis")
    st.write("This app allows you to analyze protein sequences and structures.")

    preprocessed_sequence = st.text_area(
        "Enter protein sequence (one-letter code):",
        "YGSQTPSEECLFLER",
        height=150
    )

    # Initialize LLM and prompt template
    llm = initialize_llm(groq_api_key, temperature, max_tokens)
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat interface
    with st.form("chat_form"):
        user_input = st.text_area("Enter your question:")
        submitted = st.form_submit_button("Submit")

        if submitted and user_input and preprocessed_sequence:
            # Process the sequence only after submission
            processed_protein = RunnableSequence(
                RunnableLambda(lambda x: preprocessing_protein_sequence(x)),
                RunnableLambda(lambda x: basic_sequence_analysis(x))
            )
            sequence_processed = processed_protein.invoke(preprocessed_sequence)

            # Now create system prompt using processed sequence
            system_prompt = f"""
            You are an expert bioinformatics assistant specializing in protein science.

            Here is the processed protein information:
            {sequence_processed}

            The user asks a question based on this sequence and its analysis.

            if required, you can preprocess the protein sequence and provide processed sequence.

            Answer the user's question based only on this data.
            Be concise and scientific in your explanation.
            Return your response in markdown format.
            """

            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessage(content="{user_input}")  # This now correctly refers to user_input
            ])

            # Get response from LLM
            response, updated_history = get_llm_response(
                llm,
                prompt,
                user_input,
                st.session_state.chat_history
            )

            # Update chat history
            st.session_state.chat_history = updated_history
            st.success("🧪 Response:")
            st.markdown(response)

    # Display chat history
    if st.session_state.chat_history:
        st.divider()
        st.subheader("💬 Chat History")
        for msg in st.session_state.chat_history:
            role = "👤 You" if isinstance(msg, HumanMessage) else "🤖 Assistant"
            st.markdown(f"**{role}:** {msg.content}")
    # Reset chat button
    if st.button("Reset Chat"):
        st.session_state.chat_history = []
        st.info("Chat history has been reset!")


with tab2:
    import pandas as pd
    from Bio import SeqIO
    from Bio.Seq import Seq
    import io

    st.header("🧬 Sequence + Chatbot")
    uploaded_file = st.file_uploader("📂 Upload FASTA file", type=["fasta", "fa", "txt"])

    if uploaded_file:
    # Parse FASTA using BioPython
        fasta_bytes = uploaded_file.read()
        record_dict = SeqIO.to_dict(SeqIO.parse(io.StringIO(fasta_bytes.decode("utf-8")), "fasta"))
        if record_dict:
            df = pd.DataFrame({
            "ID": record_dict.keys(),
            "Description": [rec.description for rec in record_dict.values()],
            "Sequence": [str(rec.seq) for rec in record_dict.values()],
            "Length": [len(rec) for rec in record_dict.values()]
        })
        
        # Display 5 random samples
        st.write("Preview (5 random sequences):")
        st.dataframe(df.sample(5))
              
        selected_column = st.selectbox("Select column containing sequences", options=df.columns, index=1)
        slider_point = st.slider("Select sequence index", 0, df.shape[0], 0, 1)

        preprocessed_sequence = df[selected_column][slider_point]
        st.text_area("Selected Sequence", value=preprocessed_sequence, height=200)
    else:
            st.warning("No valid sequences found in the uploaded file.")


        # Preprocess and analyze sequence
    processed_protein = RunnableSequence(
            RunnableLambda(preprocessing_protein_sequence),
            RunnableLambda(basic_sequence_analysis)
        )
    sequence_processed = processed_protein.invoke(preprocessed_sequence)

    
    search_tool_1 = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))  
    search_tool_2 = DuckDuckGoSearchResults()
    bio_tools = get_all_tools()

    all_tools = [search_tool_1, search_tool_2] + bio_tools

        # Initialize chat history
    if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Define LLM and prompt
    llm = ChatGroq(model="llama3-8b-8192", temperature=temperature, max_tokens=max_tokens)

    system_prompt = f"""
                You are a protein science expert.

                Here is the processed protein sequence data:
                {sequence_processed}

                Answer the user's questions based on this data.
                If needed, use the available tools to look up protein information.

                Instructions:
                - Be clear, concise, and scientific.
                - Do not make up facts.
                - Prefer short markdown-formatted answers.
                """

    

    prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
                HumanMessage(content="{user_input}")
            ])


        # Chat input
    user_input = st.text_input("💬 Ask a question about the selected protein sequence:")
    if user_input:
            with st.spinner("Analyzing with LLM..."):
                response, updated_history = get_llm_response(
                    llm,
                    prompt,
                    user_input,
                    st.session_state.chat_history,
                    tools=all_tools
                )
                st.session_state.chat_history = updated_history
                st.success("🧪 Response:")
                st.markdown(response)

        # Display chat history
    if st.session_state.chat_history:
            st.divider()
            st.subheader("💬 Chat History")
            for msg in st.session_state.chat_history:
                role = "👤 You" if isinstance(msg, HumanMessage) else "🤖 Assistant"
                st.markdown(f"**{role}:** {msg.content}")

        # Reset chat
    if st.button("Reset Chat ",key='reset_btn1'):
            st.session_state.chat_history = []
            st.info("Chat history has been reset!")


with tab3:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
