# CODESTAR-CODE Interactive Code Exploration Tool

CODESTAR is an innovative codebase chatbot designed to facilitate seamless interaction with your code. It caters to various user queries, whether related to code functionality, code explanation, or obtaining a high-level overview of the entire project. The tool harnesses the capabilities of the Mistral-7B Language Model (LLM) and employs a user-friendly chat interface created with Streamlit.

## Key Features

1. **GitHub Repository Interaction:** Users can effortlessly engage with their codebase by simply entering the name of their GitHub repository. Note that the repository should be public.

2. **Automated Repository Processing:** The tool automatically clones the specified repository, divides it into manageable chunks, and embeds them for efficient interaction.

3. **Langchain Integration:** The powerful Langchain tool is employed to construct a Question-Answer (QA) retriever. This enables users to engage in a conversational format with their codebase, seeking answers to their queries.

4. **Streamlined Chat Interface:** The chat interface provides a user-friendly environment where users can ask questions and have interactive sessions with their codebase.

5. **Code Functionality Exploration:** Users can inquire about the functioning of specific portions of the code, seek explanations, and obtain a comprehensive understanding of the project.

### Usage

To use this codebase chatbot, follow these steps:

1. Clone the repository:

`git clone https://github.com/example/repository.git`

2. Install the required dependencies:

`pip install -r requirements.txt`

3. Run the Streamlit app:

`streamlit run gui.py`

Access the chat interface by opening your web browser and navigating to http://localhost:8501.

Enter the name of your GitHub repository in the provided input fields.

The codebase will be chunked and embedded, and the chat interface will be displayed.

Ask questions or provide instructions using natural language, and the chatbot will respond accordingly.

### Limitations and Considerations

- The functionality of the codebase chatbot is contingent upon the Mistral 7B Language Model and its inherent capabilities.
- While opting for a more potent Language Model (LLM) is possible, it comes at the cost of increased resource utilization.
- Processing large codebases or repositories with intricate structures may result in longer chunking and embedding times.
- The accuracy and quality of responses are intricately tied to the precision of the language model and the effectiveness of code embeddings.

### Future Improvements

- Integrate with external tools and services to provide more advanced codebase analysis and insights.
