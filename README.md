Helping meterial 

1. https://python.langchain.com/docs/how_to/custom_tools/#creating-tools-from-functions ----> langchian tools 


# 🧬 Protein Science AI Chatbot

An intelligent bioinformatics assistant that analyzes protein sequences using AI and biological databases.

![App Screenshot](assets/app_demo.gif) <!-- Add your screenshot/gif path -->

## 🌟 Features

### Core Functionality
- **Sequence Analysis**
  - FASTA file parsing
  - Domain/motif detection
  - Molecular weight calculation
  - Disulfide bond prediction

### Integrated Knowledge
| Source | Coverage | Tool Used |
|--------|----------|-----------|
| Wikipedia | General protein knowledge | `WikipediaQueryRun` |
| PubMed | Scientific literature | `TavilySearch` |
| STRING DB | Protein interactions | Custom API Tool |
| CATH/InterPro | Domain architecture | `search_cath()` |

### AI Capabilities
- Groq/Llama3 for explanations
- Anti-hallucination prompt engineering
- Context-aware responses
- Tool selection automation

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ProteinSequenceChatbot.git
cd ProteinSequenceChatbot