# AI Resume Analyzer and LinkedIn Scraper using Generative AI

---

**Live Streamlit Application:** [ResumeRadarAI on Streamlit](https://resumeradarai.streamlit.app/)

## Introduction

Developed an advanced AI application that leverages Retrieval-Augmented Generation (RAG), Large Language Models (LLM), and Groq’s LLaMA 3 model for comprehensive resume analysis. It excels at summarizing the resume, evaluating strengths, identifying weaknesses, and offering personalized improvement suggestions while also recommending the perfect job titles. Additionally, it seamlessly employs Selenium to extract vital LinkedIn data, including company names, job titles, locations, job URLs, and detailed job descriptions. This application simplifies the job-seeking journey by equipping users with comprehensive insights to elevate their career opportunities.

## Table of Contents

1. [Key Technologies and Skills](#key-technologies-and-skills)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Features](#features)

## Key Technologies and Skills

- Python
- NumPy
- Pandas
- LangChain
- Large Language Model (LLM)
- Retrieval-Augmented Generation (RAG)
- Groq
- Selenium
- Hugging Face
- Streamlit

## Installation

All dependencies are listed in `requirements.txt` — see step 2 under Usage below.

## Usage

To use this project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ALISAIF849/AI-Resume-Analyzer-LinkedIn-Scraper-main.git
   ```
2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure your Groq API key:**
   Copy `.env.example` to `.env` and set `GROQ_API_KEY=your_key_here` (get a free key at
   [console.groq.com/keys](https://console.groq.com/keys)).
4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
5. **Access the app in your browser at:**
   ```bash
   http://localhost:8501
   ```

## Features

### 1. Easy User Experience

- Resume Analyzer AI makes it easy for users — just upload your resume and go. The Groq API key is configured once
  server-side (via `.env`), so nothing sensitive is ever typed or shown in the browser.
- The application is designed to be user-friendly so that anyone can use its powerful resume analysis features.
- It also uses the `PyPDF2` library to quickly extract text from your uploaded resume, which is the first step in doing a thorough analysis.

### 2. Smart Text Analysis with Langchain

- It uses a smart method via the Langchain library to break long sections of text from resumes into smaller, meaningful chunks.
- This clever technique improves the accuracy of the resume analysis and gives users practical advice to enhance their job prospects.

### 3. Enhanced LLM Integration with FAISS and Groq

- Seamlessly connects to **Groq's high-performance inference API** using your Groq API key for ultra-fast, low-latency LLM interactions.
- FAISS (Facebook AI Similarity Search) converts both the text chunks and query data into numerical vectors, simplifying and speeding up the matching process.

### 4. Intelligent Chunk Selection in RAG and LLM

- Retrieves relevant resume chunks by comparing user queries with vector embeddings and selecting the top K based on similarity scores.
- Uses the **LLaMA 3.3 70B model** hosted on Groq to analyze and generate smart responses from these retrieved chunks.

### 5. Robust Question-Answering Pipeline

- Processes the top K retrieved documents along with the user query to generate meaningful and coherent answers.
- The LLM understands context and extracts the most relevant content, delivering precise and insightful feedback.

### 6. Comprehensive Resume Analysis

- **Summary:** Offers a quick, insightful overview of your resume including qualifications, key experience, skills, projects, and achievements.
- **Strength:** Highlights strong points in your resume based on qualifications, experience, and accomplishments.
- **Weakness:** Identifies areas of improvement and provides actionable suggestions to convert them into strengths.
- **Suggestion:** Recommends personalized job titles aligned with your resume to optimize your job search experience.

### 🔍 Selenium-Powered LinkedIn Data Scraping

Utilizing **Selenium** and a **Webdriver** automated test tool, this feature enables users to input job titles and automates the data scraping process from LinkedIn. The scraped data includes crucial details such as:

- Company names
- Job titles
- Locations
- Job URLs
- Detailed job descriptions

This streamlined process allows users to easily review job listings and apply for positions, simplifying their overall job search and application experience.
