# app.py (merged and conflict-resolved)
import time
import platform
import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains.question_answering import load_qa_chain

# Selenium imports (combined, robust)
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables from .env file (once)
load_dotenv()
groq_env_api_key = os.getenv("GROQ_API_KEY", "")


def add_vertical_space(num_lines: int = 1) -> None:
    """Vertical spacing helper (streamlit_extras.add_vertical_space is deprecated in favor of st.space)."""
    st.space(num_lines * 24)


# -------------- Streamlit base config --------------
def streamlit_config():
    """Configure Streamlit page, theme, and header/title (cake.me-inspired: light, teal-accented)."""
    st.set_page_config(page_title='Resume Analyzer AI', page_icon='📄', layout="wide")

    custom_style = """
    <style>
    /* Suppress the browser's native password reveal icon (Edge/Chromium) */
    input::-ms-reveal, input::-ms-clear { display: none; }

    [data-testid="stHeader"] { background: rgba(0,0,0,0); }

    /* Light background, dark readable text */
    [data-testid="stAppViewContainer"] {
        background: #FAFBFC;
    }
    [data-testid="stAppViewContainer"] * {
        color: #1F2933;
    }

    /* Hero */
    .hero-wrap {
        text-align: center;
        padding: 2.2rem 1rem 1.6rem 1rem;
        background: radial-gradient(circle at 50% 0%, rgba(20,184,166,0.10), rgba(250,251,252,0) 70%);
        border-radius: 20px;
        margin-bottom: 1rem;
    }
    .app-title {
        font-size: 2.75rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.3rem;
    }
    .app-title .accent { color: #14B8A6; }
    .app-subtitle {
        color: #5B6472;
        font-size: 1.1rem;
        max-width: 620px;
        margin: 0 auto;
    }

    /* Feature strip */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0 0.5rem 0;
    }
    .feature-card {
        background: #FFFFFF;
        border: 1px solid #E7EBEF;
        border-radius: 14px;
        padding: 1.1rem 1rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(15,23,42,0.04);
    }
    .feature-card .icon { font-size: 1.6rem; margin-bottom: 0.4rem; }
    .feature-card .title { font-weight: 700; color: #0F172A; font-size: 0.95rem; margin-bottom: 0.2rem; }
    .feature-card .desc { color: #6B7280; font-size: 0.82rem; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E7EBEF;
    }
    [data-testid="stSidebar"] * { color: #1F2933; }

    /* Card-style treatment for forms, expanders, bordered containers */
    [data-testid="stForm"],
    [data-testid="stExpander"],
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border: 1px solid #E7EBEF;
        border-radius: 16px;
        padding: 0.5rem 1rem;
        box-shadow: 0 2px 10px rgba(15,23,42,0.04);
    }

    /* Buttons: teal, rounded, cake.me-style CTA */
    .stButton > button, .stFormSubmitButton > button {
        background: #14B8A6;
        color: #FFFFFF;
        font-weight: 700;
        border: none;
        border-radius: 999px;
        padding: 0.6rem 1.6rem;
        transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: #0F9E8E;
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(20,184,166,0.35);
        color: #FFFFFF;
    }

    /* Accent focus ring on inputs */
    input:focus, textarea:focus {
        border-color: #14B8A6 !important;
        box-shadow: 0 0 0 1px #14B8A6 !important;
    }
    </style>
    """
    st.markdown(custom_style, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="app-title">📄 <span class="accent">Resume</span> Analyzer AI</div>
            <div class="app-subtitle">Optimize your resume in seconds — get instant AI feedback,
            ATS scoring, and matching job listings powered by Groq LLaMA 3</div>
            <div class="feature-grid">
                <div class="feature-card"><div class="icon">🧠</div><div class="title">Smart Summary</div><div class="desc">Instant AI breakdown of your resume</div></div>
                <div class="feature-card"><div class="icon">💪</div><div class="title">Strengths &amp; Gaps</div><div class="desc">Know what to fix before recruiters do</div></div>
                <div class="feature-card"><div class="icon">✅</div><div class="title">ATS Check</div><div class="desc">Match score against any job description</div></div>
                <div class="feature-card"><div class="icon">🔗</div><div class="title">LinkedIn Jobs</div><div class="desc">Find matching roles automatically</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def init_session_state():
    """Ensure required session_state keys exist exactly once per browser session until refresh."""
    defaults = {
        "pdf": None,               # Uploaded file object
        "groq_api_key": groq_env_api_key,  # API key, sourced from GROQ_API_KEY in .env
        "pdf_chunks": None,        # Cached chunks from resume
        "summary_cache": None,     # Cached summary text
        "ats_score_result": None,
        "cover_letter_result": None,
        "interview_q_result": None,
        "suggested_linkedin_titles": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# -------------- Resume Analyzer --------------
class resume_analyzer:

    @staticmethod
    def pdf_to_chunks(pdf):
        """Read PDF and split into text chunks for embeddings / LLM processing."""
        pdf_reader = PdfReader(pdf)

        # extract text from each page separately
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        # Split the long text into small chunks.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=200,
            length_function=len)

        chunks = text_splitter.split_text(text=text)
        return chunks

    @staticmethod
    def groq_llm(groq_api_key, chunks, analyze):
        """Use embeddings + FAISS + Groq LLM to run a QA-like analysis on chunks."""
        # Step 1: Use HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Step 2: Create FAISS vector store from text chunks
        vectorstores = FAISS.from_texts(chunks, embedding=embeddings)

        # Step 3: Perform similarity search for top-k relevant chunks
        docs = vectorstores.similarity_search(query=analyze, k=3)

        # Step 4: Use Groq's LLaMA for question answering
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            api_key=groq_api_key
        )

        # Step 5: Load QA chain and run it
        chain = load_qa_chain(llm=llm, chain_type='stuff')
        response = chain.run(input_documents=docs, question=analyze)

        return response

    # ----- prompt helpers -----
    @staticmethod
    def summary_prompt(query_with_chunks):
        query = f'''Need a detailed summarization of the following resume and a concluding section:

{query_with_chunks}
'''
        return query

    @staticmethod
    def strength_prompt(query_with_chunks):
        query = f'''Provide a detailed analysis of the strengths of the resume below and conclude:

{query_with_chunks}
'''
        return query

    @staticmethod
    def weakness_prompt(query_with_chunks):
        query = f'''Provide a detailed analysis of the weaknesses of the resume below and practical improvements to make it better:

{query_with_chunks}
'''
        return query

    @staticmethod
    def job_title_prompt(query_with_chunks):
        query = f'''List suitable job roles (LinkedIn-style titles) based on the resume below:

{query_with_chunks}
'''
        return query

    @staticmethod
    def linkedin_search_titles_prompt(resume_summary):
        query = f'''Based on the resume summary below, output ONLY 3 to 5 concise job titles this candidate
should search for on LinkedIn, separated by commas, with no numbering, no bullet points, no explanation,
and no extra text of any kind — just the comma-separated titles. Example output format:
Software Engineer, Backend Developer, Python Developer

Resume Summary:
{resume_summary}
'''
        return query

    @staticmethod
    def ats_check_prompt(resume_summary, job_description):
        query = f"""
        You are a strict, no-nonsense ATS (Applicant Tracking System) auditor and technical recruiter with 15 years
        of experience rejecting unqualified resumes. You are known for being harsh and literal — you do NOT give
        credit for "adjacent" skills, generic enthusiasm, or vague claims. A resume only scores well if it
        demonstrably satisfies what the job description explicitly asks for.

        Score strictly using this rubric — do not default to a "safe" middle score:
        - 0-30%: Resume is missing most required skills, tools, and experience level from the job description.
        - 31-50%: Resume shares the general field but is missing several explicitly required skills/qualifications.
        - 51-70%: Resume covers roughly half the explicit requirements; notable gaps remain in key skills or seniority.
        - 71-85%: Resume covers most explicit requirements with only minor gaps.
        - 86-100%: Resume near-exhaustively matches the required skills, tools, experience level, and domain.

        Rules:
        - Count exact and near-exact keyword/skill/tool matches between the job description and resume. Do not
          assume the candidate has a skill unless it is stated in the resume.
        - Penalize missing years-of-experience requirements, missing required tools/technologies, and missing
          domain-specific qualifications explicitly named in the job description.
        - Do not round up out of politeness. Most resumes that were not written specifically for this job
          description should NOT score above 70%.

        Analyze the following resume summary and job description, then provide:

        1.  **Overall Match Score:** A single percentage reflecting strict rubric-based scoring above.
        2.  **Score Justification:** Explicitly state which rubric band was used and why.
        3.  **Summary of the Match:** A concise, honest paragraph on fit — call out weaknesses plainly.
        4.  **Missing Keywords:** Every important keyword/skill/tool from the job description absent from the resume.
        5.  **Actionable Improvements:** 3-5 specific, actionable suggestions to close the gaps found above.

        ---
        Resume Summary:
        {resume_summary}

        ---
        Job Description:
        {job_description}

        """
        return query

    @staticmethod
    def cover_letter_prompt(resume_summary, job_description):
        query = f"""
        You are a professional cover letter writer. Using the resume summary and job description provided, write a compelling and personalized cover letter.

        The letter should:
        - Be no more than one page.
        - Start with a strong introduction stating the position you're applying for.
        - Highlight relevant skills and experiences from the resume that match the job description.
        - Clearly explain why the candidate is a perfect fit for this specific role and company.
        - End with a professional call-to-action.

        ---
        Resume Summary:
        {resume_summary}

        ---
        Job Description:
        {job_description}
        """
        return query

    @staticmethod
    def interview_questions_prompt(job_description):
        query = f"""
        You are an expert career coach. Based on the job description provided, generate a list of 5-7 preparatory interview questions.

        Include a mix of:
        - Common questions (e.g., "Tell me about yourself").
        - Technical questions related to the skills listed.
        - Behavioral questions (e.g., "Tell me about a time when...").
        - Questions to ask the interviewer.

        Format the output clearly with numbered points.

        ---
        Job Description:
        {job_description}
        """
        return query


# -------------- LinkedIn Scraper --------------
class linkedin_scraper:

    @staticmethod
    def _get_chrome_options(headless=True):
        """Return configured Chrome Options suitable for local and server deployments."""
        options = Options()

        # Headless if requested (Streamlit cloud / servers need headless)
        if headless:
            # new headless can be passed as flag if supported; fallback to classic flag
            try:
                options.add_argument("--headless=new")
            except Exception:
                options.add_argument("--headless")
        else:
            # if not headless, start maximized so UI is usable
            options.add_argument("--start-maximized")

        # Common options for servers and local
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-notifications")
        options.add_argument("--window-size=1920,1080")
        # Prevent Chrome from trying to use automation extension (less detectable)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        # If user provided explicit chrome binary path via env var, use it.
        chrome_bin_env = os.environ.get("CHROME_BINARY") or os.environ.get("CHROME_BIN")
        if chrome_bin_env:
            options.binary_location = chrome_bin_env
        else:
            # Try common Linux / Mac / Windows paths
            if platform.system() == "Linux":
                for p in ["/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome", "/usr/bin/headless-chrome"]:
                    if os.path.exists(p):
                        options.binary_location = p
                        break
            elif platform.system() == "Darwin":
                # macOS default path
                possible = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
                if os.path.exists(possible):
                    options.binary_location = possible
            else:
                # Windows typically uses system-installed Chrome; no change required.
                pass

        return options

    @staticmethod
    def webdriver_setup(headless=True, fallback_local_driver_path=None):
        """
        Initialize a Chrome WebDriver using webdriver-manager. If that fails and a fallback
        local driver path is provided, try using that.
        """
        options = linkedin_scraper._get_chrome_options(headless=headless)

        # Try to auto-install correct chromedriver
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            return driver
        except WebDriverException as e:
            # If automatic install fails, try fallback path if provided
            if fallback_local_driver_path and os.path.exists(fallback_local_driver_path):
                try:
                    service = Service(fallback_local_driver_path)
                    driver = webdriver.Chrome(service=service, options=options)
                    return driver
                except Exception as e2:
                    raise RuntimeError(f"Failed launching chromedriver from fallback path: {e2}") from e2
            # Re-raise with helpful message
            raise RuntimeError(
                "Failed to start Chrome WebDriver via webdriver-manager. "
                "If you're on a server, ensure Chrome/Chromium is installed, or set CHROME_BINARY env var. "
                f"Original error: {e}"
            ) from e

    @staticmethod
    def _open_link_wait(driver, link, timeout=15):
        """Open link and wait until a LinkedIn jobs page marker appears. Raises TimeoutError if not found."""
        start = time.time()
        while True:
            try:
                driver.get(link)
                driver.implicitly_wait(5)
                time.sleep(2)
                driver.find_element(by=By.CSS_SELECTOR, value='span.switcher-tabs__placeholder-text.m-auto')
                return
            except NoSuchElementException:
                if (time.time() - start) > timeout:
                    raise TimeoutError("Timed out waiting for LinkedIn jobs page to load (CSS selector not found).")
                time.sleep(1)
                continue

    @staticmethod
    def build_url(job_title, job_location):
        b = []
        for i in job_title:
            x = i.split()
            y = '%20'.join(x)
            b.append(y)
        job_title_param = '%2C%20'.join(b) if b else ''
        link = (
            f"https://in.linkedin.com/jobs/search?keywords={job_title_param}"
            f"&location={job_location}&locationId=&geoId=102713980&f_TPR=r604800&position=1&pageNum=0"
        )
        return link

    @staticmethod
    def link_open_scrolldown(driver, link, job_count):
        linkedin_scraper._open_link_wait(driver, link)
        # Scroll generously regardless of job_count so enough candidates are loaded
        # for title/location filtering to still leave job_count matches afterward.
        scroll_iterations = max(job_count, 5)
        for _ in range(0, scroll_iterations):
            body = driver.find_element(by=By.TAG_NAME, value='body')
            body.send_keys(Keys.PAGE_UP)
            try:
                driver.find_element(
                    by=By.CSS_SELECTOR,
                    value="button[data-tracking-control-name='public_jobs_contextual-sign-in-modal_modal_dismiss']>icon>svg"
                ).click()
            except Exception:
                pass
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            driver.implicitly_wait(2)
            try:
                driver.find_element(by=By.CSS_SELECTOR, value="button[aria-label='See more jobs']").click()
                driver.implicitly_wait(5)
            except Exception:
                pass

    @staticmethod
    def job_title_filter(scrap_job_title, user_job_title_input):
        user_input = [i.lower().strip() for i in user_job_title_input]
        scrap_title = [i.lower().strip() for i in [scrap_job_title]]
        confirmation_count = 0
        for i in user_input:
            if all(j in scrap_title[0] for j in i.split()):
                confirmation_count += 1
        return scrap_job_title if confirmation_count > 0 else np.nan

    @staticmethod
    def scrap_company_data(driver, job_title_input, job_location):
        company = driver.find_elements(by=By.CSS_SELECTOR, value='h4[class="base-search-card__subtitle"]')
        company_name = [i.text for i in company]

        location = driver.find_elements(by=By.CSS_SELECTOR, value='span[class="job-search-card__location"]')
        company_location = [i.text for i in location]

        title = driver.find_elements(by=By.CSS_SELECTOR, value='h3[class="base-search-card__title"]')
        job_title = [i.text for i in title]

        url = driver.find_elements(by=By.CSS_SELECTOR, value='a.base-card__full-link')
        website_url = [i.get_attribute('href') for i in url]

        df = pd.DataFrame(company_name, columns=['Company Name'])
        df['Job Title'] = pd.DataFrame(job_title)
        df['Location'] = pd.DataFrame(company_location)
        df['Website URL'] = pd.DataFrame(website_url)

        df['Job Title'] = df['Job Title'].apply(lambda x: linkedin_scraper.job_title_filter(x, job_title_input))
        df['Location'] = df['Location'].apply(lambda x: x if job_location.lower() in x.lower() else np.nan)

        df = df.dropna()
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def scrap_job_description(driver, df, job_count):
        website_url = df['Website URL'].tolist()
        job_description = []
        description_count = 0

        for i in range(0, len(website_url)):
            try:
                linkedin_scraper._open_link_wait(driver, website_url[i])
                try:
                    driver.find_element(
                        by=By.CSS_SELECTOR,
                        value='button[data-tracking-control-name="public_jobs_show-more-html-btn"]'
                    ).click()
                except Exception:
                    pass

                driver.implicitly_wait(5)
                time.sleep(1)
                description = driver.find_elements(
                    by=By.CSS_SELECTOR,
                    value='div[class*="show-more-less-html__markup"]'
                )
                data = [elem.text for elem in description][0] if description else ''
                if len(data.strip()) > 0 and data not in job_description:
                    job_description.append(data)
                    description_count += 1
                else:
                    job_description.append('Description Not Available')
            except Exception:
                job_description.append('Description Not Available')

            if description_count == job_count:
                break

        df = df.iloc[:len(job_description), :]
        df['Job Description'] = pd.DataFrame(job_description, columns=['Description'])
        df['Job Description'] = df['Job Description'].apply(lambda x: np.nan if x == 'Description Not Available' else x)
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def display_data_userinterface(df_final):
        add_vertical_space(1)
        if len(df_final) > 0:
            for i in range(0, len(df_final)):
                st.markdown(f'<h3 style="color:#14B8A6;">Job Posting Details : {i+1}</h3>', unsafe_allow_html=True)
                st.write(f"Company Name : {df_final.iloc[i,0]}")
                st.write(f"Job Title    : {df_final.iloc[i,1]}")
                st.write(f"Location     : {df_final.iloc[i,2]}")
                st.write(f"Website URL  : {df_final.iloc[i,3]}")
                with st.expander(label='Job Desription'):
                    st.write(df_final.iloc[i, 4])
                add_vertical_space(3)
        else:
            st.markdown('<h5 style="text-align: center;color:#14B8A6;">No Matching Jobs Found</h5>', unsafe_allow_html=True)

    @staticmethod
    def main():
        driver = None
        try:
            job_title_input, job_location, job_count, submit = linkedin_scraper.get_userinput()
            add_vertical_space(2)
            if submit:
                if job_title_input != [] and job_location != '':
                    with st.spinner('Chrome Webdriver Setup Initializing...'):
                        driver = linkedin_scraper.webdriver_setup(headless=True)

                    with st.spinner('Loading More Job Listings...'):
                        link = linkedin_scraper.build_url(job_title_input, job_location)
                        linkedin_scraper.link_open_scrolldown(driver, link, job_count)

                    with st.spinner('Scraping Job Details...'):
                        df = linkedin_scraper.scrap_company_data(driver, job_title_input, job_location)
                        df_final = linkedin_scraper.scrap_job_description(driver, df, job_count)

                    linkedin_scraper.display_data_userinterface(df_final)

                elif job_title_input == []:
                    st.markdown('<h5 style="text-align: center;color:#14B8A6;">Job Title is Empty</h5>', unsafe_allow_html=True)
                elif job_location == '':
                    st.markdown('<h5 style="text-align: center;color:#14B8A6;">Job Location is Empty</h5>', unsafe_allow_html=True)
        except Exception as e:
            add_vertical_space(2)
            st.markdown(f'<h5 style="text-align: center;color:#14B8A6;">{e}</h5>', unsafe_allow_html=True)
        finally:
            if driver:
                driver.quit()

    @staticmethod
    def get_userinput():
        add_vertical_space(2)

        # Suggest job titles from the uploaded resume, if available, so the search
        # is grounded in the candidate's actual background rather than a blank guess.
        has_resume_context = (
            st.session_state.get("summary_cache")
            and st.session_state.get("groq_api_key", "").strip()
            and st.session_state.get("pdf_chunks")
        )
        if has_resume_context and not st.session_state.get("suggested_linkedin_titles"):
            with st.spinner("Suggesting job titles from your resume..."):
                titles_prompt = resume_analyzer.linkedin_search_titles_prompt(st.session_state.summary_cache)
                st.session_state.suggested_linkedin_titles = resume_analyzer.groq_llm(
                    st.session_state.groq_api_key,
                    st.session_state.pdf_chunks,
                    titles_prompt
                ).strip()

        default_title = st.session_state.get("suggested_linkedin_titles") or ""
        if default_title:
            st.caption(f"Suggested from your resume: {default_title} — edit below if you want to search something else.")
        elif not has_resume_context:
            st.caption("Tip: upload your resume on the Summary page first to get job title suggestions here.")

        with st.form(key='linkedin_scarp'):
            add_vertical_space(1)
            col1, col2, col3 = st.columns([0.5, 0.3, 0.2], gap='medium')
            with col1:
                job_title_input = st.text_input(label='Job Title', value=default_title)
                job_title_input = job_title_input.split(',') if job_title_input else []
            with col2:
                job_location = st.text_input(label='Job Location', value='India')
            with col3:
                job_count = st.number_input(label='Job Count', min_value=1, value=5, step=1)

            add_vertical_space(1)
            submit = st.form_submit_button(label='Submit')
            add_vertical_space(1)
        return job_title_input, job_location, job_count, submit


# -------------- Helper: ensure inputs exist before running --------------
def require_resume_and_key():
    """Return (ok_bool). If False, show a consistent warning."""
    pdf = st.session_state.get("pdf")
    groq_api_key = st.session_state.get("groq_api_key", "")
    if not groq_api_key.strip():
        st.warning("This feature isn't available right now. Please try again later.")
        return False
    if pdf is None:
        st.warning("Please upload your resume on the **Summary** page first.")
        return False
    return True


# -------------- App Pages --------------
streamlit_config()
init_session_state()
add_vertical_space(2)

with st.sidebar:
    add_vertical_space(2)
    st.markdown(
        '<div style="text-align:center; font-weight:700; font-size:1.1rem; color:#14B8A6; '
        'margin-bottom:0.75rem;">🧭 Navigation</div>',
        unsafe_allow_html=True
    )
    # include ATS Check option
    option = option_menu(
        menu_title='',
        options=['Summary', 'Strength', 'Weakness', 'Job Titles', 'ATS Check', 'Linkedin Jobs'],
        icons=['house-fill', 'database-fill', 'pass-fill', 'list-ul', 'clipboard-check-fill', 'linkedin'],
        styles={
            "container": {"padding": "0!important", "background-color": "#FFFFFF"},
            "icon": {"color": "#14B8A6", "font-size": "17px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "4px 0",
                "border-radius": "10px",
                "color": "#1F2933",
                "background-color": "#FFFFFF",
                "--hover-color": "#EAF7F5",
            },
            "nav-link-selected": {"background-color": "#14B8A6", "color": "#FFFFFF", "font-weight": "700"},
        }
    )


# ===== SUMMARY PAGE =====
if option == 'Summary':
    if not st.session_state.groq_api_key.strip():
        st.warning("This feature isn't available right now. Please try again later.")

    with st.form(key='Summary'):
        add_vertical_space(1)
        pdf = st.file_uploader(label='Upload Your Resume', type='pdf')
        add_vertical_space(2)
        submit = st.form_submit_button(label='Submit')

    if submit:
        if pdf is not None and st.session_state.groq_api_key.strip() != "":
            # Persist to session
            st.session_state.pdf = pdf

            # Build chunks once and cache
            with st.spinner('Reading and splitting your resume...'):
                st.session_state.pdf_chunks = resume_analyzer.pdf_to_chunks(pdf)

            # Build summary once and cache
            with st.spinner('Generating summary...'):
                summary_prompt = resume_analyzer.summary_prompt(st.session_state.pdf_chunks)
                st.session_state.summary_cache = resume_analyzer.groq_llm(
                    st.session_state.groq_api_key,
                    st.session_state.pdf_chunks,
                    summary_prompt
                )

            st.success("Saved! You can switch to other pages without re-uploading.")
            st.markdown('<h4 style="color:#14B8A6;">Summary:</h4>', unsafe_allow_html=True)
            st.write(st.session_state.summary_cache)

        elif pdf is None:
            st.warning("Please upload your resume.")
        else:
            st.warning("This feature isn't available right now. Please try again later.")

    # If already stored earlier, show a small status and (optionally) the last summary
    if (st.session_state.pdf is not None) and st.session_state.groq_api_key.strip():
        st.info("Resume already saved in this session. Switch tabs freely.")
        if st.session_state.summary_cache:
            with st.expander("Last Generated Summary"):
                st.write(st.session_state.summary_cache)


# ===== STRENGTH PAGE =====
elif option == 'Strength':
    if require_resume_and_key():
        # Ensure chunks exist
        if st.session_state.pdf_chunks is None:
            with st.spinner('Reading and splitting your resume...'):
                st.session_state.pdf_chunks = resume_analyzer.pdf_to_chunks(st.session_state.pdf)

        # Ensure base summary exists
        if st.session_state.summary_cache is None:
            with st.spinner('Generating base summary...'):
                base_summary_prompt = resume_analyzer.summary_prompt(st.session_state.pdf_chunks)
                st.session_state.summary_cache = resume_analyzer.groq_llm(
                    st.session_state.groq_api_key,
                    st.session_state.pdf_chunks,
                    base_summary_prompt
                )

        with st.spinner('Analyzing strengths...'):
            strength_prompt = resume_analyzer.strength_prompt(st.session_state.summary_cache)
            strength = resume_analyzer.groq_llm(
                st.session_state.groq_api_key,
                st.session_state.pdf_chunks,
                strength_prompt
            )

        st.markdown('<h4 style="color:#14B8A6;">Strength:</h4>', unsafe_allow_html=True)
        st.write(strength)


# ===== WEAKNESS PAGE =====
elif option == 'Weakness':
    if require_resume_and_key():
        if st.session_state.pdf_chunks is None:
            with st.spinner('Reading and splitting your resume...'):
                st.session_state.pdf_chunks = resume_analyzer.pdf_to_chunks(st.session_state.pdf)

        if st.session_state.summary_cache is None:
            with st.spinner('Generating base summary...'):
                base_summary_prompt = resume_analyzer.summary_prompt(st.session_state.pdf_chunks)
                st.session_state.summary_cache = resume_analyzer.groq_llm(
                    st.session_state.groq_api_key,
                    st.session_state.pdf_chunks,
                    base_summary_prompt
                )

        with st.spinner('Analyzing weaknesses...'):
            weakness_prompt = resume_analyzer.weakness_prompt(st.session_state.summary_cache)
            weakness = resume_analyzer.groq_llm(
                st.session_state.groq_api_key,
                st.session_state.pdf_chunks,
                weakness_prompt
            )

        st.markdown('<h4 style="color:#14B8A6;">Weakness and Suggestions:</h4>', unsafe_allow_html=True)
        st.write(weakness)


# ===== JOB TITLES PAGE =====
elif option == 'Job Titles':
    if require_resume_and_key():
        if st.session_state.pdf_chunks is None:
            with st.spinner('Reading and splitting your resume...'):
                st.session_state.pdf_chunks = resume_analyzer.pdf_to_chunks(st.session_state.pdf)

        if st.session_state.summary_cache is None:
            with st.spinner('Generating base summary...'):
                base_summary_prompt = resume_analyzer.summary_prompt(st.session_state.pdf_chunks)
                st.session_state.summary_cache = resume_analyzer.groq_llm(
                    st.session_state.groq_api_key,
                    st.session_state.pdf_chunks,
                    base_summary_prompt
                )

        with st.spinner('Finding job titles...'):
            job_title_prompt = resume_analyzer.job_title_prompt(st.session_state.summary_cache)
            job_title = resume_analyzer.groq_llm(
                st.session_state.groq_api_key,
                st.session_state.pdf_chunks,
                job_title_prompt
            )

        st.markdown('<h4 style="color:#14B8A6;">Job Titles:</h4>', unsafe_allow_html=True)
        st.write(job_title)


# ===== LINKEDIN JOBS PAGE =====
elif option == 'Linkedin Jobs':
    linkedin_scraper.main()


# ===== ATS Check ===========
elif option == 'ATS Check':
    if not st.session_state.groq_api_key:
        st.warning("This feature isn't available right now. Please try again later.")
    elif st.session_state.pdf:
        st.markdown('<h4 style="color:#14B8A6;">Resume ATS Check</h4>', unsafe_allow_html=True)
        job_description = st.text_area(label="Paste the Job Description here", height=300)

        col1, col2, col3 = st.columns(3)
        with col1:
            submit_ats = st.button("Check ATS Score")
        with col2:
            submit_cover_letter = st.button("Generate Cover Letter")
        with col3:
            submit_interview_q = st.button("Generate Interview Questions")

        # --- Button Logic to Store Results in Session State ---
        if submit_ats and job_description.strip() != "":
            with st.spinner('Analyzing for ATS score...'):
                ats_check_result = resume_analyzer.ats_check_prompt(st.session_state.summary_cache, job_description)
                st.session_state.ats_score_result = resume_analyzer.groq_llm(
                    st.session_state.groq_api_key,
                    st.session_state.pdf_chunks,
                    ats_check_result
                )
        elif submit_ats and job_description.strip() == "":
            st.warning("Please paste a job description to check your ATS score.")

        if submit_cover_letter and job_description.strip() != "":
            with st.spinner('Generating your cover letter...'):
                cover_letter_prompt = resume_analyzer.cover_letter_prompt(
                    st.session_state.summary_cache,
                    job_description
                )
                st.session_state.cover_letter_result = resume_analyzer.groq_llm(
                    st.session_state.groq_api_key,
                    st.session_state.pdf_chunks,
                    cover_letter_prompt
                )
        elif submit_cover_letter and job_description.strip() == "":
            st.warning("Please paste a job description to generate a cover letter.")

        if submit_interview_q and job_description.strip() != "":
            with st.spinner('Generating interview questions...'):
                interview_q_prompt = resume_analyzer.interview_questions_prompt(job_description)
                st.session_state.interview_q_result = resume_analyzer.groq_llm(
                    st.session_state.groq_api_key,
                    st.session_state.pdf_chunks,
                    interview_q_prompt
                )
        elif submit_interview_q and job_description.strip() == "":
            st.warning("Please paste a job description to generate interview questions.")

        # --- Conditional Display of Results ---
        if st.session_state.ats_score_result:
            st.write(st.session_state.ats_score_result)
        if st.session_state.cover_letter_result:
            st.subheader("Generated Cover Letter")
            st.write(st.session_state.cover_letter_result)
        if st.session_state.interview_q_result:
            st.subheader("Preparatory Interview Questions")
            st.write(st.session_state.interview_q_result)

    else:
        st.warning("Please upload your resume on the **Summary** page.")
