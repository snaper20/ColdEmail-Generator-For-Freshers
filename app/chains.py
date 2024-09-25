import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links, name,year,college,department):
        prompt_email = PromptTemplate.from_template(
            """
          ### JOB DESCRIPTION:
        {job_description}
    
        ### INSTRUCTION:
        You are {name}, a fresher graduating in {year}, from {college} in {department}. You are applying for the job posting mentioned above.
        Your goal is to write a cold email to the HR department, showcasing your skills, academic background, and enthusiasm for the role.
        Highlight your relevant projects and achievements to demonstrate your suitability for the position.
        Also, include the most relevant ones from the following portfolio links and experiences to showcase your capabilities: {link_list}.
        Remember to focus on how your experiences align with the specific job requirements.
        Do not provide a preamble.
    
        ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "name":name,"link_list": links,"year":year,"college":college,"department":department})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))