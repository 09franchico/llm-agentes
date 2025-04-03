from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_ollama.llms import OllamaLLM
from langchain.tools import BaseTool
import json
import os
from typing import List
import pandas as pd


def busca_dados_da_universidade(universidade: str):
    dados = pd.read_csv("documentos/universidades.csv")
    dados["NOME_FACULDADE"] = dados["NOME_FACULDADE"].str.lower()
    dados_com_essa_universidade = dados[dados["NOME_FACULDADE"] == universidade]
    if dados_com_essa_universidade.empty:
        return {}
    return dados_com_essa_universidade.iloc[:1].to_dict()


def busca_dados_das_universidades():
    dados = pd.read_csv("documentos/universidades.csv")
    return dados.to_dict()


class ExtratorDeUniversidade(BaseModel):
    universidade:str = Field("O nome da universidade em minúsculo.")


class DadosDeUniversidade(BaseTool):
    name:str = "DadosDeUniversidade"
    description:str = """Esta ferramenta extrai os dados de uma universidade.
Passe para essa ferramenta como argumento o nome da universidade."""

    def _run(self, input:str) -> str:
        # llm = ChatOpenAI(model="gpt-4o",
        #                  api_key=os.getenv("OPENAI_API_KEY"))
        # llm = ChatDeepSeek(
        #         model="deepseek-chat",
        #         temperature=0,
        #         api_key=os.getenv("DEEPSEEK_API_KEY")
                
        #     )
        llm  = OllamaLLM(model="gemma3:4b")
        parser = JsonOutputParser(pydantic_object=ExtratorDeUniversidade)
        template = PromptTemplate(template="""Você deve analisar a entrada a seguir e extrair o nome de universidade informada em minúsculo.
Entrada:
-----------------
{input}
-----------------
                        Formato de saída:
                        {formato_saida}""",
                        input_variables=["input"],
                        partial_variables={"formato_saida" : parser.get_format_instructions()})
        cadeia = template | llm | parser
        resposta = cadeia.invoke({"input" : input})
        universidade = resposta['universidade']
        universidade = universidade.lower().strip()
        dados = busca_dados_da_universidade(universidade)
        return json.dumps(dados)

class TodasUniversidades(BaseTool):
    name:str="TodasUniversidades"
    description:str="""Carrega os dados de todas as universidades. Não é necessário nenhum parâmetro de entrada."""
    
    def _run(self, input:str):
        universidades = busca_dados_das_universidades()
        return universidades