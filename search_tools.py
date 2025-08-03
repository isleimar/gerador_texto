# (Cole aqui o código das classes DuckDuckGoSearchInput e DuckDuckGoSearchTool da resposta anterior)
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from duckduckgo_search import DDGS
import json

class DuckDuckGoSearchInput(BaseModel):
    """Define os argumentos para a ferramenta de busca DuckDuckGo."""
    query: str = Field(description="A consulta de busca que será usada no DuckDuckGo para encontrar informações.")

class DuckDuckGoSearchTool(BaseTool):
    name: str = "Busca na Internet com DuckDuckGo"
    description: str = "Uma ferramenta para realizar buscas na internet usando o DuckDuckGo e retornar os primeiros resultados. Útil para encontrar informações, notícias e dados recentes."
    args_schema: type[BaseModel] = DuckDuckGoSearchInput

    def _run(self, query: str) -> str:
        """Executa a lógica da ferramenta."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(keywords=query, max_results=5))
            
            if not results:
                return "Nenhum resultado encontrado para a busca."

            result_strings = []
            for i, result in enumerate(results):
                result_strings.append(
                    f"Resultado {i+1}:\n"
                    f"  Título: {result.get('title', 'N/A')}\n"
                    f"  Link: {result.get('href', 'N/A')}\n"
                    f"  Resumo: {result.get('body', 'N/A')}\n"
                    "-----------------"
                )
            return "\n".join(result_strings)
        except Exception as e:
            return f"Ocorreu um erro ao tentar usar a ferramenta de busca: {e}"