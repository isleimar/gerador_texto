# gerador_de_conteudo.py
import os
import yaml
import json
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool
# Supondo que search_tools.py exista no mesmo diretório
from search_tools import DuckDuckGoSearchTool 

from langchain_openai import ChatOpenAI

# Carrega as variáveis de ambiente do arquivo .env no início
load_dotenv()

class GeradorDeRoteiro:
    """
    Uma classe para encapsular a lógica de geração de roteiros longos usando CrewAI.
    """
    def __init__(self, config_path='config/crew_config.yaml'):
        """
        Inicializa o gerador, configurando o LLM, as ferramentas e carregando os 
        agentes e tarefas a partir de um arquivo de configuração YAML.
        """
        print("Inicializando o Gerador de Roteiro...")
        self._configurar_llm()
        self._carregar_ferramentas()
        self._carregar_configuracao_crew(config_path)
        print("Gerador pronto para uso.")

    def _configurar_llm(self):
        """Configura a instância do LLM com base nas variáveis de ambiente."""
        llm_model = os.getenv("LLM_MODEL", "llama3:8b")
        llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        llm_temperature = float(os.getenv("LLM_TEMPERATURE", 0.7))
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "ollama")

        self.llm = ChatOpenAI(
            model=llm_model,
            base_url=llm_base_url,
            temperature=llm_temperature
        )

    def _carregar_ferramentas(self):
        """Inicializa e mapeia as ferramentas disponíveis."""
        self.tools_map = {
            'search_tool': DuckDuckGoSearchTool(),
            'scrape_tool': ScrapeWebsiteTool()
        }

    def _carregar_configuracao_crew(self, config_path):
        """Carrega e cria dinamicamente agentes e tarefas do arquivo YAML."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                agents_config = config['agents']
                tasks_config = config['tasks']
        except FileNotFoundError:
            raise Exception(f"Erro: Arquivo de configuração '{config_path}' não encontrado.")
        except Exception as e:
            raise Exception(f"Erro ao ler o arquivo '{config_path}': {e}")

        # Criação dinâmica de agentes
        self.agents = {}
        for name, cfg in agents_config.items():
            self.agents[name] = Agent(
                role=cfg['role'],
                goal=cfg['goal'],
                backstory=cfg['backstory'],
                tools=[self.tools_map[t] for t in cfg.get('tools', [])],
                verbose=cfg.get('verbose', False), # Desligado por padrão no modo módulo
                llm=self.llm,
                allow_delegation=False,
                output_json=cfg.get('output_json', False)
            )

        # Criação dinâmica de tarefas (sem contexto ainda)
        self.tasks_config = tasks_config

    def _criar_tarefa_com_contexto(self, task_name, task_map_context):
        """Cria uma instância de tarefa resolvendo suas dependências de contexto."""
        cfg = self.tasks_config[task_name]
        agent = self.agents[cfg['agent']]
        
        context_tasks = []
        if 'context' in cfg and cfg['context'] is not None:
            for context_task_name in cfg['context']:
                if context_task_name in task_map_context:
                    context_tasks.append(task_map_context[context_task_name])
        
        return Task(
            description=cfg['description'],
            agent=agent,
            expected_output=cfg['expected_output'],
            verbose=cfg.get('verbose', False), # Desligado por padrão no modo módulo
            context=context_tasks
        )

    def gerar_roteiro_longo(self, categoria: str):
        """
        Orquestra o processo completo de geração de roteiro em 3 fases.

        Args:
            categoria (str): O assunto principal para a geração do conteúdo.

        Returns:
            str: O roteiro final completo e formatado.
        """
        print(f"\nIniciando geração de roteiro para a categoria: '{categoria}'")
        
        # Fase 1: Gerar a ideia inicial e o esqueleto do roteiro
        texto_inicial, lista_topicos = self._fase1_gerar_esqueleto(categoria)
        if not lista_topicos:
            return "ERRO: Falha ao gerar o esqueleto do roteiro na Fase 1. Verifique os logs."

        # Fase 2: Desenvolver cada tópico do esqueleto
        partes_do_texto = self._fase2_desenvolver_topicos(texto_inicial, lista_topicos)

        # Fase 3: Montar o roteiro final
        roteiro_final = "\n\n".join(partes_do_texto)
        
        print("\n✅ Roteiro longo gerado com sucesso!")
        return roteiro_final

    def _fase1_gerar_esqueleto(self, categoria):
        """
        Executa a Fase 1 em um processo de duas etapas para maior robustez:
        1. Geração criativa da lista de tópicos em texto.
        2. Formatação técnica da lista de texto para JSON.
        """
        print("🚀 Fase 1: Iniciando linha de montagem para gerar o esqueleto do roteiro...")

        try:
            # --- ETAPA 1.1: GERAÇÃO CRIATIVA E INTRODUÇÃO ---
            print("   -> Etapa 1.1: Gerando introdução e lista de tópicos (rascunho)...")
            tarefas_iniciais = {
                'identificar_tendencias_atuais': self._criar_tarefa_com_contexto('identificar_tendencias_atuais', {}),
            }
            tarefas_iniciais['selecionar_tema_viral'] = self._criar_tarefa_com_contexto('selecionar_tema_viral', tarefas_iniciais)
            tarefas_iniciais['criar_roteiro_de_engajamento'] = self._criar_tarefa_com_contexto('criar_roteiro_de_engajamento', tarefas_iniciais)
            tarefas_iniciais['estruturar_roteiro_longo'] = self._criar_tarefa_com_contexto('estruturar_roteiro_longo', tarefas_iniciais)

            equipe_criativa = Crew(
                agents=[self.agents['cacador_de_tendencias'], self.agents['estrategista_de_conteudo'], self.agents['roteirista_provocador'], self.agents['arquiteto_de_narrativa']],
                tasks=[tarefas_iniciais['identificar_tendencias_atuais'], tarefas_iniciais['selecionar_tema_viral'], tarefas_iniciais['criar_roteiro_de_engajamento'], tarefas_iniciais['estruturar_roteiro_longo']],
                verbose=0
            )
            
            resultado_criativo = equipe_criativa.kickoff(inputs={"categoria": categoria})
            
            texto_inicial = str(resultado_criativo.tasks_output[2])
            lista_de_topicos_bruta = resultado_criativo.raw

            # --- ETAPA 1.2: FORMATAÇÃO TÉCNICA ---
            print("   -> Etapa 1.2: Formatando a lista de tópicos para JSON...")
            
            # Injetamos explicitamente a lista bruta no prompt do formatador
            tarefa_formatar_json = Task(
                description=f"""
                Pegue o seguinte texto, que contém uma lista de tópicos:
                ---
                {lista_de_topicos_bruta}
                ---
                Extraia cada item da lista. Monte um único objeto JSON com a chave "topics", 
                contendo um array com os itens da lista.
                Sua saída DEVE ser apenas o JSON, sem nenhum texto ou explicação adicional.
                """,
                agent=self.agents['formatador_json_de_lista'],
                expected_output='Um único bloco de código JSON bem formatado. Exemplo: {"topics": ["Tópico 1", "Tópico 2"]}'
            )

            equipe_tecnica = Crew(
                agents=[self.agents['formatador_json_de_lista']],
                tasks=[tarefa_formatar_json],
                verbose=1
            )
            
            resultado_formatado = equipe_tecnica.kickoff()
            lista_topicos_json = json.loads(resultado_formatado.raw)
            lista_topicos = lista_topicos_json['topics']

            print("\n✅ Esqueleto do roteiro gerado e formatado com sucesso!")
            return texto_inicial, lista_topicos

        except (json.JSONDecodeError, KeyError, IndexError, AttributeError, ValueError) as e:
            print(f"\n❌ ERRO CRÍTICO na Fase 1, mesmo com a nova arquitetura: {e}")
            return None, None


    def _fase2_desenvolver_topicos(self, texto_inicial, lista_topicos):
        """Executa a geração e revisão de parágrafos em loop para cada tópico."""
        print("\n🚀 Fase 2: Desenvolvendo e revisando cada tópico do roteiro...")
        
        partes_do_texto = [texto_inicial]
        agente_escritor = self.agents['roteirista_provocador']
        agente_revisor = self.agents['revisor_linguistico'] # Pega o novo agente

        for topico in lista_topicos:
            print(f"   -> Escrevendo rascunho para: '{topico}'")
            contexto_recente = "\n\n".join(partes_do_texto[-2:])

            # Tarefa para o escritor criar o rascunho
            tarefa_escrita = Task(
                description=f"""
                Baseado no contexto abaixo, desenvolva o tópico: '{topico}'.
                Escreva em português do Brasil coloquial. NÃO use emojis ou inglês.

                CONTEXTO DO QUE JÁ FOI ESCRITO:
                ---
                {contexto_recente}
                ---
                """,
                agent=agente_escritor,
                expected_output="Um ou mais parágrafos de texto que aprofundam o tópico."
            )

            # Tarefa para o revisor corrigir o rascunho
            tarefa_revisao = Task(
                description=f"""
                Revise o texto a seguir. Remova TODOS os emojis. 
                Traduza qualquer palavra em inglês para um bom equivalente em português do Brasil.
                Mantenha o tom conversacional, mas profissional.
                Retorne APENAS o texto corrigido e limpo.

                TEXTO PARA REVISAR:
                ---
                {tarefa_escrita.expected_output}
                ---
                """,
                agent=agente_revisor,
                context=[tarefa_escrita], # A revisão depende da escrita
                expected_output="O texto final, 100% em português do Brasil e sem emojis."
            )

            # Equipe temporária com escritor e revisor
            equipe_temporaria = Crew(
                agents=[agente_escritor, agente_revisor],
                tasks=[tarefa_escrita, tarefa_revisao],
                process=Process.sequential,
                verbose=0
            )

            resultado_revisado = equipe_temporaria.kickoff()
            
            # Adicionamos o resultado já revisado e limpo
            partes_do_texto.append(resultado_revisado.raw)
            print(f"   -> Tópico '{topico}' revisado e finalizado.")

        return partes_do_texto

# --- Bloco de Execução para Teste ---
# Este código só roda quando você executa `python gerador_de_conteudo.py` diretamente.
# Ele não será executado quando o módulo for importado.
if __name__ == '__main__':
    print("--- MODO DE TESTE DO MÓDULO ---")
    
    # 1. Instancia o gerador
    gerador = GeradorDeRoteiro()
    
    # 2. Define o assunto/categoria
    assunto_teste = "Inteligência Artificial e o Futuro do Emprego"
    
    # 3. Chama o método principal para gerar o roteiro
    roteiro_gerado = gerador.gerar_roteiro_longo(categoria=assunto_teste)
    
    # 4. Imprime e salva o resultado final
    print("\n\n################## ROTEIRO FINAL COMPLETO ##################\n")
    print(roteiro_gerado)
    
    # Opcional: Salvar em um arquivo
    with open("roteiro_teste.txt", "w", encoding="utf-8") as f:
        f.write(roteiro_gerado)
    print("\n\n Roteiro de teste salvo em 'roteiro_teste.txt'")