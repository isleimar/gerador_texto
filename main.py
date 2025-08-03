# usar_modulo.py
from gerador_de_conteudo import GeradorDeRoteiro

def main():
    """
    Função principal que demonstra o uso do módulo GeradorDeRoteiro.
    """
    # 1. Instancie a classe do seu módulo.
    #    Toda a configuração pesada (LLM, agentes) acontece aqui, uma única vez.
    print("Criando uma instância do gerador de roteiro...")
    gerador = GeradorDeRoteiro()

    # 2. Defina o assunto que você quer usar.
    #    Este valor será passado como argumento.
    assunto = "Tecnologia e Privacidade"
    
    # 3. Chame o método para gerar o conteúdo.
    #    É aqui que a "mágica" acontece.
    print(f"Solicitando a geração de um roteiro sobre: '{assunto}'")
    roteiro_final = gerador.gerar_roteiro_longo(categoria=assunto)

    # 4. Use o resultado como precisar.
    print("\n" + "="*50)
    print("      ROTEIRO RECEBIDO DO MÓDULO")
    print("="*50 + "\n")
    print(roteiro_final)

    # Salvar o resultado em um arquivo
    nome_arquivo = f"roteiro_{assunto.lower().replace(' ', '_')}.txt"
    with open(nome_arquivo, "w", encoding="utf-8") as f:
        f.write(roteiro_final)
    
    print(f"\n\n[INFO] Roteiro salvo com sucesso em '{nome_arquivo}'")


if __name__ == '__main__':
    main()