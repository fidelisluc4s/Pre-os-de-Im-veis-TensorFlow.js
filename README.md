# Predição de Preços de Imóveis Utilizando TensorFlow.js

Este projeto foi desenvolvido como parte de um trabalho acadêmico para a disciplina de Inteligência Artificial. O objetivo é criar um modelo de predição de preços de imóveis com base em seus metros quadrados utilizando a biblioteca TensorFlow.js.

## Descrição do Projeto

Neste projeto, desenvolvemos um modelo de aprendizado de máquina multicamadas para prever preços de imóveis. Utilizamos dados coletados e realizamos diversas etapas, incluindo a preparação dos dados, a criação e o treinamento do modelo, e a avaliação dos resultados.

## Estrutura do Projeto

- `data/` : Contém os arquivos de dados utilizados no projeto.
- `src/` : Contém o código-fonte do projeto.
- `reports/` : Contém o relatório detalhado sobre o trabalho desenvolvido.

## Etapas do Desenvolvimento

### 1. Coleta e Preparação dos Dados

Os dados foram obtidos em formato CSV e convertidos para JSON. Em seguida, os dados foram limpos e normalizados para garantir que apenas entradas válidas fossem utilizadas no modelo. Utilizamos a biblioteca `tfvis` para visualizar a relação entre o preço e os metros quadrados dos imóveis.

### 2. Criação do Modelo de Machine Learning

Utilizamos um modelo sequencial com três camadas densas:

- **Camada de entrada**: 50 unidades
- **Camada intermediária**: 50 unidades com função de ativação `relu`
- **Camada de saída**: 1 unidade

O modelo foi compilado com o otimizador `adam` e a função de perda de erro quadrático médio (MSE).

### 3. Treinamento e Avaliação do Modelo

O modelo foi treinado utilizando os dados normalizados e avaliado com novos dados. Os resultados foram visualizados e comparados com os dados originais.

### 4. Conclusão

Demonstramos que o modelo desenvolvido é eficaz para prever preços de imóveis com base nos metros quadrados. Investigamos a influência da complexidade do modelo e concluímos que adicionar mais camadas densas não resultou em uma melhoria significativa no erro de predição. Um modelo com três camadas densas foi suficiente para alcançar um bom desempenho, equilibrando precisão e eficiência.

## Relatório

O relatório detalhado pode ser encontrado na pasta `reports/`. Ele contém informações sobre os dados coletados, a preparação dos dados, a explicação do problema, detalhes sobre o modelo de Machine Learning criado, e a conclusão sobre os resultados encontrados.

## Como Executar o Projeto

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/nome-do-repositorio.git
