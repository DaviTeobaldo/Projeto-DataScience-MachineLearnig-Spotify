# 🎵 Predição de Popularidade Musical — Spotify Dataset

> Projeto desenvolvido durante o **7 Days of Code** da [Alura](https://www.alura.com.br/), com foco em Machine Learning aplicado a dados reais do Spotify.

---

## 📌 Sobre o projeto

A popularidade de uma música pode ser prevista a partir de seus atributos acústicos? Este projeto responde a essa pergunta desenvolvendo um pipeline completo de Machine Learning — da exploração dos dados à exportação do modelo — sobre um dataset com **113.999 músicas** extraídas via API do Spotify.

O problema é tratado como uma **classificação binária supervisionada**: prever se uma faixa atingirá alto índice de popularidade (score ≥ 80).

## 🔄 Pipeline

### 1. Análise Exploratória (EDA)
- Inspeção de distribuições, valores ausentes e estatísticas descritivas
- Artistas e gêneros mais populares, músicas mais longas
- Matriz de correlação de Pearson entre features acústicas
- Identificação do desbalanceamento da variável-alvo

### 2. Pré-processamento
- Remoção de duplicatas e valores nulos
- Conversão da popularidade contínua em **classe binária** (`pop_classe`: 1 se ≥ 80, 0 caso contrário)
- Descarte de colunas categóricas e normalização **min-max** das features numéricas
- Divisão treino (80%) / teste (20%) com estratificação

### 3. Modelagem e Comparação
- **Baseline:** Regressão Logística
- Avaliação comparativa: Logistic Regression, KNN, Decision Tree, Random Forest
- Três estratégias de reamostragem testadas: `NearMiss`, `RandomOverSampler` e combinação over+under
- **Melhor resultado:** Random Forest com RandomOverSampling

### 4. Tuning e Validação Cruzada
- Otimização de hiperparâmetros via `RandomizedSearchCV`
- `StratifiedKFold` com 5 folds integrado ao pipeline do `imbalanced-learn`
- Resampling aplicado apenas dentro de cada fold de treino, **sem data leakage**

### 5. Avaliação Final
- Teste no conjunto isolado (nunca visto durante o desenvolvimento)
- Métricas: Accuracy, Precision, Recall, F1-Score e AUC-ROC
- Curva ROC com intervalo de confiança por fold
- Análise de feature importance
- Modelo serializado com `pickle`

---

## 📊 Resultados

| Etapa      | Accuracy | Precision | Recall | F1 |
|------------|----------|-----------|--------|----|
| Baseline   | —        | —         | —      | —  |
| Validação  | —        | —         | —      | —  |
| **Teste**  | —        | —         | —      | —  |

> 💡 Preencha a tabela com os valores gerados ao executar o notebook.

---

## 🛠️ Tecnologias

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-013243?style=flat&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/matplotlib-11557C?style=flat&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/seaborn-4C72B0?style=flat&logo=python&logoColor=white)

| Biblioteca       | Uso                                      |
|------------------|------------------------------------------|
| pandas / numpy   | Manipulação e análise de dados           |
| scikit-learn     | Modelos, métricas e validação cruzada    |
| imbalanced-learn | Reamostragem (SMOTE, NearMiss, Pipeline) |
| seaborn / matplotlib | Visualizações e gráficos             |
| pickle           | Serialização do modelo treinado          |

---

## 💡 Aprendizados

- Em datasets desbalanceados, **F1-score e AUC-ROC** são métricas mais confiáveis que acurácia
- Integrar o resampler dentro do pipeline de cross-validation evita **data leakage**
- A análise de **feature importance** indicou que *energy*, *loudness* e *danceability* têm maior peso preditivo
- A separação rigorosa treino/validação/teste garante uma estimativa honesta de generalização

---

## ▶️ Como executar

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/seu-repositorio.git

# Instale as dependências
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib

# Abra o notebook
jupyter notebook 7_days_of_code_Spotify.ipynb
```

Ou acesse diretamente pelo Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seu-usuario/seu-repositorio/blob/main/7_days_of_code_Spotify.ipynb)

---

## 📁 Dataset

Dataset público disponível no GitHub do projeto 7 Days of Code:  
🔗 [letpires/7DaysOfCodeSpotifyML](https://github.com/letpires/7DaysOfCodeSpotifyML)

---

## 📄 Licença

Este projeto está sob a licença MIT. Sinta-se livre para usar, adaptar e compartilhar.
