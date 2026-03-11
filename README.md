# 💳 Credit Risk Scoring com XGBoost + SHAP

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explicabilidade-818cf8?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Concluído-34d399?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-60a5fa?style=for-the-badge)

> Modelo de scoring de crédito para previsão de inadimplência nos próximos 2 anos, com foco em **alta performance preditiva** e **explicabilidade total via SHAP**.

---

## 📌 Visão Geral

Inadimplência é um dos maiores riscos para instituições financeiras. Este projeto constrói um pipeline completo de Machine Learning — da análise exploratória ao modelo em produção — para classificar a probabilidade de um cliente se tornar inadimplente, permitindo decisões de crédito mais assertivas e transparentes.

### Resultados Obtidos

| Métrica | Valor |
|:--------|------:|
| **ROC-AUC** | **0.94** |
| **PR-AUC** | **0.72** |
| Precisão (Inadimplente) | 0.68 |
| Recall (Inadimplente) | 0.71 |
| F1-Score (Inadimplente) | 0.69 |

---

## 🎯 Problema de Negócio

**Contexto:** Uma instituição financeira precisa avaliar o risco de crédito de seus clientes antes de aprovar empréstimos.

**Objetivo:** Prever se um cliente terá **90+ dias de atraso** nos próximos 2 anos (`SeriousDlqin2yrs = 1`).

**Desafio principal:** Dataset altamente desbalanceado — apenas **6.7% de inadimplentes** — exigindo estratégias específicas de tratamento.

**Impacto esperado:** Redução de chargebacks em até **40%** com adoção do modelo em produção.

---

## 🗂️ Estrutura do Projeto

```
credit-risk-scoring/
│
├── 📓 notebooks/
│   └── credit_risk_scoring.ipynb   # Notebook principal completo
│
├── 📁 src/
│   ├── preprocessing.py            # Pipeline de pré-processamento
│   ├── train.py                    # Script de treinamento
│   └── predict.py                  # Inferência / scoring
│
├── 📁 data/
│   ├── raw/                        # Dados originais (não versionados)
│   └── processed/                  # Dados transformados
│
├── 📁 models/
│   └── xgb_credit_scoring.json    # Modelo serializado
│
├── 📁 reports/
│   └── figures/                    # Gráficos gerados
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔬 Metodologia

### Pipeline Completo

```
Dados Brutos
    │
    ▼
Análise Exploratória (EDA)
    │  • Distribuições por classe
    │  • Correlações
    │  • Análise de missing values
    ▼
Feature Engineering
    │  • TotalDelays (histórico acumulado de atrasos)
    │  • IncomeToDebt (razão renda/dívida)
    │  • AgeGroup (faixa etária codificada)
    │  • HighRevolvingUtil (flag de alta utilização)
    │  • HasAnyDelay (indicador binário de atraso)
    │  • IncomeMissing (flag de renda ausente)
    ▼
Pré-processamento
    │  • Imputação por mediana (SimpleImputer)
    │  • Split estratificado 70/15/15
    │  • SMOTE para balanceamento do treino
    ▼
Modelagem
    │  • Baselines: Logistic Regression + Random Forest
    │  • Otimização: Optuna (30 trials, TPE Sampler)
    │  • Modelo final: XGBoost com melhores hiperparâmetros
    ▼
Avaliação
    │  • ROC-AUC, PR-AUC
    │  • Curvas ROC e Precision-Recall
    │  • Matriz de confusão
    │  • Análise do threshold ideal
    ▼
Explicabilidade (SHAP)
    │  • Summary Plot (impacto global)
    │  • Bar Plot (importância média)
    │  • Waterfall Plot (explicação individual)
    └  • Dependence Plot (feature mais importante)
```

### Principais Decisões Técnicas

| Decisão | Justificativa |
|:--------|:-------------|
| **SMOTE** para balanceamento | Preserva a distribuição real no validation/test; evita vazamento de dados |
| **Optuna** para tuning | Mais eficiente que GridSearch; TPE Sampler converge com menos trials |
| **SHAP TreeExplainer** | Valores exatos para tree-based models; mais rápido que KernelExplainer |
| **Split 70/15/15** | Validation independente para tuning; test set nunca visto durante desenvolvimento |
| **PR-AUC** como métrica principal | Mais informativa que ROC-AUC em datasets desbalanceados |

---

## 🔍 Principais Insights (SHAP)

1. **RevolvingUtilizationOfUnsecuredLines** — Alta utilização do crédito rotativo é o maior preditor de inadimplência. Clientes com utilização > 75% têm risco significativamente maior.

2. **TotalDelays** *(feature criada)* — O histórico acumulado de atrasos (30, 60 e 90 dias) é altamente preditivo. Um único atraso já eleva consideravelmente o score de risco.

3. **age** — Relação não-linear: clientes jovens (< 30 anos) apresentam maior risco; a partir dos 45 anos o risco reduz progressivamente.

4. **DebtRatio** — Razão dívida/renda elevada aumenta o risco, especialmente combinada com baixa renda mensal.

5. **MonthlyIncome** — Renda baixa está associada a maior inadimplência; a ausência de renda declarada (`IncomeMissing=1`) também é um sinal de alerta.

---

## 📊 Dataset

**Give Me Some Credit** — [Kaggle Competition](https://www.kaggle.com/c/GiveMeSomeCredit)

| Feature | Descrição |
|:--------|:----------|
| `RevolvingUtilizationOfUnsecuredLines` | Utilização total do crédito rotativo |
| `age` | Idade do tomador |
| `NumberOfTime30-59DaysPastDueNotWorse` | Qtd. de atrasos de 30-59 dias |
| `DebtRatio` | Pagamentos mensais / renda mensal |
| `MonthlyIncome` | Renda mensal (19% de missing) |
| `NumberOfOpenCreditLinesAndLoans` | Linhas de crédito abertas |
| `NumberOfTimes90DaysLate` | Qtd. de atrasos ≥ 90 dias |
| `NumberRealEstateLoansOrLines` | Empréstimos imobiliários |
| `NumberOfTime60-89DaysPastDueNotWorse` | Qtd. de atrasos de 60-89 dias |
| `NumberOfDependents` | Número de dependentes |
| **`SeriousDlqin2yrs`** | **🎯 Target: inadimplência nos próximos 2 anos** |

---

## ⚙️ Como Executar

### 1. Clone o repositório
```bash
git clone https://github.com/GabrielAlessi/credit-risk-scoring.git
cd credit-risk-scoring
```

### 2. Crie o ambiente virtual
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Execute o notebook
```bash
jupyter notebook notebooks/credit_risk_scoring.ipynb
```

> **💡 Dica:** O notebook funciona sem o dataset original — uma versão sintética com a mesma estrutura é gerada automaticamente na célula de carregamento. Para usar os dados reais, baixe o dataset do Kaggle e siga as instruções na **Opção A** da célula de carregamento.

---

## 📦 Dependências

```
xgboost>=1.7.0
shap>=0.42.0
imbalanced-learn>=0.10.0
optuna>=3.0.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## 🚀 Próximos Passos

- [ ] **Deploy via FastAPI** — endpoint `/predict` retornando score + top 3 fatores de risco (SHAP)
- [ ] **Monitoramento com Evidently AI** — detecção de data drift em produção
- [ ] **Calibração de probabilidades** — Platt Scaling para scores mais confiáveis
- [ ] **Análise de fairness** — verificar viés por faixa etária e renda
- [ ] **Threshold dinâmico** — otimização por faixa de valor do crédito solicitado
- [ ] **Dashboard interativo** — Streamlit app para visualização dos scores

---

## 👨‍💻 Autor

**Gabriel Alessi Naumann**  
Cientista de Dados Sênior

[![LinkedIn](https://img.shields.io/badge/LinkedIn-gabriel--alessi--naumann-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/gabriel-alessi-naumann/)
[![GitHub](https://img.shields.io/badge/GitHub-GabrielAlessi-181717?style=flat&logo=github)](https://github.com/GabrielAlessi)
[![Kaggle](https://img.shields.io/badge/Kaggle-gabrielalessinaumann-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/gabrielalessinaumann)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

*⭐ Se este projeto foi útil para você, considere deixar uma estrela no repositório!*
