# Pré-registro (escrito em 2026-11-14, antes de qualquer rodada de merge)

Pergunta geral: escolhendo pesos (coeficientes de merge, checkpoints, camadas) de modelos abertos pequenos
(Qwen2.5-0.5B base/instruct/coder-instruct, SmolLM2-360M base/instruct, Pythia-160M checkpoints), obtém-se um modelo
melhor que o melhor dos originais, medido em itens de teste nunca usados na escolha?

Métrica primária (artigos 1-6): média macro de acurácia em ARC-Easy, ARC-Challenge, HellaSwag (acc_norm), WinoGrande,
1000 itens de teste por tarefa. Seleção na validação (300 itens/tarefa, disjunta do teste). Pythia (art. 7): NLL do WikiText-2.
Sucesso: IC 95% pareado (bootstrap sobre itens, 5000 reamostragens) da diferença (merge escolhido - melhor original) com limite inferior > 0.
Empate: IC contém 0. Comparação contra o melhor ORIGINAL escolhido na validação (não no teste).

Artigos e hipóteses:
1. LERP base<->instruct (Qwen): há t* interior que supera ambos. Grade t=0,0.1,...,1.
2. SLERP vs LERP: SLERP não difere de LERP (modelos próximos). Grade t=0.2,0.4,0.6,0.8.
3. Aritmética de tarefas com extrapolação: lambda>1 em base+lambda*(instruct-base) pode superar lambda=1. lambda em {0.5,0.75,1.25,1.5,2}.
4. Três modelos (base, instruct, coder): fusão ponderada/TIES/DARE supera a melhor fusão de dois. Grade lambda1,lambda2 em {.25,.5,.75}^2; TIES densidade 0.5; DARE p=0.5 (5 sementes).
5. Fusão por grupo de camadas (4 grupos) por busca em coordenadas na validação supera t* global na validação, mas a vantagem encolhe no teste (maldição do vencedor).
6. Replicação em SmolLM2-360M: mesmo protocolo do artigo 1, grade t=0,0.2,...,1.
7. Média de checkpoints do Pythia-160M (últimos k=2..6 de 100k..143k passos) reduz NLL em relação ao último checkpoint.
8. Maldição do vencedor: o ganho na validação do t escolhido é maior que o ganho no teste (grade do artigo 1; 1000 reamostragens da validação).
9. Tamanho da validação: regret de seleção (acurácia de teste do t oráculo - t escolhido) cai com n de validação em {30,60,100,200,300}.
10. Fronteira acurácia x perplexidade ao longo de t (Qwen base<->instruct): t* de acurácia difere do t* de NLL do WikiText.

Exploratório (declarado como tal): correlação cosseno/norma de vetores de tarefa; qualquer comparação fora das acima.
Regras: não alterar métrica, grades ou hipóteses após ver resultados; resultados nulos são reportados como nulos.
