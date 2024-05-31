1. começar por ver loc dos clientes 
1.1 o stor disse maybe fazer 2 clusterizaçõe: uma de loc e uma de basket e dps juntar

2. fazer tudo em .py , .ipynb só para exploração de dados 

3. cuidado com variáveis categóricas em clusters (KNN) dão muito overfit

4. não usar o single hierarchical pq é fraco(ward e k-means é bacano )

5. Não ter muitas variaveis categoricas pq "pesam" muito e consequentemente estragram  o modelo

6. usar o geopandas para ver a locations

as karens da populacao são pessoas que fazem reclamações (cascais é um caso)

thresholds de outliers:     thresholds = {
        'spend_videogames' : (1, 2200),
        'spend_fish' : (1, 3000),
        'spend_meat' : (3, 4000),
        'spend_electronics' : (0, 8000),
        'spend_petfood' : (0, 4200)
    }