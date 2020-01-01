from sklearn.metrics import silhouette_score, davies_bouldin_score, mean_squared_error
from scipy.spatial import distance
from scipy.stats import skew
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from copy import deepcopy


def get_validation_indexes(X, y_pred):
    """
        Retorna indíces de validação de clustering baseados
        nos inputs X e os grupos associados a cada cado y_pred

        Silhouette, DBScan
    """
    return {
        "Silhouette": silhouette_score(X, y_pred),
        "DBi": davies_bouldin_score(X, y_pred),
        #  "DBCV": DBCV(X, y_pred)
    }


def get_centroids_metrics(X, y_pred, centroids):
    """
        Calcula métricas baseadas nos centróides e nos grupos 
        atribuídos a cada cado no conjunto X

        Parâmetros:
        -----------
            X (pd.DataFrame): Conjunto de dados 
            y_pred (np.array): Grupos atribuídos a cada cluster
            centroids (np.array): Centróides de cada grupo
        
        Retorno:
        ---------
            dict
    """
    r = {
        "radius_list": [],
        "dist_intra_cluster_list": [],
        "skewness_list": [],
        "cluster_std_list": [],
        # "volume_list": {values[i]: counts[i] for i in range(len(values))},
    }

    # Calculates metrics for each cluster
    for j in range(len(centroids)):
        X_in_cluster = X[y_pred == j]

        try:
            # Maximum distance of a point to centroid
            r["radius_list"].append(distance.cdist(X_in_cluster, [centroids[j]]).max())
        except ValueError:
            r["radius_list"].append([0 for _ in range(len(centroids))])

        # Average intra-cluster distances
        dist_intra = distance.pdist(X_in_cluster).mean()
        r["dist_intra_cluster_list"].append(dist_intra)
        r["dist_intra_cluster_i=" + str(j)] = dist_intra

        # Skewness of cluster
        skewness = skew(X_in_cluster, axis=None)
        r["skewness_list"].append(skewness)
        r["skewness_i=" + str(j)] = skewness

        # Std of cluster
        c_std = X_in_cluster.std()
        r["cluster_std_list"].append(c_std)
        r["cluster_std_i=" + str(j)] = c_std
    return r


def get_individuals(resp_1, resp_2, key, k="equal"):
    """
        Pega metricas individuais de cada cluster do 
        agrupamento
    """
    r = {}
    key_ = key.replace("_list", "")

    if isinstance(resp_1[key], dict):
        # Se for dicionário, separa em arrays de chave-valor
        i1 = [x for x in resp_1[key].keys()]
        i2 = [x for x in resp_2[key].keys()]

        r1 = [x for x in resp_1[key].values()]
        r2 = [x for x in resp_2[key].values()]
    else:
        i1 = list(range(len(resp_1[key])))
        i2 = list(range(len(resp_2[key])))

        r1 = resp_1[key]
        r2 = resp_2[key]

    for i in range(len(i1)):
        if i1[i] in i2:
            r["diff_" + key_ + "_i=" + str(i1[i])] = r2[i2[i]] - r1[i1[i]]
        else:
            break

    return r

def get_mean_squared(centroids_1, centroids_2, cols=None):
    """
        Calculates Mean Squared Error (MSE) to compare clusterings 1 and 2
    """
    if cols is None:
        cols = list(range(centroids_1.shape[1]))
        
    mse = np.mean((centroids_1 - centroids_2) ** 2, axis=0)
    
    resp = {
        "MSE__" + str(cols[i]): mse[i] for i in range(len(cols))
    }
    resp.update({
        "total_MSE": np.sum(mse),
        "avg_MSE": np.mean(mse),
        "count_non_zero_MSE": np.count_nonzero(mse)        
    })
    return resp

def compare_clusterings(resp_1, resp_2, cols=None):
    """
        Compara dois agrupamentos em duas janelas diferentes e
        retorna métricas das variações 
    """
    r = {}

    # se não houve centroides encontrados nos dois grupos, retorna vazio
    if len(resp_1["centroids"]) == 0 or len(resp_2["centroids"]) == 0:
        return r

    for key in resp_1:
        try:
            if key != "i":
                key_ = key.replace("_list", "")
            
                # ---------------
                # diff_centroids
                # ---------------
                # Calcula a menor distância média interclusters os clusters
                if key == "centroids":
                    r["diff_" + key_] = min(
                        distance.cdist(resp_1[key], resp_2[key]).min(axis=0).mean(),
                        distance.cdist(resp_2[key], resp_1[key]).min(axis=0).mean(),
                    )

                    r.update(
                        get_mean_squared(resp_1[key], resp_2[key], cols)
                    )

                # -------------------------
                # diff_radius
                # diff_skewness
                # diff_dist_intra_cluster
                # diff_cluster_std
                # -------------------------
                # Para métricas baseadas em lista, calcula a diferença média
                elif isinstance(resp_1[key], list):
                    # Reordena os valores do cluster_1 para correspondência
                    # resp_1[key] = np.array(resp_1[key][ordem]).tolist()
                    
                    r["diff_" + key_] = (
                        np.array(resp_2[key]) - np.array(resp_1[key])
                    ).mean()

                    # Metricas individuais por cluster
                    r.update(get_individuals(resp_1, resp_2, key))


                # ------------
                # diff_volume
                # ------------
                # Para o volume, é necessário separar caso haja o grupo -1
                # (outliers no DBScan)
                # De outra forma, pega a menor distância média
                elif key == "volume_list":                    
                    if -1 in resp_1[key].keys():
                        r["diff_volume_outliers"] = (
                            resp_2[key][-1] - resp_1[key][-1]
                        ) / sum(resp_2[key].values())

                    r["diff_volume"] = (
                        distance.cdist(
                            [[x] for x in resp_1[key].values()],
                            [[x] for x in resp_2[key].values()],
                        )
                        .min(axis=0)
                        .mean()
                    )

                    # Metricas individuais por cluster
                    r.update(get_individuals(resp_1, resp_2, key))
                else:
                # -----------------
                # diff_DBi
                # diff_Silhouette
                # diff_k
                # -----------------
                # Para métricas númericas, faz diretamente a subtração
                    r["diff_" + key_] = resp_2[key] - resp_1[key]
            else:
                # --
                # i
                # --
                # Para o i, segue o valor do segundo momento
                r["i"] = resp_2["i"]
        except Exception as e:
            print(key)
            print(resp_1[key])
            # raise

    return r

def cluster_overlap(cluster_X, cluster_Y):
    """
    Calculates the overlap between two clusters, i.e. how much one 
    matches another

    Args:
        cluster_X (np.array): List of data points in the cluster
        cluster_Y (np.array): List of data points in the cluster
    
    Returns:
        (float) The overlap between the two clusters (0-1)
    """
    overlap_sum = 0
    X_sum = 0

    if len(cluster_X) == 0 or len(cluster_Y) == 0:
        return 0

    for i in range(len(cluster_X)):
        # print(cluster_X[i], cluster_Y, np.in1d(cluster_X[i], cluster_Y).all())
        if np.in1d(cluster_X[i], cluster_Y).all():
            overlap_sum += 1
            # overlap_sum += 1
        X_sum += 1

    return overlap_sum/X_sum

def run_offline_clustering_window(
    model, window, df, sliding_window=False, sliding_step=50, 
):
    """
        Roda o modelo de clusterização offline baseado em janelas

        Parâmetros:
        -----------
                  model (sklearn): Modelo parametrizado
                     window (int): Tamanho da janela a ser utilizada
                df (pd.DataFrame): Base de dados a ser utilizada 
      sliding_window(bool, False): Se utiliza janela deslizante ou não
            sliding_step(int, 50): Tamanho do passo na janela deslizante

        Retorno:
        --------
            run_df, measures_df
    """
    resp = []
    # old_X = None

    if sliding_window:
        # loop = tqdm_notebook(range(0, len(df) - window + 1, sliding_step))
        loop = range(0, len(df) - window + 1, sliding_step)
    else:
        # loop = tqdm_notebook(range(0, len(df), window))
        loop = range(0, len(df), window)

    for i in loop:
        # print(i)
        # Seleciona janela olhando para frente
        X = df.loc[i : i + window - 1].values
        # print(i, X.shape)

        # Predita modelo com a normalização dos números dos clusters
        # model.fit(X)
        # y_pred = model.labels_
        y_pred = model.fit_predict(X)

        # Faz uma lookup table para reorganizar a ordem das labels
        # dos clusters
        idx = np.argsort(model.cluster_centers_.sum(axis=1))
        lut = np.zeros_like(idx)
        lut[idx] = np.arange(len(idx))
        
        y_pred = lut[y_pred]
        
        #print(i)
        #print("Y_PRED", y_pred)
        #print("NEW_Y_PRED", y_pred)

        # Monta dicionário com respostas dos métodos
        r = {"i": i, "k": len(np.unique(y_pred[y_pred > 0]))}

        # Contagem de dados por labels preditas
        values, counts = np.unique(y_pred, return_counts=True)

        # Adaptative Window
        # if len(ref_runs) > 0:
        #    t1 = len(pd.concat([ref_runs, det_runs]).unique())
        #    t2 = len(pd.concat([new_ref_runs, new_det_runs]).unique())
        #    
        #    var_ratio = t2 / t1
        #    
        #    old_window_size = current_window_size
        #    current_window_size = int(current_window_size * var_ratio)

        # ----------------------------------
        # Calcula métricas silhouette e DBi
        # ----------------------------------
        if max(counts) >= 2 and len(values) > 1:
            r.update(get_validation_indexes(X, y_pred))

        old_model = deepcopy(model)
        # old_X = X.copy()

        r["centroids"] = (
            pd.DataFrame(X)
            .groupby(y_pred)
            .mean()
            .drop(-1, errors="ignore")
            .values
        )

        #if old_centroids is not None:
        #    ordem = distance.cdist(old_centroids, r["centroids"]).argmin(axis=1)
        #    r["centroids"] = r["centroids"][ordem]

        r["avg_dist_between_centroids"] = distance.pdist(r["centroids"]).mean()
        r.update(get_centroids_metrics(X, y_pred, r["centroids"]))

        # Adiciona iteração atual na resposta
        resp.append(r)

    run_df = pd.DataFrame(resp).set_index("i")

    # Expand values for individual clusters
    for col in [
        "radius_list",
        "dist_intra_cluster_list",
        "skewness_list",
        "cluster_std_list"
       #  "volume_list",
    ]:
        min_individuals = run_df[col].apply(len).max()

        try:
            for i in range(min_individuals):
                run_df[col.replace("_list", "") + "_i=" + str(i)] = run_df[col].apply(
                    lambda x: x[i] if i < len(x) else np.nan
                )

            # Create avegares
            if col != "volume_list":
                run_df[col.replace("_list", "")] = run_df[col].apply(lambda x: np.mean(x))
        except Exception as e:
            print(e)
            pass

    measures = [compare_clusterings(resp[i], resp[i + 1]) for i in range(len(resp) - 1)]
    measures_df = pd.DataFrame(measures).set_index("i")
    measures_df.fillna(0, inplace=True)

    return run_df, measures_df
