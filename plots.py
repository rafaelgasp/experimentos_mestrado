import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

def get_plot_file_dict(out):
    """
    Formata as colunas organizando a plotagem dos gráficos

    Args:
        out (pd.DataFrame): métricas do experimento
    
    Returns:
        (dict): grouped list of columns
    """
    plot_dict = {
        "mean": [str(x) + "_mean" for x in range(0, out.k.max())],
        "n_samples": [str(x) + "_n_samples" for x in range(0, out.k.max())],
        "radius": [str(x) + "_radius" for x in range(0, out.k.max())],
        "size": [str(x) + "_size" for x in range(0, out.k.max())],
        "skew": [str(x) + "_skew" for x in range(0, out.k.max())],
        "sq_norm": [str(x) + "_sq_norm" for x in range(0, out.k.max())],
        "squared_sum": [str(x) + "_squared_sum" for x in range(0, out.k.max())],
        "monic_center_location_transition": ["monic_" + str(x) + "_center_location_transition" 
                                                for x in range(0, out.k.max())],
        "monic_compactness_transition": ["monic_" + str(x) + "_compactness_transition" 
                                            for x in range(0, out.k.max())],
        "monic_skewness_transition": ["monic_" + str(x) + "_skewness_transition" 
                                        for x in range(0, out.k.max())],
        "monic_sub_clusters_center_location_transition": ["sub_cluster_" + str(x) + "_center_location_transition" 
                                                             for x in range(out.num_subclusters.max()-1)],
        "monic_sub_clusters_compactness_transition": ["sub_cluster_" + str(x) + "_compactness_transition" 
                                                        for x in range(out.num_subclusters.max()-1)],
        "monic_sub_clusters_skewness_transition": ["sub_cluster_" + str(x) + "_skewness_transition" 
                                                     for x in range(out.num_subclusters.max()-1)],
        'monic_sub_clusters_absorptions': ['sub_cluster_absorptions'], 
        'monic_sub_clusters_splits': ['sub_cluster_splits'],
        'monic_sub_clusters_survivors': ['sub_cluster_survivors'],

        'monic_survivors': ['monic_survivors'],
        'monic_absorptions': ['monic_absorptions'],
        'monic_splits': ['monic_splits'],

        'new_subclusters': ['new_subclusters'],
        'not_changed': ['not_changed'],
        'num_splits': ['num_splits'],
        'num_subclusters': ['num_subclusters'],
        'skew_change': ['skew_change'],
        'std_min_distances': ['std_min_distances'],
        'DBi': ['DBi'],
        'Silhouette': ['Silhouette'],
        'avg_min_distances': ['avg_min_distances'],
        'k': ['k'],
        'labels_changed': ['labels_changed'],
    }

    if len([x for x in out.columns if "diff" in x]) > 0:
        plot_dict.update({
            "mean_diff": [str(x) + "_mean_diff" for x in range(0, out.k.max())],
            "n_samples": [str(x) + "_n_samples_diff" for x in range(0, out.k.max())],
            "radius_diff": [str(x) + "_radius_diff" for x in range(0, out.k.max())],
            "size_diff": [str(x) + "_size_diff" for x in range(0, out.k.max())],
            "skew_diff": [str(x) + "_skew_diff" for x in range(0, out.k.max())],
            "sq_norm_diff": [str(x) + "_sq_norm_diff" for x in range(0, out.k.max())],
            "squared_sum_diff": [str(x) + "_squared_sum_diff" for x in range(0, out.k.max())],
        })

    return plot_dict

def plot_stats(
    folder,
    out, 
    model,
    dataset_name,
    window_size,
    sliding_window, 
    vars_list,
    max_len=0,
    vertical_line=400,
):
    """
    Plota estatísticas de uma rodada do experimento salvando como imagens 
    na pasta especificada
    
    Args:
        folder (str): Pasta destino para salvar os gráficos 
        out (pd.DataFrame): DataFrame com métricas por iteração do 
            algoritmo streaming 
        model (sklearn): Modelo utilizado 
        dataset_name (str): Nome do dataset utilizado no experimento
        window_size (int): Tamanho da janela 
        sliding_window (bool): Se utiliza janela deslizante
        vars_list (list): Lista de variáveis utilizadas na clusterização
        vertical_line (int, 400): Plota linhas na vertical a cada X
    
    Returns:
        None
    """
    plt.ioff()
    
    out.fillna(0, inplace=True)
    
    dict_out = get_plot_file_dict(out)
    
    for key in tqdm_notebook(dict_out):
        try:
            out.plot(y=dict_out[key])
        except KeyError:
            continue

        fig = plt.gcf()
        plt.gca().legend(dict_out[key], loc='upper center', 
                        bbox_to_anchor=(0.5, 1.05), ncol=3, prop={'size': 8})
        
        for i in range(0, max_len, vertical_line):
            plt.axvline(x=i, ls='--', lw=1, c='grey')

        fig.subplots_adjust(bottom=0.25)
        
        plt.suptitle("model: {}, \ndataset: {}, window_size: {}, sliding_window: {}, \nvars: {}".format(
            str(model),
            dataset_name,
            window_size,
            sliding_window, 
            vars_list
        ), fontsize=10, x=0.1, y=0.15, ha="left")

        plt.savefig(folder + '/' + key + '.png', bbox_inches='tight')
        
        plt.close(plt.gcf())
        
    plt.ion()
    
def plot_drift_vertical_lines(tamanho, resp_drift=None):
    first=True
    
    if resp_drift is None:
        resp_drift = int(tamanho * 0.1)
        
    for i in range(resp_drift, tamanho, resp_drift):
        if first:
            first=False
            plt.axvline(x=i, ls='-', lw=2, c='darkgreen', label="drift_real")
        else:
            plt.axvline(x=i, ls='-', lw=2, c='darkgreen')