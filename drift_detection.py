import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy


def get_metrics(drifts, resp, window_size=100, verbose=False):
    """
    Precision, Recall, F1 Score and Delay
    for the drift predictions
    """
    precision = 0
    recall = 0
    tp = 0
    delay = 0
    avg_delay = 0
    resp_ = resp.copy()
    predicted = [0 for x in resp_]
    
    if isinstance(window_size, int):
        window_size = np.repeat(window_size, len(drifts))
    
    for i in range(len(drifts)):    
        for j in range(len(resp_)):
            # print(drifts[i], resp_[j], drifts[i] - resp_[j])
            if 0 <= drifts[i] - resp_[j] <= 1 * window_size[i]:
                if verbose:
                    print((drifts[i], drifts[i] + window_size[i], resp_[j]))
                    
                delay += drifts[i] + window_size[i] - resp_[j]
                tp += 1
                resp_[j] = np.inf
                predicted[j] = 1
                break
    
    if len(drifts) > 0:
        precision = tp/len(drifts)
    
    if len(resp_) > 0:
        recall = tp/len(resp_)
        
    try:
        f1 = scipy.stats.hmean([precision, recall])
    except ValueError:
        f1 = 0.0
        
    if tp > 0:
        avg_delay = delay/tp
    
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Delay": avg_delay,
        "Correct_Predictions": predicted,
        "Support": sum(predicted),
        "Drifts_Found": drifts,
        "Resp": resp,
    }


def plot_deteccao_drift(
    df,
    col,
    detected_drifts,
    y_true,
    rolling_means,
    lowers,
    uppers,
    cluster_window_size=50,
    save_png="",
):
    """
        Plota o gráfico do poster ilustrando o método da detecção do concept drift

        Parâmetros:
        -----------
            df (pd.DataFrame): 

    """
    if save_png != "":
        plt.ioff()

    fig = plt.figure(figsize=(18, 6))
    ax = plt.gca()
    ax.plot(df.index, df[col], c="#ff968f", lw=2, label=col)
    ax.plot(
        df.index,
        rolling_means,
        c="#50A898",
        linestyle="-",
        lw=2,
        marker=".",
        markeredgewidth=3,
        label="média móvel",
    )
    ax.fill_between(
        df.index,
        lowers,
        uppers,
        facecolor="#52adff",
        alpha=0.1,
        label="tolerância do desvio padrão",
    )
    ax.plot(df.index, uppers, c="#52adff", alpha=0.5, marker="v", markeredgewidth=3)
    ax.plot(df.index, lowers, c="#52adff", alpha=0.5, marker="^", markeredgewidth=3)

    first = True
    for val in y_true:
        if first:
            first = False
            ax.axvline(
                x=val,
                ls="--",
                lw=3,
                c="darkgreen",
                alpha=0.5,
                label="$\it{concept}$ $\it{drift}$ real",
            )
        else:
            ax.axvline(x=val, ls="--", lw=3, c="darkgreen", alpha=0.5)

    first = True
    for val in detected_drifts:
        if first:
            first = False
            ax.axvline(
                x=val,
                ls="-",
                lw=2,
                c="#e8d690",
                label="$\it{concept}$ $\it{drift}$ predito",
            )
        else:
            ax.axvline(x=val, ls="-", lw=2, c="#e8d690")

    metrics = get_metrics(detected_drifts, y_true, cluster_window_size)
    plt.title(
        "Precision: {}  Recall: {}  F1: {}  Delay:{}".format(
            "{0:.2f}%".format(metrics["Precision"] * 100),
            "{0:.2f}%".format(metrics["Recall"] * 100),
            "{0:.2f}%".format(metrics["F1"] * 100),
            "{0:.2f}".format(metrics["Delay"]),
        )
    )

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(18)

    ax.set_xlabel("índice dos $\it{traces}$")
    plt.legend(
        fontsize=18,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        fancybox=True,
        shadow=True,
        ncol=3,
    )

    if save_png != "":
        plt.savefig(save_png, dpi=100, transparent=False)
        plt.close(plt.gcf())
        plt.ion()


def detect_concept_drift(df, var_ref, rolling_window=5, std_tolerance=3, verbose=False):
    window_buffer = []
    drifts = []
    mean = None
    std = None

    lowers = []
    uppers = []
    means = df[var_ref].rolling(window=rolling_window).mean().values.tolist()

    for i, row in df.iterrows():
        if len(window_buffer) < rolling_window:
            window_buffer.append(row[var_ref])
            lowers.append(np.nan)
            uppers.append(np.nan)

        else:
            if mean is not None:
                expected_lower = mean - (std_tolerance * std)
                expected_upper = mean + (std_tolerance * std)

                lowers.append(expected_lower)
                uppers.append(expected_upper)

                if expected_lower > row[var_ref] or row[var_ref] > expected_upper:
                    if verbose:
                        print(i, expected_lower, expected_upper, row[var_ref])
                    drifts.append(i)

                    window_buffer = []
            else:
                lowers.append(np.nan)
                uppers.append(np.nan)

            if len(window_buffer) > 0:
                window_buffer.pop(0)

            window_buffer.append(row[var_ref])
            if i in drifts:
                mean = None
                std = None
            else:
                mean = np.mean(window_buffer)
                std = np.std(window_buffer)

    return drifts, {"lowers": lowers, "uppers": uppers, "means": means}


def cumulative_detect_concept_drift(
    df, var_ref, min_buffer=2, std_tolerance=3, smoothing_factor=0.5, verbose=False
):
    window_buffer = []
    drifts = []
    mean = None
    std = None

    lowers = []
    uppers = []
    means = []

    for i, row in df.iterrows():
        if mean is not None:
            expected_lower = mean - (std_tolerance * std)
            expected_upper = mean + (std_tolerance * std)

            lowers.append(expected_lower)
            uppers.append(expected_upper)

            if expected_lower > row[var_ref] or row[var_ref] > expected_upper:
                if verbose:
                    print(i, expected_lower, expected_upper, row[var_ref])
                drifts.append(i)

                window_buffer = []
                mean = None
        else:
            lowers.append(np.nan)
            uppers.append(np.nan)

        window_buffer.append(row[var_ref])

        if len(window_buffer) >= min_buffer:
            # mean = np.mean(window_buffer)
            # std = np.std(window_buffer)
            mean = (
                pd.Series(window_buffer)
                .ewm(alpha=smoothing_factor, adjust=True)
                .mean()
                .values[-1]
            )
            std = (
                pd.Series(window_buffer)
                .ewm(alpha=smoothing_factor, adjust=True)
                .std()
                .values[-1]
            )
            means.append(mean)
        else:
            means.append(np.nan)

    return drifts, {"lowers": lowers, "uppers": uppers, "means": means}


def exponential_smooth_detect_concept_drift(
    df, var_ref, min_buffer=2, std_tolerance=3, smoothing_factor=0.5, verbose=False
):
    window_buffer = []
    drifts = []
    mean = None
    std = None

    lowers = []
    uppers = []
    means = []

    for i, row in df.iterrows():
        if mean is not None:
            expected_lower = mean - (std_tolerance * std)
            expected_upper = mean + (std_tolerance * std)

            lowers.append(expected_lower)
            uppers.append(expected_upper)

            if expected_lower > row[var_ref] or row[var_ref] > expected_upper:
                if verbose:
                    print(i, expected_lower, expected_upper, row[var_ref])
                drifts.append(i)

                window_buffer = []
                mean = None
        else:
            lowers.append(np.nan)
            uppers.append(np.nan)

        window_buffer.append(row[var_ref])

        if len(window_buffer) >= min_buffer:
            # model = SimpleExpSmoothing(np.array(window_buffer))
            # model = model.fit(smoothing_factor)
            # mean = model.forecast(1)[0]
            # std = np.std(model.fittedvalues)

            mean = (
                pd.Series(window_buffer)
                .ewm(alpha=smoothing_factor, adjust=True)
                .mean()
                .values[-1]
            )
            std = (
                pd.Series(window_buffer)
                .ewm(alpha=smoothing_factor, adjust=True)
                .std()
                .values[-1]
            )
            means.append(mean)
        else:
            means.append(np.nan)

    return drifts, {"lowers": lowers, "uppers": uppers, "means": means}