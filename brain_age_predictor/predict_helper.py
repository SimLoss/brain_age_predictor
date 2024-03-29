# pylint: disable= too-many-arguments, invalid-name

"""
Helper module containing useful function for making plots of final scores.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

def plot_scores(y_test,
                age_predicted,
                metrics,
                model_name="Regressor model",
                dataframe_name="Dataframe",
                ):
    """
    Plots the results of the predictions vs ground truth with related metrics
    scores.

    Parameters
    ----------

    y_test : pandas dataframe
             Pandas dataframe column containing the ground truth age.

    age_predicted : array-like
                    Array containing the predicted age of each subject.

    metrics : dictionary
            Dictionary containing names of metrics as keys and result metrics .
            for a specific model as values.

    model_name : string
                Model's name, DEFAULT="Regressor Model"

    dataframe_name : string
                Dataframe's name, DEFAULT="Dataset Metrics".
    """
    mse, mae, pr = metrics["MSE"], metrics["MAE"], metrics["PR"]

    ax = plt.subplots(figsize=(8, 8))[1]
    ax.scatter(y_test,
                age_predicted,
                marker="*",
                c="r",
              )
    plt.xlabel("Ground truth Age [years]", fontsize=18)
    plt.ylabel("Predicted Age [years]", fontsize=18)
    ax.set_ylim(0, 40)
    ax.set_xlim(0,40)
    ax.plot(ax.get_xlim(),
            ax.get_ylim(),
            ls="--",
            c="blue",
            label="Expected prediction"
            )
    plt.axis('scaled')
    plt.title(f"Predicted vs real subject's age with"
              f" \n{model_name} model",
              fontsize=20)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.legend(loc="upper left", fontsize=14)
    anchored_text = AnchoredText(f"{dataframe_name} results:"
                                 f"\nMAE= {mae} [years]"
                                 f"\n MSE= {mse} [years^2]"
                                 f"\n PR= {pr}",
                                 loc=4,
                                 borderpad=0.,
                                 frameon=True,
                                 prop=dict(fontweight="bold"),
                                )
    ax.add_artist(anchored_text)


    plt.savefig(f"images/{dataframe_name}_{model_name}.png",
                dpi=200,
                format="png",
                bbox_inches="tight",
                )

    plt.show()

def residual_plot(true_age1,
                  pred_age1,
                  true_age2,
                  pred_age2,
                  model_name,
                  harm_flag):
    """
    Computes the difference(delta) between predicted age find with a
    specific model and true age on control test and ASD dataframes.

    Parameters
    ----------
    true_age1 : array-like
        Test feature from the first dataframe.

    pred_age1 : array-like
        Predicted feauture from the first dataframe.

    true_age2 : array-like
        Test feature from the second dataframe.

    pred_age2 : array-like
        Predicted feature from the second dataframe.

    model_name : string-like
        Name of the model used for prediction.

    harm_flag : boolean.
        Flag indicating if the dataframe on which prediction was performed
        has been previously harmonized.

    """

    if harm_flag is True:
        harm_status = "Harmonized"
    else:
        harm_status = "Unharmonized"

    plt.figure(figsize=(8, 8))
    plt.scatter(true_age1, pred_age1 - true_age1, c="b", label="Control")
    plt.scatter(true_age2, pred_age2 - true_age2, alpha=0.5, c="g", label="ASD")

    plt.axhline(
        y=(pred_age1 - true_age1).mean(),
        alpha=0.5,
        color='r',
        linestyle='-',
        label=f"Δ CTR mean:{round((pred_age1 - true_age1).mean(),3)}",
    )
    plt.axhline(
        y=(pred_age2 - true_age2).mean(),
        alpha=0.5,
        color='b',
        linestyle='-',
        label=f"Δ ASD mean:{round((pred_age2 - true_age2).mean(),3)}",
    )
    plt.xlabel("Ground truth Age [years]", fontsize=18)
    plt.ylabel("Delta Age [years]", fontsize=18)
    plt.title(
        f"Delta age versus ground truth age with \n{model_name}",
        fontsize=20,)
    plt.tick_params(axis="x", labelsize=18)
    plt.tick_params(axis="y", labelsize=18)
    plt.legend(loc="upper right", fontsize=14)

    plt.savefig(
        f"images/delta_pred_{model_name}_{harm_status}.png",
        dpi=200,
        format="png")

    plt.show()
