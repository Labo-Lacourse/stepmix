from stepmix.datasets import data_bakk_response, data_generation_gaussian, data_gaussian_binary


from stepmix.stepmix import StepMix, StepMixClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.base import clone
from matplotlib.colors import ListedColormap

cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
cm_custom = ListedColormap(["#a24253", "#629ac6"])
cm_pick = cm_custom
X, Y, gt = data_gaussian_binary(n_samples=2000, random_state=42)
Y = Y[:, 0]
model = StepMixClassifier(n_components=4, n_steps=1, measurement='gaussian_full',
                          structural='binary', random_state=43, verbose=0, max_iter=1, n_init=1)

# X, Y = make_circles(1000, noise=0.2, factor=0.5,  random_state=42)

# X, Y = make_moons(1000, noise=0.15, random_state=0)
# model = StepMixClassifier(n_components=6, n_steps=1, measurement='gaussian_diag',
#                           structural='binary', random_state=2, verbose=0, max_iter=1, n_init=1)

for i in range(1, 65):
    model_i = clone(model)
    model_i.set_params(max_iter=i)

    model_i.fit(X, Y)


    probs = model_i.predict_proba(X)
    predictions = model_i.predict(X)
    print("Accuracy", accuracy_score(Y, predictions))
    classes = model_i.predict_class(X)
    # probs = model.predict_proba_class(X, Y)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    axes = axes.flatten()

    axes[0].set_title(f"Observed Data")
    axes[0].scatter(*X.T, c=Y, cmap=cm_pick, edgecolors="k")

    axes[1].set_title(f"Clusters")
    axes[1].scatter(*X.T, c=classes, edgecolors="k")

    model_i.sample(11 * i)
    X_gen, Y_gen, _ = model_i.sample(1000)
    axes[2].set_title(f"Model Samples")
    axes[2].scatter(*X_gen.T, c=Y_gen, cmap=cm_pick, edgecolors="k")

    axes[3].set_title(f"Classifier")
    DecisionBoundaryDisplay.from_estimator(
        model_i, X, cmap=cm, alpha=0.8, ax=axes[3], eps=0.5
    )
    axes[3].scatter(
        X[:, 0], X[:, 1], c=Y, cmap=cm_pick, edgecolors="k"
    )

    for ax in axes:
        ax.set_axis_off()

    plt.savefig(f"temp/decision_boundary_{i}.png")
    plt.clf()
