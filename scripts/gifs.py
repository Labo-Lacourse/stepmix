import os
import shutil
import numpy as np

from stepmix.datasets import data_bakk_response, data_generation_gaussian, data_gaussian_binary


from stepmix.stepmix import StepMix, StepMixClassifier
from sklearn.datasets import make_circles, make_moons
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.base import clone
from matplotlib.colors import ListedColormap
from PIL import Image

cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
cm_custom = ListedColormap(["#a24253", "#629ac6"])
cm_gaussian = ListedColormap(["#bc5916", "#fefe98", "#bcacd2", "#7ec87e"])
cm_pick = cm_custom
X, Y, gt = data_gaussian_binary(n_samples=2000, random_state=42)
Y = Y[:, 0]
model = StepMixClassifier(n_components=4, n_steps=1, measurement='gaussian_full',
                          structural='binary', random_state=43, verbose=0, max_iter=1, n_init=1)

# X, Y = make_circles(1000, noise=0.2, factor=0.5,  random_state=42)

# X, Y = make_moons(1000, noise=0.15, random_state=0)
# model = StepMixClassifier(n_components=6, n_steps=1, measurement='gaussian_full',
#                           structural='binary', random_state=3, verbose=0, max_iter=1, n_init=1)
shutil.rmtree("animation.gif", ignore_errors=True)

images = []

for i in range(1, 70):
    model_i = clone(model)
    model_i.set_params(max_iter=i)

    model_i.fit(X, Y)


    probs = model_i.predict_proba(X)
    predictions = model_i.predict(X)
    print("Accuracy", accuracy_score(Y, predictions))
    classes = model_i.predict_class(X)
    # probs = model.predict_proba_class(X, Y)

    # fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    axes = axes.flatten()

    axes[0].set_title(f"Observed Data", fontsize=24)
    axes[0].scatter(*X.T, c=Y, cmap=cm_pick, edgecolors="k")

    axes[1].set_title(f"Clusters", fontsize=24)
    axes[1].scatter(*X.T, c=classes, edgecolors="k", cmap=cm_gaussian)

    model_i.sample(11 * i)
    X_gen, Y_gen, _ = model_i.sample(1000)
    axes[2].set_title(f"Model Samples", fontsize=24)
    axes[2].scatter(*X_gen.T, c=Y_gen, cmap=cm_pick, edgecolors="k")
    axes[2].set_xlim(axes[0].get_xlim())
    axes[2].set_ylim(axes[0].get_ylim())

    axes[3].set_title(f"Classifier", fontsize=24)
    DecisionBoundaryDisplay.from_estimator(
        model_i, X, cmap=cm, alpha=0.8, ax=axes[3], eps=0.5
    )
    axes[3].scatter(
        X[:, 0], X[:, 1], c=Y, cmap=cm_pick, edgecolors="k"
    )

    for ax in axes:
        ax.set_axis_off()


    # Add title at the bottom of the figure
    # fig.suptitle(f"StepMix EM Iter. {i:<2}", fontsize=24, y=0.05)
    fig.suptitle(f"StepMix EM Iter. {i:<2}", fontsize=24, y=0.08)

    # Tight layout but leave space for bottom title
    # fig.tight_layout(rect=[0, 0.06, 1, 1.])
    fig.tight_layout(rect=[0, 0.10, 1, 1.])


    plt.savefig(f"temp.png")
    # Open temp.png as a PIL image object
    image = Image.open("temp.png")
    images.append(image)

    # Remove temp.png
    os.remove("temp.png")

    # Close figure
    plt.close(fig)


# Convert images to gif
images[0].save('animation.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=75, loop=0)