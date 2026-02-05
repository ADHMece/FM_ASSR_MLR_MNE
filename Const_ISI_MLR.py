import matplotlib.pyplot as plt
import numpy as np

def view_classify(img, ps, classes=None):
    """
    img : torch tensor of shape (1,28,28)
    ps : predicted probabilities (softmax output) for all classes
    classes : list of class labels, optional
    """
    img = img.squeeze()  # remove channel dim if needed
    ps = ps.detach().numpy().squeeze()  # convert to numpy

    if classes is None:
        classes = [str(i) for i in range(len(ps))]

    fig, (ax1, ax2) = plt.subplots(figsize=(6,3), ncols=2)
    # Plot image
    ax1.imshow(img, cmap='gray')
    ax1.axis('off')

    # Plot probabilities
    ax2.barh(np.arange(len(ps)), ps)
    ax2.set_yticks(np.arange(len(ps)))
    ax2.set_yticklabels(classes)
    ax2.invert_yaxis()  # largest probs on top
    plt.show()