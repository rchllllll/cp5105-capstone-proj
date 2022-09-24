{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": [],
      "mount_file_id": "1JlQW6YAwVMYlLE2i7RIRqKupDP-5rQVj",
      "authorship_tag": "ABX9TyO6/fmWQjm9pupRCLLOFr68"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "# reference: https://stackoverflow.com/questions/48905127/importing-py-files-in-google-colab\n",
        "!cp /content/drive/MyDrive/colab/dataset.py .\n",
        "!cp /content/drive/MyDrive/colab/config.py .\n",
        "!cp /content/drive/MyDrive/colab/model.py ."
      ],
      "metadata": {
        "id": "qAwwD2xbDvU-"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "id": "H1TzqjUK-oVH"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import math\n",
        "import random\n",
        "import matplotlib.pyplot as plt\n",
        "import time\n",
        "import pickle\n",
        "import os\n",
        "from pathlib import Path\n",
        "import time \n",
        "import collections\n",
        "\n",
        "import torch\n",
        "import torchvision\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "import torch.optim as optim\n",
        "from torchvision.io import read_image\n",
        "import torchvision.transforms as transforms\n",
        "import torchvision.transforms as T\n",
        "from torch.utils.data import Dataset, DataLoader \n",
        "from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights\n",
        "\n",
        "from dataset import SiameseDataset\n",
        "from model import SiameseModel\n",
        "\n",
        "# to ensure reproducibility\n",
        "torch.manual_seed(0)\n",
        "random.seed(0)\n",
        "np.random.seed(0)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "f = open('/content/drive/MyDrive/capstone/data/train_dataset_b16_lr0.0001_num1028_emb100.pickle', 'rb')\n",
        "train_dataset = pickle.load(f)"
      ],
      "metadata": {
        "id": "NEUeFmMA_BoA"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "rSFYZgC-Fj05"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}