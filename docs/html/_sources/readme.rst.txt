from litmodels.integrations import LightningModelCheckpoint

.. container::

   .. rubric:: Effortless Model Management for Your Development ⚡
      :name: effortless-model-management-for-your-development

   Effortless management for your ML models.

   🚀 `Quick start <#quick-start>`__ 📦
   `Examples <#saving-and-loading-models>`__ 📚
   `Documentation <https://lightning.ai/docs/overview/model-registry>`__
   💬 `Get help on Discord <https://discord.com/invite/XncpTy7DSt>`__ 📋
   `License: Apache
   2.0 <https://github.com/Lightning-AI/models/blob/main/LICENSE>`__

--------------

**Lightning Models** is a streamlined toolkit for effortlessly saving,
loading, and managing your model checkpoints. Designed to simplify the
entire model lifecycle—from training and inference to sharing,
deployment, and cloud integration—Lightning Models supports any
framework that produces model checkpoints, including but not limited to
PyTorch Lightning.

.. raw:: html

   <pre>
   ✅ Seamless Model Saving & Loading
   ✅ Robust Checkpoint Management
   ✅ Cloud Integration Out of the Box
   ✅ Versatile Across Frameworks
   </pre>

Quick start
===========

Install Lightning Models via pip (more installation options below):

.. code:: bash

   pip install -U litmodels

Or install directly from source:

.. code:: bash

   pip install https://github.com/Lightning-AI/models/archive/refs/heads/main.zip

Saving and Loading Models
-------------------------

Lightning Models offers a simple API to manage your model checkpoints.
Train your model using your preferred framework (our fist examples show
``scikit-learn``) and then save your best checkpoint with a single
function call.

Train scikit-learn model and save it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from sklearn import datasets, model_selection, svm
   from litmodels import upload_model

   # Unique model identifier: <organization>/<teamspace>/<model-name>
   MY_MODEL_NAME = "your_org/your_team/sklearn-svm-model"

   # Load example dataset
   iris = datasets.load_iris()
   X, y = iris.data, iris.target

   # Split dataset into training and test sets
   X_train, X_test, y_train, y_test = model_selection.train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   # Train a simple SVC model
   model = svm.SVC()
   model.fit(X_train, y_train)

   # Upload the saved model using litmodels
   upload_model(model=model, name=MY_MODEL_NAME)

Download and Load the Model for inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from litmodels import load_model

   # Unique model identifier: <organization>/<teamspace>/<model-name>
   MY_MODEL_NAME = "your_org/your_team/sklearn-svm-model"

   # Download and load the model file from cloud storage
   model = load_model(name=MY_MODEL_NAME, download_dir="my_models")

   # Example: run inference with the loaded model
   sample_input = [[5.1, 3.5, 1.4, 0.2]]
   prediction = model.predict(sample_input)
   print(f"Prediction: {prediction}")

Saving and Loading Models with Pytorch Lightning
------------------------------------------------

Next examples demonstrate seamless PyTorch Lightning integration with
Lightning Models.

Train a simple Lightning model and save it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from lightning import Trainer
   from litmodels import upload_model
   from litmodels.demos import BoringModel

   # Define the model name - this should be unique to your model
   MY_MODEL_NAME = "<organization>/<teamspace>/<model-name>"

   # Configure Lightning Trainer
   trainer = Trainer(max_epochs=2)
   # Define the model and train it
   trainer.fit(BoringModel())

   # Upload the best model to cloud storage
   checkpoint_path = getattr(trainer.checkpoint_callback, "best_model_path")
   upload_model(model=checkpoint_path, name=MY_MODEL_NAME)

Download and Load the Model for fine-tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from lightning import Trainer
   from litmodels import download_model
   from litmodels.demos import BoringModel

   # Define the model name - this should be unique to your model
   MY_MODEL_NAME = "<organization>/<teamspace>/<model-name>:<model-version>"

   # Load the model from cloud storage
   checkpoint_path = download_model(name=MY_MODEL_NAME, download_dir="my_models")
   print(f"model: {checkpoint_path}")

   # Train the model with extended training period
   trainer = Trainer(max_epochs=4)
   trainer.fit(BoringModel(), ckpt_path=checkpoint_path)

.. raw:: html

   <details>

.. raw:: html

   <summary>

Checkpointing Workflow with Lightning

.. raw:: html

   </summary>

Enhance your training process with an automatic checkpointing callback
that uploads the model at the end of each epoch.

.. code:: python

   import torch.utils.data as data
   import torchvision as tv
   from lightning import Trainer
   from litmodels.integrations import LightningModelCheckpoint
   from litmodels.demos import BoringModel

   # Define the model name - this should be unique to your model
   MY_MODEL_NAME = "<organization>/<teamspace>/<model-name>"

   dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())
   train, val = data.random_split(dataset, [55000, 5000])

   trainer = Trainer(
       max_epochs=2,
       callbacks=[LightningModelCheckpoint(model_name=MY_MODEL_NAME)],
   )
   trainer.fit(
       BoringModel(),
       data.DataLoader(train, batch_size=256),
       data.DataLoader(val, batch_size=256),
   )

.. raw:: html

   </details>
