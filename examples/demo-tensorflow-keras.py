from tensorflow import keras

from litmodels import upload_model, load_model


if __name__ == "__main__":
    # Define the model
    model = keras.Sequential([
        keras.layers.Dense(10, input_shape=(784,), name='dense_1'),
        keras.layers.Dense(10, name='dense_2')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Save the model
    upload_model("lightning-ai/jirka/sample-tf-keras-model", model=model)

    # Load the model
    model_ = load_model("lightning-ai/jirka/sample-tf-keras-model", download_dir="./my-model")