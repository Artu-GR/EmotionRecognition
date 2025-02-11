from tf_keras.models import Sequential
from tf_keras.layers import Dense, LSTM, Dropout, Bidirectional
from tf_keras.regularizers import l2
from tf_keras.optimizers import Adam


def create_model(input_shape, num_classes):      
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.4),
        LSTM(128),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    #optimizer = Adam(learning_rate=0.0001)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model