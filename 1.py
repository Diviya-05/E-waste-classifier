import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

# Paths to your dataset
train_dir = "kaggledataset/modified-dataset/train"
val_dir = "kaggledataset/modified-dataset/val"

# Image preprocessing
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=16, class_mode='categorical')
val_data = datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=16, class_mode='categorical')

# Base model (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train it
model.fit(train_data, validation_data=val_data, epochs=5)

# Save the model
model.save("e_waste_classifier.h5")
print("✅ Model trained and saved as 'e_waste_classifier.h5'")
